#!/usr/bin/env python3
"""
load_with_staging.py - Production-Grade NTSB Historical Data Loader with Staging Tables

Phase 1 Sprint 2: Historical Data Integration with Deduplication
Version: 2.0.0
Date: 2025-11-06

This loader implements the staging table pattern for handling duplicate events
across multiple historical NTSB databases:
- avall.mdb (2008-2025): Current data, monthly updates
- Pre2008.mdb (2000-2007): Historical data, static
- PRE1982.MDB (1962-1981): Legacy data, static, different schema

Architecture:
1. Extract CSV from MDB → /data/{source}/*.csv
2. Bulk COPY into staging tables (no constraints)
3. Identify duplicates between staging and production
4. Merge only NEW events into production
5. Load ALL child records (even for duplicate events)
6. Update load_tracking table
7. Generate detailed load report

Usage:
    python scripts/load_with_staging.py --source Pre2008.mdb
    python scripts/load_with_staging.py --source PRE1982.MDB --force
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2
import logging
import subprocess
from typing import List, Tuple, Optional
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_with_staging.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Database configuration
# Support environment variables for containerized deployments
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ntsb_aviation"),
    "user": os.getenv("DB_USER", os.getenv("USER")),
    "password": os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", "")),
}

# Table load order (respects foreign keys)
TABLE_ORDER = [
    "events",  # Parent table (no foreign keys)
    "aircraft",  # References events
    "Flight_Crew",  # References events + aircraft
    "injury",  # References events
    "Findings",  # References events
    "Occurrences",  # References events
    "seq_of_events",  # References events
    "Events_Sequence",  # References events
    "engines",  # References events + aircraft
    "narratives",  # References events
    "NTSB_Admin",  # References events
]


def convert_ntsb_time_to_postgres(time_value):
    """
    Convert NTSB integer time format (HHMM) to PostgreSQL TIME format (HH:MM:SS).

    NTSB stores times as 4-digit integers representing 24-hour time without colon:
    - 0 = 00:00 (midnight)
    - 825 = 08:25 (8:25 AM)
    - 1430 = 14:30 (2:30 PM)
    - 2359 = 23:59 (11:59 PM)

    Examples:
        825 → "08:25:00"
        1430 → "14:30:00"
        0 → "00:00:00"
        2359 → "23:59:00"
        NaN/None → None
        9999 → None (invalid hour)

    Args:
        time_value: Integer or float representing HHMM format (e.g., 825.0 from pandas)

    Returns:
        str: TIME in HH:MM:SS format, or None if invalid/missing
    """
    # Handle missing values
    if pd.isna(time_value) or time_value == "":
        return None

    try:
        # Convert to integer (pandas reads as float: 825.0 → 825)
        time_int = int(float(time_value))

        # Extract hours and minutes from HHMM format
        hours = time_int // 100  # 825 // 100 = 8
        minutes = time_int % 100  # 825 % 100 = 25

        # Validate ranges (24-hour format)
        if hours < 0 or hours > 23:
            return None  # Invalid hour
        if minutes < 0 or minutes > 59:
            return None  # Invalid minute

        # Format as HH:MM:SS (PostgreSQL TIME format)
        return f"{hours:02d}:{minutes:02d}:00"

    except (ValueError, TypeError):
        # Handle non-numeric values
        return None


class StagingTableLoader:
    """Load historical NTSB data using staging tables for deduplication."""

    # Date columns by table
    DATE_COLUMNS = {
        "events": ["ev_date"],
        "Flight_Crew": ["pilot_med_date"],
        "NTSB_Admin": ["invest_start_date", "report_date"],
    }

    # Numeric columns by table (all numeric types: INTEGER, NUMERIC, DECIMAL)
    NUMERIC_COLUMNS = {
        "events": [
            "inj_tot_f",
            "inj_tot_s",
            "inj_tot_m",
            "inj_tot_n",
            "wx_temp",
            "wx_wind_dir",
            "wx_wind_speed",
            "wx_vis",
            "ev_nr_apt_dist",
            "dec_latitude",
            "dec_longitude",
        ],
        "aircraft": ["cert_max_gr_wt", "num_eng"],
        "Flight_Crew": [
            "crew_age",
            "pilot_tot_time",
            "pilot_make_time",
            "pilot_90_days",
            "pilot_30_days",
            "pilot_24_hrs",
        ],
        "injury": ["inj_person_count"],
        "seq_of_events": ["seq_event_no", "altitude"],
        "Events_Sequence": ["sequence_number"],
        "engines": ["eng_hp_or_lbs"],
    }

    # INTEGER columns (subset of NUMERIC_COLUMNS that need float-to-int conversion)
    # These columns are defined as INTEGER in PostgreSQL schema and cannot accept decimal points
    # Extracted from scripts/schema.sql
    INTEGER_COLUMNS = {
        "events": [
            "ev_year",
            "ev_month",  # Generated by trigger, but included for safety
            "inj_tot_f",
            "inj_tot_s",
            "inj_tot_m",
            "inj_tot_n",
            "wx_temp",
            "wx_wind_dir",
            "wx_wind_speed",
        ],
        "aircraft": ["cert_max_gr_wt", "num_eng"],
        "Flight_Crew": [
            "crew_age",
            "pilot_tot_time",
            "pilot_make_time",
            "pilot_90_days",
            "pilot_30_days",
            "pilot_24_hrs",
        ],
        "injury": ["inj_person_count"],
        "seq_of_events": ["seq_event_no", "altitude"],
        "Events_Sequence": ["sequence_number"],
        "engines": ["eng_hp_or_lbs"],
    }

    # TIME columns need HHMM integer → HH:MM:SS string conversion
    # NTSB stores times as 4-digit integers (e.g., 825 = 8:25 AM, 1430 = 2:30 PM)
    # PostgreSQL TIME columns require HH:MM:SS format
    TIME_COLUMNS = {
        "events": ["ev_time"],
        # Add other tables here if TIME columns are discovered
    }

    def __init__(self, source_db: str, mdb_path: str, force_reload: bool = False):
        """
        Initialize loader.

        Args:
            source_db: Database name ('avall.mdb', 'Pre2008.mdb', or 'PRE1982.MDB')
            mdb_path: Full path to MDB file
            force_reload: Allow reloading already-loaded databases (dangerous!)
        """
        self.source_db = source_db
        self.mdb_path = Path(mdb_path)
        self.force_reload = force_reload
        self.csv_dir = (
            Path("data") / source_db.replace(".", "_").replace(" ", "_").lower()
        )
        self.conn = None
        self.cursor = None

        # Statistics
        self.stats = {
            "events_in_staging": 0,
            "events_in_production": 0,
            "duplicate_events": 0,
            "new_events_added": 0,
            "child_records_loaded": {},
            "start_time": datetime.now(),
            "end_time": None,
        }

        if not self.mdb_path.exists():
            raise FileNotFoundError(f"MDB file not found: {mdb_path}")

        logger.info(f"Initialized StagingTableLoader for {source_db}")
        logger.info(f"  MDB path: {self.mdb_path}")
        logger.info(f"  CSV dir: {self.csv_dir}")
        logger.info(f"  Force reload: {force_reload}")

    def connect_db(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            logger.info("✓ Database connected")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def check_load_status(self) -> str:
        """Check if this database has already been loaded."""
        self.cursor.execute(
            """
            SELECT load_status, load_completed_at
            FROM load_tracking
            WHERE database_name = %s
        """,
            (self.source_db,),
        )

        result = self.cursor.fetchone()
        if result:
            status, completed_at = result
            if status == "completed" and completed_at:
                # Allow avall.mdb to be re-loaded for monthly updates (idempotent)
                # Block historical databases (Pre2008, PRE1982) unless --force is used
                if self.source_db == "avall.mdb":
                    logger.info(
                        f"ℹ {self.source_db} was previously loaded on {completed_at}"
                    )
                    logger.info(
                        "ℹ This is a monthly update - proceeding with idempotent re-load"
                    )
                    logger.info("ℹ Duplicate events will be skipped automatically")
                else:
                    logger.error(f"✗ {self.source_db} already loaded on {completed_at}")
                    logger.error(
                        "✗ Historical databases (PRE1982, Pre2008) should only load ONCE"
                    )
                    logger.error("✗ Only avall.mdb should be periodically updated")

                    if not self.force_reload:
                        logger.error("✗ Use --force flag to override (dangerous!)")
                        sys.exit(1)
                    else:
                        logger.warning(
                            "⚠ --force flag detected, proceeding with re-load"
                        )
            return status
        else:
            logger.warning(f"⚠ {self.source_db} not found in load_tracking table")
            return "unknown"

    def update_load_status(self, status: str, **kwargs):
        """Update load tracking table."""
        fields = ["load_status = %s"]
        values = [status]

        if status == "in_progress":
            fields.append("load_started_at = %s")
            values.append(datetime.now())
        elif status == "completed":
            fields.append("load_completed_at = %s")
            values.append(datetime.now())

            # Calculate duration
            if self.stats["start_time"]:
                duration = (datetime.now() - self.stats["start_time"]).total_seconds()
                fields.append("load_duration_seconds = %s")
                values.append(int(duration))

        for key, value in kwargs.items():
            fields.append(f"{key} = %s")
            values.append(value)

        query = f"UPDATE load_tracking SET {', '.join(fields)} WHERE database_name = %s"
        values.append(self.source_db)

        self.cursor.execute(query, values)
        self.conn.commit()

    def extract_csv(self, table_name: str) -> Optional[Path]:
        """Extract table from MDB to CSV."""
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_dir / f"{table_name}.csv"

        logger.info(f"Extracting {table_name} from {self.mdb_path.name}...")

        try:
            with open(csv_path, "w") as f:
                subprocess.run(
                    ["mdb-export", "-D", "%Y-%m-%d", str(self.mdb_path), table_name],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=300,
                )

            row_count = sum(1 for _ in open(csv_path)) - 1
            if row_count == 0:
                logger.warning("  ⊘ Empty table, skipping")
                return None

            logger.info(f"  ✓ Extracted {row_count:,} rows to {csv_path.name}")
            return csv_path

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Extraction failed: {e.stderr.decode()}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("  ✗ Extraction timeout (>5 minutes)")
            return None

    def clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Apply data cleaning and validation."""
        original_count = len(df)

        # Lowercase column names
        df.columns = df.columns.str.strip().str.lower()

        # Clean common issues
        df = df.replace(
            {"": None, "UNK": None, "UNKN": None, "NONE": None, "nan": None}
        )

        # Handle legacy Cause_Factor → cm_inPC conversion (Pre2008 databases)
        if table_name == "Findings" and "cause_factor" in df.columns:
            logger.info("  Converting legacy Cause_Factor to cm_inPC (boolean)")
            df["cm_inpc"] = df["cause_factor"].apply(
                lambda x: True
                if (pd.notna(x) and str(x).strip().upper() == "C")
                else (
                    False if (pd.notna(x) and str(x).strip().upper() == "F") else None
                )
            )
            df = df.drop(columns=["cause_factor"])

        # Convert date columns
        if table_name in self.DATE_COLUMNS:
            for col in self.DATE_COLUMNS[table_name]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numeric columns
        if table_name in self.NUMERIC_COLUMNS:
            for col in self.NUMERIC_COLUMNS[table_name]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert INTEGER columns from float to int (PostgreSQL INTEGER cannot accept "0.0" format)
        # This must happen AFTER pd.to_numeric() conversion and BEFORE CSV write
        if table_name in self.INTEGER_COLUMNS:
            for col in self.INTEGER_COLUMNS[table_name]:
                if (
                    col in df.columns
                    and col in df.select_dtypes(include=["number"]).columns
                ):
                    # Convert float to int, preserving NaN/NULL as None
                    # Must use Int64 (nullable integer dtype) to prevent pandas from converting back to float64
                    df[col] = (
                        df[col]
                        .apply(lambda x: int(x) if pd.notna(x) and x != "" else pd.NA)
                        .astype("Int64")
                    )

        # Convert TIME columns (HHMM integer format → HH:MM:SS string)
        # NTSB stores times as integers (e.g., 825 = 8:25 AM)
        # PostgreSQL TIME columns require HH:MM:SS format (e.g., "08:25:00")
        if table_name in self.TIME_COLUMNS:
            time_cols = [
                col for col in self.TIME_COLUMNS[table_name] if col in df.columns
            ]
            if time_cols:
                logger.info(f"  Converting TIME columns: {time_cols}")
                for col in time_cols:
                    # Apply HHMM → HH:MM:SS conversion
                    df[col] = df[col].apply(convert_ntsb_time_to_postgres)

                    # Log conversion statistics
                    valid_count = df[col].notna().sum()
                    total_count = len(df)
                    logger.info(
                        f"    ✓ {col}: {valid_count:,}/{total_count:,} valid times converted"
                    )

        # Table-specific cleaning
        if table_name == "events":
            # Remove rows with NULL ev_date
            if "ev_date" in df.columns:
                df = df[df["ev_date"].notna()]

            # Date validation
            current_year = datetime.now().year
            df = df[
                (df["ev_date"].dt.year >= 1962)
                & (df["ev_date"].dt.year <= current_year + 1)
            ]

            # Coordinate validation
            if "dec_latitude" in df.columns:
                df.loc[
                    (df["dec_latitude"] < -90) | (df["dec_latitude"] > 90),
                    "dec_latitude",
                ] = None

            if "dec_longitude" in df.columns:
                df.loc[
                    (df["dec_longitude"] < -180) | (df["dec_longitude"] > 180),
                    "dec_longitude",
                ] = None

        elif table_name == "Flight_Crew":
            # Age validation (10-120 years)
            if "crew_age" in df.columns:
                invalid_mask = (df["crew_age"].notna()) & (
                    (df["crew_age"] < 10) | (df["crew_age"] > 120)
                )
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    logger.warning(
                        f"  ⚠ Converting {invalid_count} invalid crew ages to NULL"
                    )
                df.loc[invalid_mask, "crew_age"] = None

        elif table_name == "Occurrences":
            # Clean occurrence codes (convert "0" and invalid codes to NULL)
            if "occurrence_code" in df.columns:
                df["occurrence_code"] = pd.to_numeric(
                    df["occurrence_code"], errors="coerce"
                )
                df.loc[
                    (df["occurrence_code"] <= 0) | (df["occurrence_code"] > 999),
                    "occurrence_code",
                ] = None

        # Trim whitespace from string columns (skip boolean columns)
        for col in df.select_dtypes(include=["object"]).columns:
            # Skip if column is actually boolean (e.g., cm_inpc after conversion)
            if df[col].dtype != "object" or col in ["cm_inpc"]:
                continue
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                # Column is not string type, skip
                pass

        cleaned_count = len(df)
        if cleaned_count < original_count:
            logger.warning(
                f"  ⚠ Removed {original_count - cleaned_count} invalid rows during cleaning"
            )

        return df

    def bulk_copy_to_staging(self, table_name: str, df: pd.DataFrame):
        """Bulk load DataFrame into staging table using COPY."""
        staging_table = f"staging.{table_name.lower()}"

        # Align columns with staging table schema
        self.cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'staging' AND table_name = %s
            ORDER BY ordinal_position
        """,
            (table_name.lower(),),
        )

        db_columns = [row[0] for row in self.cursor.fetchall()]
        df_columns = [col for col in db_columns if col in df.columns]

        df = df[df_columns]
        df = df.where(pd.notnull(df), None)

        # Use COPY for fast bulk load
        output = StringIO()
        df.to_csv(output, sep="\t", header=False, index=False, na_rep="\\N")
        output.seek(0)

        columns_str = ", ".join(df_columns)

        try:
            self.cursor.copy_expert(
                f"COPY {staging_table} ({columns_str}) FROM STDIN WITH CSV DELIMITER E'\\t' NULL '\\N'",
                output,
            )
            self.conn.commit()
            logger.info(f"  ✓ Loaded {len(df):,} rows into {staging_table}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"  ✗ Failed to load {staging_table}: {e}")
            raise

    def identify_duplicates(self) -> Tuple[List[str], List[str]]:
        """
        Identify duplicate vs unique events.

        Returns:
            (duplicate_ev_ids, unique_ev_ids)
        """
        logger.info("Analyzing event duplicates...")

        # Get all ev_ids from staging
        self.cursor.execute("SELECT ev_id FROM staging.events")
        staging_ids = {row[0] for row in self.cursor.fetchall()}
        self.stats["events_in_staging"] = len(staging_ids)

        # Get all ev_ids from production
        self.cursor.execute("SELECT ev_id FROM public.events")
        production_ids = {row[0] for row in self.cursor.fetchall()}
        self.stats["events_in_production"] = len(production_ids)

        # Identify duplicates and unique events
        duplicate_ids = staging_ids & production_ids
        unique_ids = staging_ids - production_ids

        self.stats["duplicate_events"] = len(duplicate_ids)
        self.stats["new_events_added"] = len(unique_ids)

        logger.info(f"  Events in staging: {len(staging_ids):,}")
        logger.info(f"  Events in production: {len(production_ids):,}")
        logger.info(f"  Duplicates found: {len(duplicate_ids):,}")
        logger.info(f"  New unique events: {len(unique_ids):,}")

        return list(duplicate_ids), list(unique_ids)

    def merge_unique_events(self, unique_ids: List[str]):
        """Insert only unique events into production tables."""
        if not unique_ids:
            logger.info("No new unique events to merge")
            return

        logger.info(f"Merging {len(unique_ids):,} unique events into production...")

        # Get insertable columns (excludes generated columns like location_geom)
        columns = self._get_insertable_columns("events")
        column_list = ", ".join(columns)

        # Insert unique events
        self.cursor.execute(
            f"""
            INSERT INTO public.events ({column_list})
            SELECT {column_list} FROM staging.events
            WHERE ev_id = ANY(%s)
        """,
            (unique_ids,),
        )

        rows_inserted = self.cursor.rowcount
        self.conn.commit()

        logger.info(f"  ✓ Inserted {rows_inserted:,} new events")

    def load_child_table(self, table_name: str):
        """
        Load child table records from staging.
        Handles both new events and duplicates properly.
        """
        logger.info(f"Loading {table_name} records...")

        # Get staging row count
        self.cursor.execute(f"SELECT COUNT(*) FROM staging.{table_name.lower()}")
        staging_count = self.cursor.fetchone()[0]

        if staging_count == 0:
            logger.info(f"  ⊘ No records in staging.{table_name.lower()}")
            return

        # Get insertable columns (excludes generated columns like search_vector)
        columns = self._get_insertable_columns(table_name)
        column_list = ", ".join(columns)
        # Prefix columns with 's.' for qualified references in JOIN
        qualified_column_list = ", ".join([f"s.{col}" for col in columns])

        # Determine primary key for conflict resolution
        pk_columns = self._get_primary_key_columns(table_name)

        if not pk_columns:
            logger.warning(
                f"  ⚠ No primary key found for {table_name}, using basic INSERT"
            )
            self.cursor.execute(f"""
                INSERT INTO public.{table_name.lower()} ({column_list})
                SELECT {column_list} FROM staging.{table_name.lower()}
            """)
        else:
            pk_conflict = ", ".join(pk_columns)

            # Use ON CONFLICT DO NOTHING to skip duplicates
            # Qualify column names with 's.' to avoid ambiguity when joining
            self.cursor.execute(f"""
                INSERT INTO public.{table_name.lower()} ({column_list})
                SELECT {qualified_column_list} FROM staging.{table_name.lower()} s
                INNER JOIN public.events e ON s.ev_id = e.ev_id
                ON CONFLICT ({pk_conflict}) DO NOTHING
            """)

        rows_inserted = self.cursor.rowcount
        self.conn.commit()

        self.stats["child_records_loaded"][table_name] = rows_inserted
        logger.info(f"  ✓ Loaded {rows_inserted:,} {table_name} records")

    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        """Get primary key column names for a table."""
        self.cursor.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
        """,
            (f"public.{table_name.lower()}",),
        )

        return [row[0] for row in self.cursor.fetchall()]

    def _get_insertable_columns(self, table_name: str) -> List[str]:
        """
        Get list of columns that can be inserted into (excludes generated columns).

        Generated columns are automatically computed by PostgreSQL and cannot be
        explicitly inserted. This method queries information_schema to identify
        non-generated columns.

        Args:
            table_name: Table name to get columns for

        Returns:
            List of column names (non-generated only), ordered by position
        """
        self.cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND is_generated = 'NEVER'
            ORDER BY ordinal_position
        """,
            (table_name.lower(),),
        )

        columns = [row[0] for row in self.cursor.fetchall()]
        logger.debug(
            f"Table {table_name}: {len(columns)} insertable columns (excluding generated)"
        )
        return columns

    def cleanup_staging(self):
        """Truncate staging tables after successful load."""
        logger.info("Cleaning up staging tables...")

        self.cursor.execute("SELECT staging.truncate_all_tables()")
        self.conn.commit()

        logger.info("  ✓ Staging tables truncated")

    def generate_load_report(self):
        """Generate detailed load statistics report."""
        duration = (datetime.now() - self.stats["start_time"]).total_seconds()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             NTSB Historical Data Load Report                 ║
╚══════════════════════════════════════════════════════════════╝

Database: {self.source_db}
Path: {self.mdb_path}
Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)

EVENT STATISTICS:
  Events in staging:      {self.stats["events_in_staging"]:>8,}
  Events in production:   {self.stats["events_in_production"]:>8,}
  Duplicate events:       {self.stats["duplicate_events"]:>8,}
  New unique events:      {self.stats["new_events_added"]:>8,}

CHILD RECORDS LOADED:
"""
        for table, count in self.stats["child_records_loaded"].items():
            report += f"  {table:20s}  {count:>8,}\n"

        total_child = sum(self.stats["child_records_loaded"].values())
        report += f"\nTotal child records:    {total_child:>8,}\n"
        report += f"Grand total loaded:     {self.stats['new_events_added'] + total_child:>8,}\n"

        logger.info(report)

        # Save to file
        report_file = (
            Path("docs") / f"{self.source_db.replace('.', '_')}_load_report.txt"
        )
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(report)
        logger.info(f"Report saved to {report_file}")

    def run(self):
        """Execute complete load process."""
        logger.info("=" * 70)
        logger.info(f"NTSB Historical Data Loader - {self.source_db}")
        logger.info("=" * 70)

        try:
            # Step 1: Database connection and status check
            self.connect_db()
            self.check_load_status()  # Validates database is ready to load
            self.update_load_status("in_progress")

            # Step 2: Extract and load to staging
            for table in TABLE_ORDER:
                logger.info(f"\n--- Processing {table} ---")

                # Extract CSV
                csv_path = self.extract_csv(table)
                if not csv_path:
                    continue

                # Read and clean
                df = pd.read_csv(csv_path, low_memory=False)
                df = self.clean_dataframe(df, table)

                # Load to staging
                self.bulk_copy_to_staging(table, df)

            # Step 3: Identify duplicates
            logger.info("\n--- Analyzing Duplicates ---")
            duplicate_ids, unique_ids = self.identify_duplicates()

            # Step 4: Merge unique events
            logger.info("\n--- Merging Unique Events ---")
            self.merge_unique_events(unique_ids)

            # Step 5: Load child records (for ALL events, including duplicates)
            logger.info("\n--- Loading Child Tables ---")
            for table in TABLE_ORDER[1:]:  # Skip 'events' (already merged)
                self.load_child_table(table)

            # Step 6: Cleanup
            self.cleanup_staging()

            # Step 7: Update tracking
            self.update_load_status(
                "completed",
                events_loaded=self.stats["new_events_added"],
                total_rows_loaded=self.stats["new_events_added"]
                + sum(self.stats["child_records_loaded"].values()),
                duplicate_events_found=self.stats["duplicate_events"],
            )

            # Step 8: Generate report
            self.generate_load_report()

            logger.info("\n✓ Load completed successfully!")

        except Exception as e:
            logger.error(f"\n✗ Load failed: {e}", exc_info=True)
            self.update_load_status("failed", notes=str(e))
            raise

        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load NTSB historical data with staging tables"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source database name (e.g., Pre2008.mdb, PRE1982.MDB)",
    )
    parser.add_argument("--mdb-path", help="Override MDB file path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reload even if already completed (dangerous!)",
    )

    args = parser.parse_args()

    # Default paths
    if args.mdb_path:
        mdb_path = args.mdb_path
    else:
        mdb_path = f"datasets/{args.source}"

    # Execute load
    loader = StagingTableLoader(args.source, mdb_path, force_reload=args.force)
    loader.run()


if __name__ == "__main__":
    main()
