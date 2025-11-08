#!/usr/bin/env python3
"""
load_historical_data.py - Load historical CSV data into PostgreSQL database

Phase 1 Sprint 2: Historical Data Integration
Version: 1.0.0
Date: 2025-11-06

Handles legacy schema differences:
- Pre2008: Uses Cause_Factor instead of cm_inPC in Findings table
- Different date formats and field mappings

Usage:
    python scripts/load_historical_data.py --source pre2008
    python scripts/load_historical_data.py --source pre1982
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_historical_data.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Load historical CSV data into PostgreSQL database"""

    # Tables in dependency order (respect foreign keys)
    TABLES = [
        "events",
        "aircraft",
        "Flight_Crew",
        "injury",
        "Findings",
        "Occurrences",
        "seq_of_events",
        "Events_Sequence",
        "engines",
        "narratives",
        "NTSB_Admin",
    ]

    # Column mappings and transformations
    DATE_COLUMNS = {
        "events": ["ev_date", "pilot_med_date"],
        "Flight_Crew": ["pilot_med_date"],
        "NTSB_Admin": ["invest_start_date", "report_date"],
    }

    NUMERIC_COLUMNS = {
        "events": [
            "inj_tot_f",
            "inj_tot_s",
            "inj_tot_m",
            "inj_tot_n",
            "inj_f_grnd",
            "inj_m_grnd",
            "inj_s_grnd",
            "inj_tot_t",
            "wx_temp",
            "wx_dew_pt",
            "wind_dir_deg",
            "wind_vel_kts",
            "gust_kts",
            "wx_dens_alt",
            "ev_nr_apt_dist",
            "dec_latitude",
            "dec_longitude",
        ],
        "aircraft": ["cert_max_gr_wt", "num_eng", "Aircraft_Key"],
        "Flight_Crew": [
            "crew_age",
            "pilot_tot_time",
            "pilot_make_time",
            "pilot_90_days",
            "pilot_30_days",
            "pilot_24_hrs",
            "Aircraft_Key",
            "crew_no",
        ],
        "injury": ["inj_person_count", "Aircraft_Key"],
        "seq_of_events": ["seq_event_no", "altitude", "Aircraft_Key"],
        "Events_Sequence": ["sequence_number", "Aircraft_Key"],
        "engines": ["eng_hp_or_lbs", "Aircraft_Key"],
        "Findings": ["Aircraft_Key"],
        "Occurrences": ["Aircraft_Key"],
        "narratives": ["Aircraft_Key"],
    }

    # Legacy column mappings (old name -> new name)
    LEGACY_COLUMN_MAP = {
        "Findings": {
            "Cause_Factor": "cm_inPC",  # Map to boolean
        }
    }

    def __init__(
        self,
        data_dir: str = "data/pre2008",
        db_name: str = "ntsb_aviation",
        db_user: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.db_name = db_name
        self.db_user = db_user or os.getenv("USER")
        self.load_stats = {}
        self.is_legacy = "pre2008" in str(data_dir) or "pre1982" in str(data_dir)

        # Create SQLAlchemy engine
        self.engine = create_engine(
            f"postgresql://{self.db_user}@localhost/{self.db_name}"
        )

        logger.info(f"Initialized loader for database: {self.db_name}")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Legacy mode: {self.is_legacy}")

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                logger.info(f"✓ Database connected: {version[:50]}...")
                return True
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False

    def convert_cause_factor_to_boolean(self, value):
        """Convert Cause_Factor (C/F/empty) to boolean for cm_inPC"""
        if pd.isna(value) or value == "":
            return None
        # C = Cause (in probable cause), F = Factor (contributing), empty = neither
        return value.strip().upper() == "C"

    def clean_dataframe(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Clean and transform dataframe"""

        # Clean column names (lowercase, remove spaces)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Handle legacy column mappings
        if self.is_legacy and table in self.LEGACY_COLUMN_MAP:
            for old_col, new_col in self.LEGACY_COLUMN_MAP[table].items():
                old_col_lower = old_col.lower()
                if old_col_lower in df.columns:
                    logger.info(f"  Mapping legacy column {old_col_lower} -> {new_col}")
                    if table == "Findings" and old_col == "Cause_Factor":
                        # Convert C/F to boolean
                        df[new_col] = df[old_col_lower].apply(
                            self.convert_cause_factor_to_boolean
                        )
                        df = df.drop(columns=[old_col_lower])
                    else:
                        df = df.rename(columns={old_col_lower: new_col})

        # Convert date columns
        if table in self.DATE_COLUMNS:
            for col in self.DATE_COLUMNS[table]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numeric columns
        if table in self.NUMERIC_COLUMNS:
            for col in self.NUMERIC_COLUMNS[table]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean occurrence codes (convert invalid codes to NULL)
        if table == "Occurrences" and "occurrence_code" in df.columns:
            # Convert "0" or other invalid codes to NULL
            df.loc[df["occurrence_code"] == 0, "occurrence_code"] = None
            # Ensure codes are 3-digit strings
            df["occurrence_code"] = df["occurrence_code"].apply(
                lambda x: str(int(x)).zfill(3) if pd.notna(x) and x > 0 else None
            )

        # Clean coordinate bounds for events table
        if table == "events":
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

            # Convert ev_time from numeric (HHMM format) to TIME
            if "ev_time" in df.columns:

                def convert_time(val):
                    """Convert numeric HHMM format to TIME string"""
                    if pd.isna(val):
                        return None
                    try:
                        # Handle string or numeric input
                        if isinstance(val, str):
                            val = val.strip()
                            if val == "" or val.upper() in ["UNK", "UNKN"]:
                                return None
                        # Convert to int and extract hours/minutes
                        time_int = int(float(val))
                        hours = time_int // 100
                        minutes = time_int % 100
                        # Validate and format
                        if 0 <= hours <= 23 and 0 <= minutes <= 59:
                            return f"{hours:02d}:{minutes:02d}:00"
                        return None
                    except (ValueError, TypeError):
                        return None

                df["ev_time"] = df["ev_time"].apply(convert_time)

            # Validate event dates (1962 to current year + 1)
            if "ev_date" in df.columns:
                current_year = datetime.now().year
                # Set invalid dates to None
                invalid_dates = (
                    (df["ev_date"].isna())
                    | (df["ev_date"].dt.year < 1962)
                    | (df["ev_date"].dt.year > current_year + 1)
                )
                invalid_count = invalid_dates.sum()
                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} rows with NULL or invalid ev_date - removing these rows"
                    )
                    df = df[~invalid_dates]

        # Clean Flight_Crew age data
        if table == "Flight_Crew":
            if "crew_age" in df.columns:
                # Convert invalid ages (< 10 or > 120) to NULL
                invalid_mask = (df["crew_age"].notna()) & (
                    (df["crew_age"] < 10) | (df["crew_age"] > 120)
                )
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    logger.warning(
                        f"Converting {invalid_count} invalid crew ages (< 10 or > 120) to NULL"
                    )
                df.loc[invalid_mask, "crew_age"] = None

        # Replace empty strings with None
        df = df.replace({"": None, "UNK": None, "UNKN": None, "NONE": None})

        # Trim whitespace from string columns (exclude narrative fields and boolean columns)
        for col in df.select_dtypes(include=["object"]).columns:
            # Skip narrative fields and any column that might be boolean
            if col not in [
                "narr_accp",
                "narr_cause",
                "narr_accf",
                "narr_inc",
                "narr_rectification",
                "cm_inpc",
            ]:
                try:
                    df[col] = df[col].str.strip()
                except AttributeError:
                    # Skip if column doesn't support .str accessor (e.g., already converted to bool)
                    pass

        # Truncate string fields to match database CHAR constraints
        # owner_state, ev_state are CHAR(2)
        # far_part is CHAR(4)
        # ev_dow is CHAR(2)
        # etc.
        char_limits = {
            "owner_state": 2,
            "ev_state": 2,
            "ev_dow": 2,
            "ev_country": 3,
            "far_part": 4,
            "damage": 4,
            "acft_category": 4,
            "fixed_retractable": 4,
            "crew_sex": 1,
            "crew_category": 3,
        }

        for col, max_len in char_limits.items():
            if col in df.columns:
                # Truncate strings exceeding max length
                df[col] = df[col].astype(str).str[:max_len]
                # Convert 'nan' string back to None
                df.loc[df[col] == "nan", col] = None

        # Remove rows with NULL primary keys
        if table == "events" and "ev_id" in df.columns:
            before_count = len(df)
            df = df[df["ev_id"].notna()]
            removed = before_count - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} rows with NULL ev_id")

        return df

    def get_existing_columns(self, table: str) -> list:
        """Get list of columns that exist in PostgreSQL table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """
                    ),
                    {"table_name": table.lower()},
                )
                columns = [row[0] for row in result.fetchall()]
                return columns
        except Exception as e:
            logger.error(f"Failed to get columns for {table}: {e}")
            return []

    def align_dataframe_to_schema(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Align dataframe columns to PostgreSQL schema"""

        db_columns = self.get_existing_columns(table)
        if not db_columns:
            return df

        # Remove columns not in schema (exclude auto-generated columns)
        auto_columns = [
            "id",
            "crew_no",
            "eng_no",
            "created_at",
            "updated_at",
            "content_hash",
            "ev_year",
            "ev_month",
            "location_geom",
            "search_vector",
            "lchg_date",
            "lchg_userid",
        ]
        db_columns_input = [col for col in db_columns if col not in auto_columns]

        # Keep only columns that exist in database
        df_columns = [col for col in df.columns if col in db_columns_input]

        # Log dropped columns
        dropped = [
            col
            for col in df.columns
            if col not in df_columns and col not in auto_columns
        ]
        if dropped:
            logger.info(f"  Dropping columns not in schema: {', '.join(dropped)}")

        df = df[df_columns]

        logger.info(f"  Aligned {len(df_columns)} columns to schema")

        return df

    def get_existing_event_ids(self) -> set:
        """Get set of event IDs already in database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT ev_id FROM events"))
                return {row[0] for row in result.fetchall()}
        except Exception as e:
            logger.error(f"Failed to fetch existing event IDs: {e}")
            return set()

    def filter_existing_records(
        self,
        df: pd.DataFrame,
        table: str,
        existing_ev_ids: set,
        all_db_ev_ids: set = None,
    ) -> pd.DataFrame:
        """Filter out records that already exist in database"""
        if table == "events" and "ev_id" in df.columns:
            # For events table: skip events that already exist in DB
            before_count = len(df)
            df = df[~df["ev_id"].isin(existing_ev_ids)]
            after_count = len(df)
            skipped = before_count - after_count
            if skipped > 0:
                logger.info(
                    f"  Skipping {skipped:,} existing events (already in database)"
                )
            return df
        elif table != "events" and "ev_id" in df.columns and all_db_ev_ids is not None:
            # For child tables: only keep records whose parent event exists in DB
            # (Either was there before OR was just loaded)
            before_count = len(df)
            df = df[df["ev_id"].isin(all_db_ev_ids)]
            after_count = len(df)
            skipped = before_count - after_count
            if skipped > 0:
                logger.info(
                    f"  Skipping {skipped:,} records (parent events not in database)"
                )
            return df
        return df

    def load_table(
        self,
        table: str,
        chunk_size: int = 1000,
        existing_ev_ids: set = None,
        all_db_ev_ids: set = None,
    ) -> dict:
        """Load single table from CSV to PostgreSQL"""

        csv_file = self.data_dir / f"{table}.csv"

        if not csv_file.exists():
            logger.warning(f"✗ {table:20} CSV file not found: {csv_file}")
            return {"status": "skipped", "reason": "file_not_found"}

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Loading table: {table}")
            logger.info(f"{'=' * 60}")

            # Read CSV
            logger.info(f"Reading CSV: {csv_file}")
            df = pd.read_csv(csv_file, low_memory=False)
            total_rows = len(df)

            if total_rows == 0:
                logger.warning("  Empty CSV file, skipping...")
                return {"status": "skipped", "reason": "empty"}

            logger.info(f"  Rows: {total_rows:,}")
            logger.info(f"  Columns: {len(df.columns)}")

            # Clean and transform
            logger.info("  Cleaning data...")
            df = self.clean_dataframe(df, table)

            # Filter existing records (skip duplicates)
            if existing_ev_ids is not None or all_db_ev_ids is not None:
                df = self.filter_existing_records(
                    df,
                    table,
                    existing_ev_ids if existing_ev_ids else set(),
                    all_db_ev_ids,
                )
                if len(df) == 0:
                    logger.warning(
                        "  All records already exist or have no parent events, skipping table load"
                    )
                    return {
                        "status": "skipped",
                        "reason": "all_filtered",
                        "rows_skipped": total_rows,
                    }

            # Align to schema
            logger.info("  Aligning to PostgreSQL schema...")
            df = self.align_dataframe_to_schema(df, table)

            # Load to PostgreSQL
            logger.info("  Loading to PostgreSQL...")
            logger.info(f"  DataFrame shape before insert: {df.shape}")
            logger.info(f"  DataFrame columns: {list(df.columns)}")
            if len(df) > 0:
                logger.info(f"  First row sample: {df.iloc[0].to_dict()}")
            start_time = datetime.now()

            # Use explicit connection to ensure commit
            # For tables with composite primary keys, use INSERT ... ON CONFLICT DO NOTHING
            # to skip duplicates gracefully
            with self.engine.begin() as conn:
                if table in [
                    "aircraft",
                    "Flight_Crew",
                    "injury",
                    "Findings",
                    "Occurrences",
                    "seq_of_events",
                    "Events_Sequence",
                    "engines",
                ]:
                    # Use custom insert method with ON CONFLICT DO NOTHING
                    from sqlalchemy import MetaData, Table as SQLATable
                    from sqlalchemy.dialects.postgresql import insert

                    metadata = MetaData()
                    sql_table = SQLATable(table.lower(), metadata, autoload_with=conn)

                    # Insert in chunks
                    chunk_list = [
                        df[i : i + chunk_size] for i in range(0, len(df), chunk_size)
                    ]
                    for chunk in chunk_list:
                        records = chunk.to_dict("records")
                        stmt = insert(sql_table).values(records)
                        stmt = stmt.on_conflict_do_nothing()
                        conn.execute(stmt)
                else:
                    # For events and other tables, use standard append
                    df.to_sql(
                        table.lower(),
                        conn,
                        if_exists="append",
                        index=False,
                        method=None,
                        chunksize=chunk_size,
                    )

            duration = (datetime.now() - start_time).total_seconds()
            rows_per_sec = total_rows / duration if duration > 0 else 0

            logger.info("  ✓ Success!")
            logger.info(f"  Duration: {duration:.1f}s ({rows_per_sec:.0f} rows/sec)")

            return {
                "status": "success",
                "rows_loaded": total_rows,
                "duration_sec": round(duration, 2),
                "rows_per_sec": round(rows_per_sec, 0),
            }

        except IntegrityError as e:
            logger.error(f"  ✗ Integrity error: {e}")
            return {"status": "failed", "error": "integrity_error", "message": str(e)}
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"status": "failed", "error": type(e).__name__, "message": str(e)}

    def verify_load(self, table: str) -> dict:
        """Verify loaded data"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table.lower()}"))
                count = result.fetchone()[0]
                return {"table": table, "count": count}
        except Exception as e:
            logger.error(f"Failed to verify {table}: {e}")
            return {"table": table, "count": 0, "error": str(e)}

    def run_load(self, skip_tables: list = None) -> dict:
        """Run complete data load"""

        skip_tables = skip_tables or []

        logger.info("\n" + "=" * 60)
        logger.info("NTSB Historical Data - PostgreSQL Data Load")
        logger.info("=" * 60)
        logger.info(f"Database: {self.db_name}")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Legacy mode: {self.is_legacy}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60 + "\n")

        # Test connection
        if not self.test_connection():
            logger.error("Database connection failed. Exiting.")
            return {"status": "connection_failed"}

        # Get existing event IDs to avoid duplicates
        logger.info("Fetching existing event IDs from database...")
        existing_ev_ids_before = self.get_existing_event_ids()
        logger.info(f"  Found {len(existing_ev_ids_before):,} existing events\n")

        # Load each table
        results = {}
        total_rows = 0
        total_duration = 0

        for table in self.TABLES:
            if table in skip_tables:
                logger.info(f"Skipping table: {table}")
                results[table] = {"status": "skipped", "reason": "user_request"}
                continue

            # For events table, pass existing IDs to skip duplicates
            # For other tables, use all DB event IDs to ensure foreign key integrity
            if table == "events":
                result = self.load_table(table, existing_ev_ids=existing_ev_ids_before)
                # Refresh existing IDs after loading events
                existing_ev_ids_after = self.get_existing_event_ids()
            else:
                # For child tables, only keep records whose parent exists in DB
                result = self.load_table(table, all_db_ev_ids=existing_ev_ids_after)

            results[table] = result

            if result["status"] == "success":
                total_rows += result["rows_loaded"]
                total_duration += result["duration_sec"]

        # Verify loads
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)

        verification = {}
        for table in self.TABLES:
            if table not in skip_tables:
                verify_result = self.verify_load(table)
                verification[table] = verify_result
                logger.info(f"  {table:20} {verify_result['count']:>8,} rows")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("LOAD SUMMARY")
        logger.info("=" * 60)

        success_count = sum(1 for r in results.values() if r["status"] == "success")
        failed_count = sum(1 for r in results.values() if r["status"] == "failed")
        skipped_count = sum(1 for r in results.values() if r["status"] == "skipped")

        logger.info(f"Tables processed: {len(self.TABLES)}")
        logger.info(f"  ✓ Success: {success_count}")
        logger.info(f"  ✗ Failed: {failed_count}")
        logger.info(f"  ⊘ Skipped: {skipped_count}")
        logger.info(f"Total rows loaded: {total_rows:,}")
        logger.info(f"Total duration: {total_duration:.1f}s")
        logger.info("=" * 60 + "\n")

        return {
            "status": "completed" if failed_count == 0 else "partial",
            "results": results,
            "verification": verification,
            "summary": {
                "success": success_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "total_rows": total_rows,
                "total_duration": total_duration,
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Load historical CSV data into PostgreSQL database"
    )
    parser.add_argument(
        "--source",
        choices=["pre2008", "pre1982"],
        required=True,
        help="Historical data source (pre2008 or pre1982)",
    )
    parser.add_argument(
        "--db-name", default="ntsb_aviation", help="PostgreSQL database name"
    )
    parser.add_argument("--db-user", help="PostgreSQL user (default: current user)")
    parser.add_argument(
        "--skip-tables", nargs="+", help="Tables to skip (space-separated)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for bulk inserts (default: 1000)",
    )

    args = parser.parse_args()

    # Determine data directory
    data_dir = f"data/{args.source}"

    loader = HistoricalDataLoader(
        data_dir=data_dir, db_name=args.db_name, db_user=args.db_user
    )

    results = loader.run_load(skip_tables=args.skip_tables)

    # Exit code based on results
    if results.get("status") == "completed":
        logger.info("✓ Data load completed successfully")
        return 0
    elif results.get("status") == "partial":
        logger.warning("⚠ Data load completed with some failures")
        return 1
    else:
        logger.error("✗ Data load failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
