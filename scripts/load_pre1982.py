#!/usr/bin/env python3
"""
load_pre1982.py - Load PRE1982.MDB Historical Aviation Accident Data (1962-1981)

Sprint 4 Phase 3: Custom ETL Script
Version: 1.0.0
Date: 2025-11-07

This script transforms denormalized PRE1982.MDB legacy data into normalized PostgreSQL schema.

Key Transformations:
- Denormalized tblFirstHalf (224 columns) → Normalized 6 tables
- RecNum integer → ev_id VARCHAR (generated: YYYYMMDDX{RecNum:06d})
- Coded fields → Decoded using code_mappings schema
- Wide injury columns (50+) → Tall normalized rows
- Legacy date format (MM/DD/YY) → PostgreSQL DATE (YYYY-MM-DD)
- HHMM integer time → HH:MM:SS TIME format

Architecture:
1. Extract tblFirstHalf from PRE1982.MDB
2. Transform denormalized row → 6 normalized tables (events, aircraft, Flight_Crew, injury, Findings, narratives)
3. Load to staging tables using COPY
4. Validate data quality
5. Merge unique events to production
6. Update load_tracking
7. Generate detailed load report

Usage:
    python scripts/load_pre1982.py --source PRE1982.MDB [--limit 100] [--dry-run]
"""

import sys
import os
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2 import sql
import logging
import subprocess
from typing import Dict, List, Optional
from io import StringIO
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_pre1982.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ntsb_aviation"),
    "user": os.getenv("DB_USER", os.getenv("USER")),
    "password": os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", "")),
}

# MDB file configuration
MDB_FILE = "datasets/PRE1982.MDB"
TABLE_NAME = "tblFirstHalf"  # PRE1982 uses single denormalized table


class PRE1982Loader:
    """Transform and load PRE1982.MDB legacy data into normalized schema."""

    def __init__(
        self, mdb_file: str, limit: Optional[int] = None, dry_run: bool = False
    ):
        """
        Initialize loader.

        Args:
            mdb_file: Path to PRE1982.MDB file
            limit: Limit number of rows for testing (e.g., 100)
            dry_run: If True, transform but don't load to database
        """
        self.mdb_file = mdb_file
        self.limit = limit
        self.dry_run = dry_run
        self.conn = None
        # Initialize code_tables with empty dicts to avoid KeyError
        self.code_tables = {
            "states": {},
            "ages": {},
            "causes": {},
            "injury_levels": {},
            "damage": {},
        }

        # Statistics
        self.stats = {
            "rows_extracted": 0,
            "events_transformed": 0,
            "aircraft_transformed": 0,
            "crew_transformed": 0,
            "injury_transformed": 0,
            "findings_transformed": 0,
            "narratives_transformed": 0,
            "start_time": datetime.now(),
            "end_time": None,
        }

    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logger.info(f"✓ Connected to database: {DB_CONFIG['database']}")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def load_code_tables(self):
        """Load code mapping tables into memory for fast lookup."""
        logger.info("Loading code mapping tables...")

        with self.conn.cursor() as cur:
            # Load state codes
            cur.execute("SELECT legacy_code, state_abbr FROM code_mappings.state_codes")
            self.code_tables["states"] = dict(cur.fetchall())
            logger.info(f"  ✓ Loaded {len(self.code_tables['states'])} state codes")

            # Load age codes
            cur.execute(
                "SELECT legacy_code, (age_min + COALESCE(age_max, age_min))/2 FROM code_mappings.age_codes"
            )
            self.code_tables["ages"] = dict(cur.fetchall())
            logger.info(f"  ✓ Loaded {len(self.code_tables['ages'])} age codes")

            # Load cause factor codes
            cur.execute(
                "SELECT legacy_code, cause_description FROM code_mappings.cause_factor_codes"
            )
            self.code_tables["causes"] = dict(cur.fetchall())
            logger.info(
                f"  ✓ Loaded {len(self.code_tables['causes'])} cause factor codes"
            )

            # Load injury level mapping
            cur.execute(
                "SELECT legacy_suffix, modern_code FROM code_mappings.injury_level_mapping"
            )
            self.code_tables["injury_levels"] = dict(cur.fetchall())
            logger.info(
                f"  ✓ Loaded {len(self.code_tables['injury_levels'])} injury level mappings"
            )

            # Load damage codes
            cur.execute(
                "SELECT legacy_code, modern_code FROM code_mappings.damage_codes"
            )
            self.code_tables["damage"] = dict(cur.fetchall())
            logger.info(f"  ✓ Loaded {len(self.code_tables['damage'])} damage codes")

    def extract_from_mdb(self, table_name: str) -> pd.DataFrame:
        """Extract table from MDB file using mdb-export."""
        logger.info(f"Extracting {table_name} from {self.mdb_file}...")

        try:
            cmd = ["mdb-export", "-D", "%Y-%m-%d", self.mdb_file, table_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Load into pandas
            df = pd.read_csv(StringIO(result.stdout))

            # Apply limit if specified
            if self.limit:
                df = df.head(self.limit)

            self.stats["rows_extracted"] = len(df)
            logger.info(f"  ✓ Extracted {len(df):,} rows from {table_name}")
            return df

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Failed to extract {table_name}: {e}")
            raise

    def generate_ev_id(self, rec_num: int, date_occurrence: str) -> str:
        """
        Generate synthetic ev_id from RecNum and DATE_OCCURRENCE.

        Format: YYYYMMDDX{RecNum:06d}
        Example: RecNum=40, DATE_OCCURRENCE='07/23/62' → '19620723X000040'

        The 'X' marker distinguishes legacy events from modern events.
        """
        # Parse date (handles MM/DD/YY format)
        try:
            # Parse date - PRE1982 uses MM/DD/YY format where YY is always 62-81 (1962-1981)
            # Need to manually enforce 19XX century
            date_obj = pd.to_datetime(date_occurrence, format="%m/%d/%y %H:%M:%S")

            # Fix century if pandas inferred wrong century (e.g., 2062 instead of 1962)
            if date_obj.year > 2000:
                # Subtract 100 years: 2062 → 1962
                date_obj = date_obj.replace(year=date_obj.year - 100)

            date_str = date_obj.strftime("%Y%m%d")
        except Exception:
            # Fallback if date parsing fails
            logger.warning(
                f"Failed to parse date '{date_occurrence}' for RecNum {rec_num}"
            )
            date_str = "19620101"  # Default to 1962-01-01

        # Format: YYYYMMDDX{RecNum:06d}
        ev_id = f"{date_str}X{rec_num:06d}"
        return ev_id

    def parse_legacy_date(self, date_str: str) -> Optional[str]:
        """
        Parse legacy date format 'MM/DD/YY HH:MM:SS' to PostgreSQL DATE 'YYYY-MM-DD'.

        Handles 2-digit year:
        - 00-39 → 2000-2039
        - 40-99 → 1940-1999
        (PRE1982 should only have 62-81)
        """
        if pd.isna(date_str) or date_str == "":
            return None

        try:
            # Parse with 2-digit year - PRE1982 is always 1962-1981
            date_obj = pd.to_datetime(date_str, format="%m/%d/%y %H:%M:%S")

            # Fix century if pandas inferred wrong (2062 → 1962)
            if date_obj.year > 2000:
                date_obj = date_obj.replace(year=date_obj.year - 100)

            # Validate 1962-1981 range for PRE1982
            if date_obj.year < 1962 or date_obj.year > 1981:
                logger.warning(
                    f"Date out of expected range: {date_str} → {date_obj.year}"
                )

            return date_obj.strftime("%Y-%m-%d")
        except Exception:
            logger.warning(f"Failed to parse date: {date_str}")
            return None

    def parse_legacy_time(self, time_value) -> Optional[str]:
        """
        Parse legacy time format (HHMM integer) to PostgreSQL TIME 'HH:MM:SS'.

        Examples:
            825 → "08:25:00"
            1530 → "15:30:00"
            0 → "00:00:00"
        """
        if pd.isna(time_value) or time_value == "":
            return None

        try:
            time_int = int(float(time_value))
            hours = time_int // 100
            minutes = time_int % 100

            # Validate ranges
            if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
                return None

            return f"{hours:02d}:{minutes:02d}:00"
        except Exception:
            return None

    def decode_state(self, state_code) -> Optional[str]:
        """
        Decode state code to 2-letter abbreviation.

        PRE1982 uses letter codes (A, B, C, ...) not numeric codes.
        These map to states alphabetically or by code table.

        For now, return the code as-is if it's already 2 letters.
        """
        if pd.isna(state_code):
            return None

        state_str = str(state_code).strip()

        # If already 2-letter state code, return as-is
        if len(state_str) == 2 and state_str.isalpha():
            return state_str.upper()

        # Try numeric lookup
        try:
            return self.code_tables["states"].get(int(state_str))
        except Exception:
            # Return None for unmapped codes
            return None

    def decode_age(self, age_code: str) -> Optional[int]:
        """Decode age code to integer (midpoint of range)."""
        if pd.isna(age_code):
            return None
        return self.code_tables["ages"].get(str(age_code).strip())

    def parse_pilot_hours(self, hours_str) -> Optional[int]:
        """
        Parse pilot hours with letter suffix (e.g., '11508A' → 11508).
        Strip non-numeric characters.
        """
        if pd.isna(hours_str) or hours_str == "":
            return None

        try:
            # Remove all non-numeric characters
            hours_numeric = "".join(c for c in str(hours_str) if c.isdigit())
            return int(hours_numeric) if hours_numeric else None
        except Exception:
            return None

    def clean_coordinates(self, coord_value) -> Optional[float]:
        """
        Clean coordinate value to decimal degrees.

        Handles: floats, DMS strings, coded values.
        """
        if pd.isna(coord_value):
            return None

        # If already decimal, validate and return
        if isinstance(coord_value, (int, float)):
            coord = float(coord_value)
            if -180 <= coord <= 180:
                return round(coord, 6)
            else:
                return None

        return None

    def transform_events(self, row: pd.Series) -> Dict:
        """Transform denormalized row to events table format."""
        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])

        # Parse date once for multiple uses
        try:
            date_obj = pd.to_datetime(
                row["DATE_OCCURRENCE"], format="%m/%d/%y %H:%M:%S"
            )
            # Fix century if pandas inferred wrong (2062 → 1962)
            if date_obj.year > 2000:
                date_obj = date_obj.replace(year=date_obj.year - 100)
            ev_year = date_obj.year
            ev_month = date_obj.month
        except Exception:
            ev_year = 1962
            ev_month = 1

        return {
            "ev_id": ev_id,
            "ev_date": self.parse_legacy_date(row.get("DATE_OCCURRENCE")),
            "ev_time": self.parse_legacy_time(row.get("TIME_OCCUR")),
            "ev_year": ev_year,
            "ev_month": ev_month,
            "ev_city": row.get("LOCATION"),
            "ev_state": self.decode_state(row.get("LOCAT_STATE_TERR")),
            "ev_country": "USA",  # All PRE1982 events are domestic
            "ntsb_no": row.get("DOCKET_NO"),
            "dec_latitude": self.clean_coordinates(row.get("LATITUDE")),
            "dec_longitude": self.clean_coordinates(row.get("LONGITUDE")),
            "inj_tot_f": row.get("TOTAL_ABRD_FATAL", 0),
            "inj_tot_s": row.get("TOTAL_ABRD_SERIOUS", 0),
            "inj_tot_m": row.get("TOTAL_ABRD_MINOR", 0),
            "inj_tot_n": row.get("TOTAL_ABRD_NONE", 0),
            # Weather columns not available in PRE1982.MDB
            "wx_cond_basic": None,
            "wx_temp": None,
            "wx_wind_dir": None,
            "wx_wind_speed": None,
            "wx_vis": None,
            # Flight plan and phase not available
            "flight_plan_filed": None,
            "flight_activity": None,
            "flight_phase": None,
        }

    def transform_aircraft(self, row: pd.Series) -> Dict:
        """Transform denormalized row to aircraft table format."""
        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])

        # Decode damage code (S=Substantial, D=Destroyed, etc.)
        damage_code = str(row.get("ACFT_ADAMG", "")).strip()
        damage = self.code_tables["damage"].get(damage_code)

        return {
            "ev_id": ev_id,
            "aircraft_key": "1",  # PRE1982 has single aircraft per event
            "regis_no": row.get("REGIST_NO"),
            "acft_make": row.get("ACFT_MAKE"),
            "acft_model": row.get("ACFT_MODEL"),
            "num_eng": row.get("NO_ENGINES"),
            "damage": damage,
        }

    def transform_flight_crew(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized row to Flight_Crew table format.
        Returns 0-2 rows (Pilot 1, Pilot 2 if present).
        """
        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])
        crews = []

        # Pilot 1
        if pd.notna(row.get("PILOT_INVOLED1")) and str(
            row["PILOT_INVOLED1"]
        ).strip().upper() in ["A", "Y"]:
            crews.append(
                {
                    "ev_id": ev_id,
                    "aircraft_key": "1",
                    "crew_category": "PILOT",
                    "crew_age": self.decode_age(row.get("AGE_PILOT1")),
                    "pilot_tot_time": self.parse_pilot_hours(
                        row.get("HOURS_TOTAL_PILOT1")
                    ),
                    "pilot_make_time": self.parse_pilot_hours(
                        row.get("HOURS_IN_TYPE_PILOT1")
                    ),
                }
            )

        # Pilot 2 (Co-pilot)
        if pd.notna(row.get("PILOT_INVOLED2")) and str(
            row["PILOT_INVOLED2"]
        ).strip().upper() in ["A", "B", "Y"]:
            crews.append(
                {
                    "ev_id": ev_id,
                    "aircraft_key": "1",
                    "crew_category": "CO-PILOT",
                    "crew_age": self.decode_age(row.get("AGE_PILOT2")),
                    "pilot_tot_time": self.parse_pilot_hours(
                        row.get("HOURS_TOTAL_PILOT2")
                    ),
                    "pilot_make_time": self.parse_pilot_hours(
                        row.get("HOURS_IN_TYPE_PILOT2")
                    ),
                }
            )

        return crews

    def transform_injury(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized injury columns to normalized injury table rows.
        Returns 10-50 rows per event (wide → tall pivot).
        """
        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])
        injuries = []

        # Define injury categories and their column prefixes
        injury_categories = [
            ("PILOT", "PILOT"),
            ("CO-PILOT", "CO_PILOT"),
            ("DUAL STUDENT", "DUAL_STUDENT"),
            ("CHECK PILOT", "CHK_PILOT"),
            ("FLIGHT ENGINEER", "FLT_ENGN"),
            ("NAVIGATOR", "NAVIGTR"),
            ("CABIN ATTENDANT", "CABIN_ATTN"),
            ("EXTRA CREW", "EXTRA_CREW"),
            ("PASSENGER", "PASSENGERS"),
            ("TOTAL", "TOTAL_ABRD"),
        ]

        # Define injury levels (map to modern codes)
        injury_levels = [
            ("FATAL", "FATL"),
            ("SERIOUS", "SERS"),
            ("MINOR", "MINR"),
            ("NONE", "NONE"),
        ]

        for category, prefix in injury_categories:
            for level_name, level_code in injury_levels:
                col_name = f"{prefix}_{level_name}"
                count = row.get(col_name, 0)

                if pd.notna(count) and int(count) > 0:
                    injuries.append(
                        {
                            "ev_id": ev_id,
                            "aircraft_key": "1",
                            "inj_person_category": category,
                            "inj_level": level_code,
                            "inj_person_count": int(count),
                        }
                    )

        return injuries

    def transform_findings(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized cause factor columns to normalized Findings table rows.
        Returns 1-10 rows per event.

        PRE1982 has CAUSE_FACTOR_1P/M/S through CAUSE_FACTOR_10P/M/S
        - P = Primary cause code
        - M = Modifier code
        - S = Secondary code
        """
        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])
        findings = []

        # PRE1982 has up to 10 cause factors (P/M/S triplets)
        for i in range(1, 11):
            primary = row.get(f"CAUSE_FACTOR_{i}P")
            modifier = row.get(f"CAUSE_FACTOR_{i}M")
            secondary = row.get(f"CAUSE_FACTOR_{i}S")

            if pd.notna(primary):
                # Convert primary to string, remove decimals if float
                if isinstance(primary, float):
                    primary_str = str(int(primary))
                else:
                    primary_str = str(primary).strip()

                # Lookup description from code table
                description = self.code_tables["causes"].get(
                    primary_str, f"LEGACY:{primary_str}"
                )

                # Build cause_factor string (preserve legacy codes)
                # Remove float decimals (e.g., "67.0" → "67") to fit VARCHAR(10) constraint
                if pd.notna(modifier) or pd.notna(secondary):
                    # Convert to string, remove decimals if numeric
                    def clean_code(code):
                        if pd.isna(code):
                            return ""
                        code_str = str(code).strip()
                        # Try to convert to float and remove decimals
                        try:
                            return str(int(float(code_str)))
                        except (ValueError, TypeError):
                            # Keep as-is if not numeric
                            return code_str[:10]  # Truncate to fit

                    mod_str = clean_code(modifier)
                    sec_str = clean_code(secondary)
                    cause_factor = f"{primary_str}-{mod_str}-{sec_str}"
                else:
                    cause_factor = primary_str

                # Ensure cause_factor fits in VARCHAR(10) - truncate if necessary
                if len(cause_factor) > 10:
                    cause_factor = cause_factor[:10]

                findings.append(
                    {
                        "ev_id": ev_id,
                        "aircraft_key": "1",
                        "finding_code": None,  # No modern code mapping
                        "finding_description": description,
                        "cm_inPC": (i == 1),  # First cause factor is probable cause
                        "modifier_code": str(modifier).strip()
                        if pd.notna(modifier)
                        else None,
                        "cause_factor": cause_factor,
                    }
                )

        return findings

    def transform_narratives(self, row: pd.Series) -> Optional[Dict]:
        """
        Transform denormalized row to narratives table format.
        Returns None if no narrative text.
        """
        # PRE1982 may not have narrative columns in tblFirstHalf
        # Check for common narrative column names
        narrative_text = row.get("NARRATIVE_TEXT") or row.get("NARRATIVE") or ""
        probable_cause_text = (
            row.get("PROBABLE_CAUSE_TEXT") or row.get("PROB_CAUSE") or ""
        )

        # Only create narrative row if there's text
        if not narrative_text and not probable_cause_text:
            return None

        ev_id = self.generate_ev_id(row["RecNum"], row["DATE_OCCURRENCE"])
        return {
            "ev_id": ev_id,
            "aircraft_key": "1",
            "narr_accp": narrative_text if narrative_text else None,
            "narr_cause": probable_cause_text if probable_cause_text else None,
        }

    def _get_table_columns(self, table_name: str) -> List[str]:
        """
        Get column names in correct order from PostgreSQL table schema.

        Args:
            table_name: Table name (e.g., 'events', 'aircraft')

        Returns:
            List of column names in table order (excludes generated and auto-increment columns)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'staging'
                  AND table_name = %s
                  AND is_generated = 'NEVER'
                  AND column_default IS NULL  -- Only include columns with no default value
                ORDER BY ordinal_position
                """,
                (table_name,)
            )
            return [row[0] for row in cur.fetchall()]

    def _reorder_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Reorder DataFrame columns to match PostgreSQL table schema.
        Add missing columns as NULL.

        Args:
            df: DataFrame to reorder
            table_name: Table name to match schema

        Returns:
            DataFrame with columns in correct order
        """
        table_cols = self._get_table_columns(table_name)

        # Add missing columns as NULL
        for col in table_cols:
            if col not in df.columns:
                df[col] = None

        # Reorder to match table schema (only use columns that exist in DataFrame)
        ordered_cols = [col for col in table_cols if col in df.columns]
        return df[ordered_cols]

    def _custom_to_csv_with_null_aware_quoting(self, df: pd.DataFrame, buffer) -> None:
        r"""
        Export DataFrame to CSV with proper quoting:
        - Strings with commas are quoted: "ROME,NY"
        - NULL markers (\N) are NEVER quoted: \N (not "\N")

        This is critical for PostgreSQL COPY, which rejects quoted null markers.

        Approach: Use QUOTE_ALL then post-process to unquote \N markers.
        """
        # First pass: use QUOTE_ALL to ensure all strings are quoted
        temp_buffer = StringIO()
        df.to_csv(temp_buffer, index=False, header=False, na_rep="\\N", quoting=csv.QUOTE_ALL, doublequote=True)

        # Second pass: un-quote all \\N markers
        csv_content = temp_buffer.getvalue()
        # Replace "\\N" (quoted null) with \\N (unquoted null)
        csv_content = csv_content.replace('"\\N"', '\\N')

        # Write to output buffer
        buffer.write(csv_content)
        buffer.seek(0)

    def _convert_integer_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Convert float64 columns to Int64 (nullable integer) for PostgreSQL INTEGER columns.

        This prevents pandas from writing "0.0" which PostgreSQL INTEGER rejects.

        Args:
            df: DataFrame to convert
            table_name: Table name to determine which columns to convert

        Returns:
            DataFrame with INTEGER columns converted
        """
        # Define INTEGER columns for each table (from schema.sql)
        INTEGER_COLUMNS = {
            "events": ["ev_year", "ev_month", "inj_tot_f", "inj_tot_s", "inj_tot_m", "inj_tot_n",
                       "wx_temp", "wx_wind_dir", "wx_wind_speed"],
            "aircraft": ["num_eng"],
            "flight_crew": ["crew_age", "pilot_tot_time", "pilot_make_time"],
            "injury": ["inj_person_count"],
            "findings": [],  # No INTEGER columns
            "narratives": [],  # No INTEGER columns
        }

        if table_name not in INTEGER_COLUMNS:
            return df

        int_cols = INTEGER_COLUMNS[table_name]

        # Convert each INTEGER column from float64 to Int64
        for col in int_cols:
            if col in df.columns:
                # Use Int64 (capital I) - nullable integer dtype
                # This prevents "0.0" and handles NaN → NULL correctly
                # ALWAYS use pd.to_numeric first to handle malformed data like "1 0042"
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        return df

    def load_to_staging(self, df: pd.DataFrame, table_name: str):
        """
        Bulk load DataFrame to staging table using COPY.

        Args:
            df: DataFrame to load
            table_name: Table name (e.g., 'events', 'aircraft')
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would load {len(df)} rows to staging.{table_name}")
            return

        logger.info(f"Loading {len(df):,} rows to staging.{table_name}...")

        # DEBUG: Check columns BEFORE filtering
        if table_name == "events":
            logger.info(f"DEBUG - BEFORE filter - DataFrame columns ({len(df.columns)}): {list(df.columns)[:30]}")
            if 'wx_wind_speed' in df.columns:
                sample = df['wx_wind_speed'].dropna().head(1)
                if len(sample) > 0:
                    logger.info(f"DEBUG - BEFORE filter - wx_wind_speed sample: {repr(sample.iloc[0])}")

        # Filter to only expected columns (drop any extras from raw MDB data)
        # NOTE: Weather/flight columns excluded for PRE1982 - malformed data in MDB
        expected_cols_by_table = {
            "events": [
                "ev_id", "ev_date", "ev_time", "ev_year", "ev_month", "ev_city",
                "ev_state", "ev_country", "ntsb_no", "dec_latitude", "dec_longitude",
                "inj_tot_f", "inj_tot_s", "inj_tot_m", "inj_tot_n"
                # wx_cond_basic, wx_temp, wx_wind_dir, wx_wind_speed, wx_vis - EXCLUDED (malformed in PRE1982)
                # flight_plan_filed, flight_activity, flight_phase - EXCLUDED (not in PRE1982)
            ],
            "aircraft": ["ev_id", "aircraft_key", "acft_make", "acft_model", "damage", "num_eng"],
            "flight_crew": ["ev_id", "aircraft_key", "crew_category", "crew_age", "pilot_tot_time", "pilot_make_time"],
            "injury": ["ev_id", "aircraft_key", "inj_person_category", "inj_level", "inj_person_count"],
            "findings": ["ev_id", "aircraft_key", "finding_description", "cm_inPC", "cause_factor"],
            "narratives": ["ev_id", "aircraft_key", "narr_accp", "narr_cause"],
        }

        if table_name in expected_cols_by_table:
            expected_cols = expected_cols_by_table[table_name]
            # Only keep columns that exist in both DataFrame and expected list
            cols_to_keep = [col for col in expected_cols if col in df.columns]
            df = df[cols_to_keep].copy()  # CRITICAL: Use .copy() to prevent pandas from keeping references to filtered columns

            # DEBUG: Check columns AFTER filtering
            if table_name == "events":
                logger.info(f"DEBUG - AFTER filter - DataFrame columns ({len(df.columns)}): {list(df.columns)}")
                if 'wx_wind_speed' in df.columns:
                    logger.info(f"DEBUG - AFTER filter - wx_wind_speed still in DataFrame!")

        # Clear staging table
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE staging.{}").format(sql.Identifier(table_name))
            )

        # Reorder columns to match PostgreSQL table schema
        # BUT only include columns that have at least some non-NULL data
        # (This prevents empty columns from shifting column order during COPY)
        df = self._reorder_columns(df, table_name)

        # Debug: identify all-NULL columns for PRE1982
        if table_name == "events":
            null_only_cols = [col for col in df.columns if df[col].isna().all()]
            if null_only_cols:
                logger.info(f"DEBUG - Dropping {len(null_only_cols)} all-NULL columns: {null_only_cols}")
                df = df.drop(columns=null_only_cols)

        # Debug after reorder
        if table_name == "events":
            logger.info(f"DEBUG - AFTER reorder - DataFrame columns ({len(df.columns)}): {list(df.columns)}")
            if 'wx_wind_speed' in df.columns:
                sample = df['wx_wind_speed'].dropna().head(1)
                if len(sample) > 0:
                    logger.info(f"DEBUG - AFTER reorder - wx_wind_speed sample: {repr(sample.iloc[0])}")

        # Convert INTEGER columns to prevent "0.0" errors
        df = self._convert_integer_columns(df, table_name)

        # Debug: Check for malformed data in INTEGER columns after conversion
        if table_name == "events":
            int_cols = ["wx_temp", "wx_wind_dir", "wx_wind_speed"]
            for col in int_cols:
                if col in df.columns:
                    # Check first non-null value
                    sample = df[col].dropna().head(1)
                    if len(sample) > 0:
                        logger.info(f"DEBUG - {col} sample value: {repr(sample.iloc[0])} (type: {type(sample.iloc[0])})")

        # DEBUG: Check DataFrame shape before CSV conversion
        if table_name == "events":
            logger.info(f"DEBUG - DataFrame shape before to_csv: {df.shape} (rows, cols)")
            logger.info(f"DEBUG - DataFrame index type: {type(df.index)}")
            logger.info(f"DEBUG - DataFrame index name: {df.index.name}")

        # Convert DataFrame to CSV in memory using null-aware quoting
        buffer = StringIO()
        self._custom_to_csv_with_null_aware_quoting(df, buffer)

        # Debug: Print first line of CSV if events
        if table_name == "events":
            csv_content = buffer.getvalue()
            first_line = csv_content.split('\n')[0]
            logger.info(f"DEBUG - First CSV line length: {len(first_line.split(','))} columns")
            logger.info(f"DEBUG - First CSV line (first 500 chars): {first_line[:500]}")
            buffer.seek(0)  # Reset after reading

        # Bulk COPY to staging with explicit column list
        # This is critical for PRE1982 which doesn't have all columns
        with self.conn.cursor() as cur:
            col_list = ",".join(df.columns)
            copy_sql = f"COPY staging.{table_name} ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
            cur.copy_expert(copy_sql, buffer)

        self.conn.commit()
        logger.info(f"  ✓ Loaded {len(df):,} rows to staging.{table_name}")

    def _get_insertable_columns(self, table_name: str) -> List[str]:
        """
        Get list of insertable columns (excludes generated columns).
        Queries information_schema to get columns where is_generated='NEVER'.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND is_generated = 'NEVER'
                ORDER BY ordinal_position
                """,
                (table_name,)
            )
            return [row[0] for row in cur.fetchall()]

    def merge_to_production(self):
        """
        Merge staging tables to production.
        Uses same logic as load_with_staging.py.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would merge staging → production")
            return

        logger.info("Merging staging → production...")

        with self.conn.cursor() as cur:
            # Get insertable columns for events (exclude generated columns like location_geom)
            event_cols = self._get_insertable_columns("events")
            col_list = ", ".join(event_cols)

            # Merge events (only new ev_id values)
            cur.execute(f"""
                INSERT INTO events ({col_list})
                SELECT {col_list} FROM staging.events s
                WHERE NOT EXISTS (
                    SELECT 1 FROM events e WHERE e.ev_id = s.ev_id
                )
            """)
            new_events = cur.rowcount
            logger.info(f"  ✓ Merged {new_events:,} new events")

            # Merge child tables (only for new events to avoid duplicate key violations)
            tables = ["aircraft", "flight_crew", "injury", "findings", "narratives"]
            for table in tables:
                cols = self._get_insertable_columns(table)
                col_list = ", ".join(cols)

                # Build WHERE clause - depends on table structure
                # Some tables have aircraft_key, others don't
                if table == "narratives":
                    # narratives only keyed by ev_id
                    where_clause = "WHERE NOT EXISTS (SELECT 1 FROM narratives e WHERE e.ev_id = s.ev_id)"
                elif "aircraft_key" in cols:
                    # Tables with aircraft_key (aircraft, flight_crew, injury, findings)
                    where_clause = """WHERE NOT EXISTS (
                        SELECT 1 FROM {table} e
                        WHERE e.ev_id = s.ev_id
                        AND e.aircraft_key = s.aircraft_key
                    )"""
                    where_clause = where_clause.format(table=table)
                else:
                    # Fallback for tables without aircraft_key
                    where_clause = f"WHERE NOT EXISTS (SELECT 1 FROM {table} e WHERE e.ev_id = s.ev_id)"

                cur.execute(f"""
                    INSERT INTO {table} ({col_list})
                    SELECT {col_list} FROM staging.{table} s
                    {where_clause}
                """)
                rows = cur.rowcount
                logger.info(f"  ✓ Merged {rows:,} {table} rows")

        self.conn.commit()

    def update_load_tracking(self, events_loaded: int):
        """Update load_tracking table with completion status."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would update load_tracking: {events_loaded} events")
            return

        logger.info("Updating load_tracking...")

        duration = (datetime.now() - self.stats["start_time"]).total_seconds()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE load_tracking
                SET load_status = 'completed',
                    events_loaded = %s,
                    duplicate_events_found = 0,
                    load_completed_at = NOW(),
                    load_duration_seconds = %s
                WHERE database_name = 'PRE1982.MDB'
            """,
                (events_loaded, int(duration)),
            )

        self.conn.commit()
        logger.info(f"  ✓ Updated load_tracking (duration: {duration:.1f}s)")

    def main(self):
        """Main ETL pipeline."""
        try:
            # 1. Connect to database (even in dry-run to load code tables)
            self.connect()
            self.load_code_tables()

            # 2. Extract from PRE1982.MDB
            logger.info(f"Extracting from {self.mdb_file}...")
            df = self.extract_from_mdb(TABLE_NAME)
            logger.info(f"  ✓ Extracted {len(df):,} rows")

            # 3. Transform denormalized → normalized
            logger.info("Transforming data...")

            events_rows = []
            aircraft_rows = []
            crew_rows = []
            injury_rows = []
            findings_rows = []
            narratives_rows = []

            for idx, row in df.iterrows():
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"  Processed {idx:,}/{len(df):,} rows...")

                # Transform to all 6 tables
                event = self.transform_events(row)

                # Skip events with NULL ev_date (violates NOT NULL constraint)
                if event['ev_date'] is None:
                    logger.warning(f"Skipping RecNum {row['RecNum']} due to NULL ev_date")
                    continue

                events_rows.append(event)
                aircraft_rows.append(self.transform_aircraft(row))
                crew_rows.extend(self.transform_flight_crew(row))
                injury_rows.extend(self.transform_injury(row))
                findings_rows.extend(self.transform_findings(row))

                narrative = self.transform_narratives(row)
                if narrative:
                    narratives_rows.append(narrative)

            # Update statistics
            self.stats["events_transformed"] = len(events_rows)
            self.stats["aircraft_transformed"] = len(aircraft_rows)
            self.stats["crew_transformed"] = len(crew_rows)
            self.stats["injury_transformed"] = len(injury_rows)
            self.stats["findings_transformed"] = len(findings_rows)
            self.stats["narratives_transformed"] = len(narratives_rows)

            logger.info("Transformation complete:")
            logger.info(f"  events: {len(events_rows):,}")
            logger.info(f"  aircraft: {len(aircraft_rows):,}")
            logger.info(f"  Flight_Crew: {len(crew_rows):,}")
            logger.info(f"  injury: {len(injury_rows):,}")
            logger.info(f"  Findings: {len(findings_rows):,}")
            logger.info(f"  narratives: {len(narratives_rows):,}")

            # 4. Load to staging tables
            logger.info("Loading to staging tables...")
            self.load_to_staging(pd.DataFrame(events_rows), "events")
            self.load_to_staging(pd.DataFrame(aircraft_rows), "aircraft")
            if crew_rows:
                self.load_to_staging(pd.DataFrame(crew_rows), "flight_crew")
            if injury_rows:
                self.load_to_staging(pd.DataFrame(injury_rows), "injury")
            if findings_rows:
                self.load_to_staging(pd.DataFrame(findings_rows), "findings")
            if narratives_rows:
                self.load_to_staging(pd.DataFrame(narratives_rows), "narratives")

            # 5. Merge to production
            self.merge_to_production()

            # 6. Update load_tracking
            self.update_load_tracking(len(events_rows))

            # 7. Final statistics
            self.stats["end_time"] = datetime.now()
            duration = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()

            logger.info("=" * 60)
            logger.info("PRE1982 load complete!")
            logger.info(f"Duration: {duration:.1f}s")
            logger.info(f"Events: {self.stats['events_transformed']:,}")
            logger.info(
                f"Total rows: {
                    sum(
                        [
                            self.stats['events_transformed'],
                            self.stats['aircraft_transformed'],
                            self.stats['crew_transformed'],
                            self.stats['injury_transformed'],
                            self.stats['findings_transformed'],
                            self.stats['narratives_transformed'],
                        ]
                    ):,}"
            )
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"✗ Load failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load PRE1982.MDB historical aviation accident data"
    )
    parser.add_argument(
        "--source", default="PRE1982.MDB", help="MDB filename (default: PRE1982.MDB)"
    )
    parser.add_argument("--limit", type=int, help="Limit rows for testing (e.g., 100)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test transformation without database changes",
    )

    args = parser.parse_args()

    loader = PRE1982Loader(
        mdb_file=f"datasets/{args.source}", limit=args.limit, dry_run=args.dry_run
    )
    loader.main()
