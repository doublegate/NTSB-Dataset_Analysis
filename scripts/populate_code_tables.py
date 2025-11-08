#!/usr/bin/env python3
"""
populate_code_tables.py - Populate PRE1982 code mapping tables from ct_Pre1982 CSV

Sprint 4: PRE1982 Integration - Phase 2
Purpose: Bulk load 945 cause factor codes and other mappings from extracted CSV
Created: 2025-11-07
Version: 1.0.0

Usage:
    python scripts/populate_code_tables.py
"""

import sys
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("populate_code_tables.log"),
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
    "password": os.getenv("DB_PASSWORD", ""),
}

# CSV file paths
CT_PRE1982_CSV = "data/pre1982/ct_Pre1982.csv"
CAUSE_FACTOR_CSV = "data/pre1982/cause_factor_codes.csv"


def connect_db():
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"✅ Connected to database: {DB_CONFIG['database']}")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise


def load_cause_factor_codes(conn):
    """
    Load 945 cause factor codes from CSV into code_mappings.cause_factor_codes.

    CSV Format:
        Name,Code,Meang
        CAUSE_FACTOR,64,PILOT IN COMMAND
        CAUSE_FACTOR,6401,ATTEMPTED OPERATION W/KNOWN DEFICIENCIES...
    """
    logger.info("Loading cause factor codes from CSV...")

    # Load CSV
    codes = pd.read_csv(CAUSE_FACTOR_CSV)
    logger.info(f"Read {len(codes)} cause factor codes from CSV")

    # Filter to cause factor codes only, skip header rows
    cause_codes = codes[codes["Name"] == "CAUSE_FACTOR"].copy()
    cause_codes = cause_codes[cause_codes["Code"] != "**"]  # Skip header rows

    logger.info(f"Filtered to {len(cause_codes)} valid cause factor codes")

    # Categorize codes based on leading digits
    def categorize_cause(code):
        """Categorize cause factor code based on first 2 digits."""
        code_str = str(code)
        if code_str.startswith("64"):
            return "Pilot In Command"
        elif code_str.startswith("65"):
            return "Copilot"
        elif code_str.startswith("66"):
            return "Flight Crew"
        elif code_str.startswith("70"):
            return "Aircraft"
        elif code_str.startswith("71"):
            return "Engine"
        elif code_str.startswith("72"):
            return "Systems"
        elif code_str.startswith("80"):
            return "Weather"
        elif code_str.startswith("81"):
            return "Airport/Environment"
        elif code_str.startswith("90"):
            return "Maintenance"
        elif code_str.startswith("91"):
            return "Organizational"
        else:
            return "Other"

    cause_codes["cause_category"] = cause_codes["Code"].apply(categorize_cause)

    # Prepare data for insertion
    insert_data = []
    for _, row in cause_codes.iterrows():
        insert_data.append(
            (
                str(row["Code"]),  # legacy_code
                row["Meang"],  # cause_description
                row["cause_category"],  # cause_category
                None,  # modern_finding_code (not mapped yet)
                "ct_Pre1982",  # source
            )
        )

    # Bulk insert
    cur = conn.cursor()
    try:
        execute_batch(
            cur,
            """
            INSERT INTO code_mappings.cause_factor_codes
                (legacy_code, cause_description, cause_category, modern_finding_code, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (legacy_code) DO UPDATE SET
                cause_description = EXCLUDED.cause_description,
                cause_category = EXCLUDED.cause_category
            """,
            insert_data,
            page_size=100,
        )
        conn.commit()
        logger.info(f"✅ Inserted/updated {len(insert_data)} cause factor codes")

        # Verify row count
        cur.execute("SELECT COUNT(*) FROM code_mappings.cause_factor_codes")
        total_count = cur.fetchone()[0]
        logger.info(f"Total cause factor codes in database: {total_count}")

    except Exception as e:
        logger.error(f"❌ Failed to insert cause factor codes: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()


def load_age_codes(conn):
    """
    Load age codes from ct_Pre1982 if available.
    PRE1982 may store ages as direct integers or coded ranges.
    """
    logger.info("Checking for age codes in ct_Pre1982...")

    # Load full code table
    codes = pd.read_csv(CT_PRE1982_CSV)

    # Look for age-related codes
    age_codes = codes[codes["Name"].str.contains("AGE", case=False, na=False)].copy()
    age_codes = age_codes[age_codes["Code"] != "**"]  # Skip headers

    if len(age_codes) == 0:
        logger.info("No age codes found in ct_Pre1982 (ages may be stored as integers)")
        return

    logger.info(f"Found {len(age_codes)} age-related codes")

    # For now, just log them - may need manual mapping
    for _, row in age_codes.head(10).iterrows():
        logger.info(f"  {row['Name']}: {row['Code']} = {row['Meang']}")


def verify_mappings(conn):
    """Verify code mappings are loaded correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)

    cur = conn.cursor()

    # Check row counts
    tables = [
        "state_codes",
        "age_codes",
        "cause_factor_codes",
        "injury_level_mapping",
        "damage_codes",
    ]

    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM code_mappings.{table}")
        count = cur.fetchone()[0]
        logger.info(f"✅ {table}: {count} rows")

    # Test sample queries
    logger.info("\nSample Decoding Tests:")

    # Test state decoding
    cur.execute("SELECT code_mappings.decode_state(32)")
    result = cur.fetchone()[0]
    logger.info(f"  decode_state(32) → '{result}' (expected 'NY')")

    cur.execute("SELECT code_mappings.decode_state(5)")
    result = cur.fetchone()[0]
    logger.info(f"  decode_state(5) → '{result}' (expected 'CA')")

    # Test damage decoding
    cur.execute("SELECT code_mappings.decode_damage('D')")
    result = cur.fetchone()[0]
    logger.info(f"  decode_damage('D') → '{result}' (expected 'DEST')")

    # Test cause factor decoding
    cur.execute("SELECT code_mappings.decode_cause_factor('6401')")
    result = cur.fetchone()[0]
    logger.info(f"  decode_cause_factor('6401') → '{result[:50]}...'")

    # Display mapping statistics
    logger.info("\nMapping Statistics:")
    cur.execute("SELECT * FROM code_mappings.mapping_stats ORDER BY table_name")
    for row in cur.fetchall():
        logger.info(
            f"  {row[0]}: {row[1]} total codes, {row[2]} unique values ({row[3]})"
        )

    cur.close()
    logger.info("=" * 60 + "\n")


def main():
    """Main execution flow."""
    logger.info("Starting code table population...")
    logger.info(f"CSV file: {CT_PRE1982_CSV}")

    # Check CSV exists
    if not os.path.exists(CT_PRE1982_CSV):
        logger.error(f"❌ CSV file not found: {CT_PRE1982_CSV}")
        logger.error(
            "Run: mdb-export datasets/PRE1982.MDB ct_Pre1982 > data/pre1982/ct_Pre1982.csv"
        )
        sys.exit(1)

    # Connect to database
    conn = connect_db()

    try:
        # 1. Load cause factor codes (945 codes)
        load_cause_factor_codes(conn)

        # 2. Check for age codes
        load_age_codes(conn)

        # 3. Verify all mappings
        verify_mappings(conn)

        logger.info("\n✅ Code table population complete!")
        logger.info("\nNext steps:")
        logger.info("1. Review verification results above")
        logger.info(
            "2. Test queries: psql -d ntsb_aviation -c 'SELECT * FROM code_mappings.mapping_stats;'"
        )
        logger.info("3. Proceed to Phase 3: Create load_pre1982.py ETL script")

    except Exception as e:
        logger.error(f"❌ Population failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
