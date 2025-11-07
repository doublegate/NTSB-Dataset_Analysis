#!/usr/bin/env python3
"""
load_transformed_pre1982.py - Load Pre-Transformed PRE1982 CSV Data into PostgreSQL

This script loads CSV files that have been transformed by transform_pre1982.py
into the NTSB PostgreSQL database using the staging table pattern.

Unlike load_with_staging.py which extracts from MDB files, this loader:
1. Reads pre-transformed CSV files from data/pre1982_transformed/
2. Loads into staging tables
3. Merges into production (should find ZERO duplicates, 1962-1981 has no overlap)
4. Updates load_tracking

Usage:
    # Step 1: Transform PRE1982.MDB to CSVs
    python scripts/transform_pre1982.py

    # Step 2: Load transformed CSVs to PostgreSQL
    python scripts/load_transformed_pre1982.py --source data/pre1982_transformed

Author: NTSB Dataset Analysis Project
Date: 2025-11-07
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from typing import Dict, List, Tuple, Optional
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("load_transformed_pre1982.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ntsb_aviation',
    'user': os.getenv('USER'),
    'password': os.getenv('PGPASSWORD', '')
}

# Table load order (respects foreign keys)
TABLE_ORDER = [
    'events',          # Parent table (no foreign keys)
    'aircraft',        # References events
    'Flight_Crew',     # References events + aircraft
    'injury',          # References events
    'Findings',        # References events
    'Occurrences',     # References events
    'seq_of_events',   # References events
    'Events_Sequence', # References events
    'engines',         # References events + aircraft
    'narratives',      # References events
    'NTSB_Admin'       # References events
]


class TransformedPRE1982Loader:
    """Load pre-transformed PRE1982 CSV data into PostgreSQL."""

    def __init__(self, csv_dir: str, force_reload: bool = False):
        """
        Initialize loader.

        Args:
            csv_dir: Directory containing transformed CSV files
            force_reload: Allow reloading PRE1982 (dangerous!)
        """
        self.csv_dir = Path(csv_dir)
        self.force_reload = force_reload
        self.conn = None
        self.cursor = None

        # Statistics
        self.stats = {
            'events_in_staging': 0,
            'events_in_production': 0,
            'duplicate_events': 0,
            'new_events_added': 0,
            'child_records_loaded': {},
            'start_time': datetime.now(),
            'end_time': None
        }

        if not self.csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        logger.info(f"Initialized PRE1982 Loader")
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
        """Check if PRE1982 has already been loaded."""
        self.cursor.execute("""
            SELECT load_status, load_completed_at
            FROM load_tracking
            WHERE database_name = 'PRE1982.MDB'
        """)

        result = self.cursor.fetchone()
        if result:
            status, completed_at = result
            if status == 'completed' and completed_at:
                logger.error(f"✗ PRE1982.MDB already loaded on {completed_at}")
                logger.error("✗ PRE1982 is historical data and should only load ONCE")

                if not self.force_reload:
                    logger.error("✗ Use --force flag to override (dangerous!)")
                    sys.exit(1)
                else:
                    logger.warning("⚠ --force flag detected, proceeding with re-load")
            return status
        else:
            logger.warning("⚠ PRE1982.MDB not found in load_tracking table")
            return 'unknown'

    def update_load_status(self, status: str, **kwargs):
        """Update load tracking table."""
        fields = ['load_status = %s']
        values = [status]

        if status == 'in_progress':
            fields.append('load_started_at = %s')
            values.append(datetime.now())
        elif status == 'completed':
            fields.append('load_completed_at = %s')
            values.append(datetime.now())

            # Calculate duration
            if self.stats['start_time']:
                duration = (datetime.now() - self.stats['start_time']).total_seconds()
                fields.append('load_duration_seconds = %s')
                values.append(int(duration))

        for key, value in kwargs.items():
            fields.append(f"{key} = %s")
            values.append(value)

        query = f"UPDATE load_tracking SET {', '.join(fields)} WHERE database_name = 'PRE1982.MDB'"
        values.append('PRE1982.MDB')

        # Insert if not exists
        self.cursor.execute("""
            INSERT INTO load_tracking (database_name, load_status)
            VALUES ('PRE1982.MDB', %s)
            ON CONFLICT (database_name) DO UPDATE SET load_status = EXCLUDED.load_status
        """, (status,))

        # Now update with all fields
        self.cursor.execute(query, values)
        self.conn.commit()

    def load_csv_to_staging(self, table_name: str):
        """Load CSV file into staging table using COPY."""
        csv_path = self.csv_dir / f"{table_name}.csv"

        if not csv_path.exists():
            logger.warning(f"  ⊘ CSV not found: {csv_path.name}, skipping")
            return

        logger.info(f"Loading {table_name} from {csv_path.name}...")

        try:
            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)

            if df.empty:
                logger.info(f"  ⊘ Empty CSV, skipping")
                return

            # Align columns with staging table schema
            self.cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'staging' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name.lower(),))

            db_columns = [row[0] for row in self.cursor.fetchall()]

            if not db_columns:
                logger.warning(f"  ⚠ Staging table not found for {table_name}")
                return

            # Select only columns that exist in both CSV and database
            df_columns = [col for col in db_columns if col in df.columns]
            df = df[df_columns]
            df = df.where(pd.notnull(df), None)

            # Use COPY for fast bulk load
            output = StringIO()
            df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
            output.seek(0)

            columns_str = ', '.join(df_columns)
            staging_table = f"staging.{table_name.lower()}"

            self.cursor.copy_expert(
                f"COPY {staging_table} ({columns_str}) FROM STDIN WITH CSV DELIMITER E'\\t' NULL '\\N'",
                output
            )
            self.conn.commit()
            logger.info(f"  ✓ Loaded {len(df):,} rows into {staging_table}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"  ✗ Failed to load {table_name}: {e}")
            raise

    def identify_duplicates(self) -> Tuple[List[str], List[str]]:
        """
        Identify duplicate vs unique events.

        Expected: ZERO duplicates (PRE1982 is 1962-1981, no overlap with 2000-2025)

        Returns:
            (duplicate_ev_ids, unique_ev_ids)
        """
        logger.info("Analyzing event duplicates...")

        # Get all ev_ids from staging
        self.cursor.execute("SELECT ev_id FROM staging.events")
        staging_ids = {row[0] for row in self.cursor.fetchall()}
        self.stats['events_in_staging'] = len(staging_ids)

        # Get all ev_ids from production
        self.cursor.execute("SELECT ev_id FROM public.events")
        production_ids = {row[0] for row in self.cursor.fetchall()}
        self.stats['events_in_production'] = len(production_ids)

        # Identify duplicates and unique events
        duplicate_ids = staging_ids & production_ids
        unique_ids = staging_ids - production_ids

        self.stats['duplicate_events'] = len(duplicate_ids)
        self.stats['new_events_added'] = len(unique_ids)

        logger.info(f"  Events in staging: {len(staging_ids):,}")
        logger.info(f"  Events in production: {len(production_ids):,}")
        logger.info(f"  Duplicates found: {len(duplicate_ids):,}")
        logger.info(f"  New unique events: {len(unique_ids):,}")

        if len(duplicate_ids) > 0:
            logger.warning(f"  ⚠ UNEXPECTED: Found {len(duplicate_ids)} duplicates!")
            logger.warning(f"  ⚠ PRE1982 (1962-1981) should have ZERO overlap with existing data")

        return list(duplicate_ids), list(unique_ids)

    def merge_unique_events(self, unique_ids: List[str]):
        """Insert only unique events into production tables."""
        if not unique_ids:
            logger.info("No new unique events to merge")
            return

        logger.info(f"Merging {len(unique_ids):,} unique events into production...")

        # Insert unique events
        self.cursor.execute("""
            INSERT INTO public.events
            SELECT * FROM staging.events
            WHERE ev_id = ANY(%s)
        """, (unique_ids,))

        rows_inserted = self.cursor.rowcount
        self.conn.commit()

        logger.info(f"  ✓ Inserted {rows_inserted:,} new events")

    def load_child_table(self, table_name: str):
        """Load child table records from staging."""
        logger.info(f"Loading {table_name} records...")

        # Get staging row count
        self.cursor.execute(f"SELECT COUNT(*) FROM staging.{table_name.lower()}")
        staging_count = self.cursor.fetchone()[0]

        if staging_count == 0:
            logger.info(f"  ⊘ No records in staging.{table_name.lower()}")
            return

        # Determine primary key for conflict resolution
        pk_columns = self._get_primary_key_columns(table_name)

        if not pk_columns:
            logger.warning(f"  ⚠ No primary key found for {table_name}, using basic INSERT")
            self.cursor.execute(f"""
                INSERT INTO public.{table_name.lower()}
                SELECT * FROM staging.{table_name.lower()}
            """)
        else:
            pk_conflict = ', '.join(pk_columns)

            # Use ON CONFLICT DO NOTHING to skip duplicates
            self.cursor.execute(f"""
                INSERT INTO public.{table_name.lower()}
                SELECT s.* FROM staging.{table_name.lower()} s
                INNER JOIN public.events e ON s.ev_id = e.ev_id
                ON CONFLICT ({pk_conflict}) DO NOTHING
            """)

        rows_inserted = self.cursor.rowcount
        self.conn.commit()

        self.stats['child_records_loaded'][table_name] = rows_inserted
        logger.info(f"  ✓ Loaded {rows_inserted:,} {table_name} records")

    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        """Get primary key column names for a table."""
        self.cursor.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
        """, (f"public.{table_name.lower()}",))

        return [row[0] for row in self.cursor.fetchall()]

    def cleanup_staging(self):
        """Truncate staging tables after successful load."""
        logger.info("Cleaning up staging tables...")

        self.cursor.execute("SELECT staging.truncate_all_tables()")
        self.conn.commit()

        logger.info("  ✓ Staging tables truncated")

    def generate_load_report(self):
        """Generate detailed load statistics report."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             PRE1982 Historical Data Load Report              ║
╚══════════════════════════════════════════════════════════════╝

Database: PRE1982.MDB (1962-1981)
Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)

EVENT STATISTICS:
  Events in staging:      {self.stats['events_in_staging']:>8,}
  Events in production:   {self.stats['events_in_production']:>8,}
  Duplicate events:       {self.stats['duplicate_events']:>8,}
  New unique events:      {self.stats['new_events_added']:>8,}

CHILD RECORDS LOADED:
"""
        for table, count in self.stats['child_records_loaded'].items():
            report += f"  {table:20s}  {count:>8,}\n"

        total_child = sum(self.stats['child_records_loaded'].values())
        report += f"\nTotal child records:    {total_child:>8,}\n"
        report += f"Grand total loaded:     {self.stats['new_events_added'] + total_child:>8,}\n"

        if self.stats['duplicate_events'] > 0:
            report += f"\n⚠ WARNING: Found {self.stats['duplicate_events']} duplicate events\n"
            report += "⚠ This is unexpected - PRE1982 (1962-1981) should have NO overlap\n"

        logger.info(report)

        # Save to file
        report_file = Path('docs') / "PRE1982_load_report.txt"
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(report)
        logger.info(f"Report saved to {report_file}")

    def run(self):
        """Execute complete load process."""
        logger.info("=" * 70)
        logger.info("PRE1982 Transformed Data Loader")
        logger.info("=" * 70)

        try:
            # Step 1: Database connection and status check
            self.connect_db()
            status = self.check_load_status()
            self.update_load_status('in_progress')

            # Step 2: Load CSVs to staging
            logger.info("\n--- Loading CSVs to Staging ---")
            for table in TABLE_ORDER:
                self.load_csv_to_staging(table)

            # Step 3: Identify duplicates
            logger.info("\n--- Analyzing Duplicates ---")
            duplicate_ids, unique_ids = self.identify_duplicates()

            # Step 4: Merge unique events
            logger.info("\n--- Merging Unique Events ---")
            self.merge_unique_events(unique_ids)

            # Step 5: Load child records
            logger.info("\n--- Loading Child Tables ---")
            for table in TABLE_ORDER[1:]:  # Skip 'events' (already merged)
                self.load_child_table(table)

            # Step 6: Cleanup
            self.cleanup_staging()

            # Step 7: Update tracking
            self.update_load_status(
                'completed',
                events_loaded=self.stats['new_events_added'],
                total_rows_loaded=self.stats['new_events_added'] + sum(self.stats['child_records_loaded'].values()),
                duplicate_events_found=self.stats['duplicate_events']
            )

            # Step 8: Generate report
            self.generate_load_report()

            logger.info("\n✓ PRE1982 load completed successfully!")

        except Exception as e:
            logger.error(f"\n✗ Load failed: {e}", exc_info=True)
            self.update_load_status('failed', notes=str(e))
            raise

        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Load pre-transformed PRE1982 CSV data into PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load with default paths
  python scripts/load_transformed_pre1982.py

  # Load with custom CSV directory
  python scripts/load_transformed_pre1982.py --source /path/to/transformed/csvs

  # Force reload (dangerous!)
  python scripts/load_transformed_pre1982.py --force

Prerequisites:
  1. Run transform_pre1982.py first to generate transformed CSVs
  2. PostgreSQL database must be running
  3. Staging tables must exist (run scripts/create_staging_tables.sql)
        """
    )

    parser.add_argument(
        '--source',
        default='data/pre1982_transformed',
        help='Directory containing transformed CSV files (default: data/pre1982_transformed)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reload even if already completed (dangerous!)'
    )

    args = parser.parse_args()

    # Check if CSV directory exists
    if not Path(args.source).exists():
        logger.error(f"✗ CSV directory not found: {args.source}")
        logger.error("✗ Run transform_pre1982.py first to generate transformed CSVs")
        sys.exit(1)

    # Execute load
    loader = TransformedPRE1982Loader(args.source, force_reload=args.force)
    loader.run()


if __name__ == '__main__':
    main()
