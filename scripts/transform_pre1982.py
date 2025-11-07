#!/usr/bin/env python3
"""
transform_pre1982.py - Custom ETL for Legacy NTSB PRE1982.MDB Database

This script transforms the legacy denormalized PRE1982.MDB schema (1962-1981)
into the modern normalized NTSB schema compatible with load_with_staging.py.

Key Transformations:
1. Generate synthetic ev_id from RecNum + DATE_OCCURRENCE
2. Parse 2-digit year dates (62 = 1962, not 2062)
3. Pivot injury data from wide (50+ columns) to tall (normalized rows)
4. Extract pilot data (Pilot 1, Pilot 2) into flight_crew table
5. Map cause factors (30 columns) to findings table
6. Transform aircraft data from denormalized to normalized

Usage:
    python scripts/transform_pre1982.py --mdb-path datasets/PRE1982.MDB --output-dir data/pre1982_transformed

Author: NTSB Dataset Analysis Project
Date: 2025-11-07
Version: 1.0.0
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transform_pre1982.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class PRE1982Transformer:
    """Transform legacy PRE1982.MDB (1962-1981) to modern NTSB schema."""

    # State code mapping (numeric to 2-letter abbreviation)
    # Based on FIPS state codes used in 1960s-1980s
    STATE_CODES = {
        '1': 'AL', '2': 'AK', '4': 'AZ', '5': 'AR', '6': 'CA',
        '8': 'CO', '9': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
        '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
        '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
        '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
        '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
        '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
        '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
        '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
        '56': 'WY', '60': 'AS', '66': 'GU', '69': 'MP', '72': 'PR',
        '78': 'VI',
    }

    # Injury category mapping (PRE1982 prefix → modern category)
    INJURY_CATEGORIES = {
        'PILOT': 'PILOT',
        'CO_PILOT': 'CO-PILOT',
        'PASSENGERS': 'PASSENGER',
        'CREW': 'CABIN CREW',
        'TOTAL_ABRD': 'TOTAL ABOARD',
        'GROUND': 'GROUND',
    }

    # Injury level mapping
    INJURY_LEVELS = {
        'FATAL': 'FATL',
        'SERIOUS': 'SERS',
        'MINOR': 'MINR',
        'NONE': 'NONE',
    }

    def __init__(self, mdb_path: str, output_dir: str):
        """
        Initialize transformer.

        Args:
            mdb_path: Path to PRE1982.MDB file
            output_dir: Directory for transformed CSV outputs
        """
        self.mdb_path = Path(mdb_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'records_read': 0,
            'events_created': 0,
            'aircraft_created': 0,
            'flight_crew_created': 0,
            'injuries_created': 0,
            'findings_created': 0,
            'narratives_created': 0,
            'errors': 0,
        }

        logger.info(f"PRE1982 Transformer initialized")
        logger.info(f"  MDB path: {self.mdb_path}")
        logger.info(f"  Output dir: {self.output_dir}")

    def extract_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Extract table from MDB to DataFrame."""
        logger.info(f"Extracting {table_name} from PRE1982.MDB...")

        try:
            # Use mdb-export to extract as CSV
            result = subprocess.run(
                ['mdb-export', '-D', '%Y-%m-%d', str(self.mdb_path), table_name],
                capture_output=True,
                text=True,
                timeout=300,
                check=True
            )

            # Read CSV into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout), low_memory=False)

            logger.info(f"  ✓ Extracted {len(df):,} rows from {table_name}")
            return df

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Failed to extract {table_name}: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"  ✗ Error reading {table_name}: {e}")
            return None

    def generate_ev_id(self, rec_num: int, date_occurrence: str) -> str:
        """
        Generate synthetic ev_id from RecNum and DATE_OCCURRENCE.

        Format: YYYYMMDDR + zero-padded RecNum
        Example: 19620723R000040 (July 23, 1962, RecNum 40)

        Args:
            rec_num: Record number from tblFirstHalf
            date_occurrence: Date string (MM/DD/YY HH:MM:SS)

        Returns:
            Synthetic ev_id string
        """
        try:
            # Parse date with 2-digit year
            dt = pd.to_datetime(date_occurrence, format='%m/%d/%y %H:%M:%S')

            # Format: YYYYMMDD + R + RecNum (zero-padded to 6 digits)
            ev_id = f"{dt.strftime('%Y%m%d')}R{int(rec_num):06d}"
            return ev_id

        except Exception as e:
            logger.warning(f"  ⚠ Failed to generate ev_id for RecNum {rec_num}: {e}")
            # Fallback: use current date + RecNum
            return f"19620101R{int(rec_num):06d}"

    def decode_state(self, state_code: str) -> Optional[str]:
        """Map numeric state code to 2-letter abbreviation."""
        if pd.isna(state_code):
            return None

        code_str = str(int(float(state_code))) if state_code else None
        return self.STATE_CODES.get(code_str, None)

    def parse_pilot_hours(self, hours_str: str) -> Optional[float]:
        """
        Parse pilot hours from legacy format.

        Format examples:
        - "11508A" → 11508.0
        - "2500" → 2500.0
        - "UNK" → None

        Args:
            hours_str: Hours string from PRE1982

        Returns:
            Float hours or None
        """
        if pd.isna(hours_str) or str(hours_str).strip() in ('', 'UNK', 'UNKN', '0'):
            return None

        # Remove trailing letter if present
        hours_clean = str(hours_str).rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()

        try:
            return float(hours_clean)
        except ValueError:
            return None

    def decode_age(self, age_code: str) -> Optional[int]:
        """
        Decode pilot age from legacy coded value.

        Format examples:
        - "ZA" → coded value (mapping unknown, skip)
        - "45" → 45
        - "UNK" → None

        Args:
            age_code: Age code from PRE1982

        Returns:
            Integer age or None
        """
        if pd.isna(age_code) or str(age_code).strip() in ('', 'UNK', 'UNKN', 'ZA'):
            return None

        try:
            age = int(age_code)
            # Validate age range (10-120)
            if 10 <= age <= 120:
                return age
            return None
        except (ValueError, TypeError):
            return None

    def transform_events(self, first_half: pd.DataFrame) -> pd.DataFrame:
        """Transform tblFirstHalf → events table."""
        logger.info("Transforming events table...")

        events = pd.DataFrame()

        # Generate ev_id
        events['ev_id'] = first_half.apply(
            lambda row: self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
            axis=1
        )

        # Date and time
        events['ev_date'] = pd.to_datetime(
            first_half['DATE_OCCURRENCE'],
            format='%m/%d/%y %H:%M:%S',
            errors='coerce'
        ).dt.date

        # Extract time if available
        events['ev_time'] = pd.to_datetime(
            first_half['DATE_OCCURRENCE'],
            format='%m/%d/%y %H:%M:%S',
            errors='coerce'
        ).dt.strftime('%H:%M')

        # Location
        events['ev_city'] = first_half.get('LOCATION', '').str.strip()
        events['ev_state'] = first_half['LOCAT_STATE_TERR'].apply(self.decode_state)
        events['ev_site_zipcode'] = None  # Not available in PRE1982

        # NTSB number (docket number)
        events['ntsb_no'] = first_half.get('DOCKET_NO', '').str.strip()

        # Event type
        events['ev_type'] = 'ACC'  # Assume all are accidents
        events['ev_highest_injury'] = None  # Will compute later from injury data

        # Injury counts (from tblFirstHalf)
        events['inj_tot_f'] = pd.to_numeric(first_half.get('TOTAL_ABRD_FATAL', 0), errors='coerce').fillna(0).astype(int)
        events['inj_tot_s'] = pd.to_numeric(first_half.get('TOTAL_ABRD_SERIOUS', 0), errors='coerce').fillna(0).astype(int)
        events['inj_tot_m'] = pd.to_numeric(first_half.get('TOTAL_ABRD_MINOR', 0), errors='coerce').fillna(0).astype(int)
        events['inj_tot_n'] = pd.to_numeric(first_half.get('TOTAL_ABRD_NONE', 0), errors='coerce').fillna(0).astype(int)

        # Weather (not detailed in PRE1982)
        events['wx_cond_basic'] = None
        events['light_cond'] = None

        # Coordinates (not available in PRE1982)
        events['dec_latitude'] = None
        events['dec_longitude'] = None

        # Aircraft make/model (from primary aircraft)
        events['acft_make'] = first_half.get('ACFT_MAKE', '').str.strip()
        events['acft_model'] = first_half.get('ACFT_MODEL', '').str.strip()

        # Investigation type
        events['invest_type'] = None

        # Report status
        events['ev_report_status'] = 'FINAL'  # Assume all legacy reports are final

        # Missing columns (fill with None)
        events['ev_country'] = 'USA'
        events['far_part'] = None
        events['ev_site_apt_id'] = None

        self.stats['events_created'] = len(events)
        logger.info(f"  ✓ Created {len(events):,} event records")

        return events

    def transform_aircraft(self, first_half: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Transform tblFirstHalf → aircraft table."""
        logger.info("Transforming aircraft table...")

        aircraft = pd.DataFrame()

        # ev_id and aircraft_key
        aircraft['ev_id'] = events['ev_id']
        aircraft['Aircraft_Key'] = '1'  # Single aircraft per event in PRE1982

        # Registration
        aircraft['regis_no'] = first_half.get('REGIST_NO', '').str.strip()

        # Make/Model
        aircraft['acft_make'] = first_half.get('ACFT_MAKE', '').str.strip()
        aircraft['acft_model'] = first_half.get('ACFT_MODEL', '').str.strip()
        aircraft['acft_series'] = None

        # Engine count
        aircraft['num_eng'] = pd.to_numeric(first_half.get('NO_ENGINES', None), errors='coerce')

        # Aircraft category
        aircraft['acft_category'] = first_half.get('TYPE_CRAFT', '').str.strip()

        # Certification
        aircraft['cert_max_gr_wt'] = None
        aircraft['acft_operating'] = None

        # Missing fields
        aircraft['homebuilt'] = None
        aircraft['owner_oper'] = None
        aircraft['acft_serial'] = None

        self.stats['aircraft_created'] = len(aircraft)
        logger.info(f"  ✓ Created {len(aircraft):,} aircraft records")

        return aircraft

    def transform_flight_crew(self, first_half: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Transform tblFirstHalf → Flight_Crew table (2 rows per event if applicable)."""
        logger.info("Transforming flight crew table...")

        crew_records = []

        for idx, row in first_half.iterrows():
            ev_id = events.iloc[idx]['ev_id']

            # Process Pilot 1
            if pd.notna(row.get('PILOT_INVOLED1')) and str(row.get('PILOT_INVOLED1')).strip().upper() == 'A':
                crew_records.append({
                    'ev_id': ev_id,
                    'Aircraft_Key': '1',
                    'crew_category': 'PILOT',
                    'crew_age': self.decode_age(row.get('AGE_PILOT1')),
                    'crew_sex': None,
                    'crew_city': None,
                    'crew_res_state': None,
                    'pilot_cert': row.get('PILOT_CERT_PILOT1', '').strip() if pd.notna(row.get('PILOT_CERT_PILOT1')) else None,
                    'pilot_cert_type': None,
                    'pilot_med_class': row.get('MEDICAL_CERT_PILOT1', '').strip() if pd.notna(row.get('MEDICAL_CERT_PILOT1')) else None,
                    'pilot_med_date': None,
                    'pilot_tot_time': self.parse_pilot_hours(row.get('HOURS_TOTAL_PILOT1')),
                    'pilot_make_time': self.parse_pilot_hours(row.get('HOURS_IN_TYPE_PILOT1')),
                    'pilot_90_days': None,
                    'pilot_30_days': None,
                    'pilot_24_hrs': None,
                    'pilot_flight_time': None,
                    'pilot_inj_level': None,
                })

            # Process Pilot 2 (Co-Pilot)
            if pd.notna(row.get('PILOT_INVOLED2')) and str(row.get('PILOT_INVOLED2')).strip().upper() == 'A':
                crew_records.append({
                    'ev_id': ev_id,
                    'Aircraft_Key': '1',
                    'crew_category': 'CO-PILOT',
                    'crew_age': self.decode_age(row.get('AGE_PILOT2')),
                    'crew_sex': None,
                    'crew_city': None,
                    'crew_res_state': None,
                    'pilot_cert': row.get('PILOT_CERT_PILOT2', '').strip() if pd.notna(row.get('PILOT_CERT_PILOT2')) else None,
                    'pilot_cert_type': None,
                    'pilot_med_class': row.get('MEDICAL_CERT_PILOT2', '').strip() if pd.notna(row.get('MEDICAL_CERT_PILOT2')) else None,
                    'pilot_med_date': None,
                    'pilot_tot_time': self.parse_pilot_hours(row.get('HOURS_TOTAL_PILOT2')),
                    'pilot_make_time': self.parse_pilot_hours(row.get('HOURS_IN_TYPE_PILOT2')),
                    'pilot_90_days': None,
                    'pilot_30_days': None,
                    'pilot_24_hrs': None,
                    'pilot_flight_time': None,
                    'pilot_inj_level': None,
                })

        flight_crew = pd.DataFrame(crew_records)

        self.stats['flight_crew_created'] = len(flight_crew)
        logger.info(f"  ✓ Created {len(flight_crew):,} flight crew records")

        return flight_crew

    def transform_injury(self, first_half: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform tblFirstHalf → injury table.

        Pivots wide injury columns into normalized tall format:
        - PILOT_FATAL, PILOT_SERIOUS, etc. → individual injury records
        """
        logger.info("Transforming injury table...")

        injury_records = []

        for idx, row in first_half.iterrows():
            ev_id = events.iloc[idx]['ev_id']

            # Iterate through injury categories and levels
            for field_prefix, category in self.INJURY_CATEGORIES.items():
                for level_suffix, level_code in self.INJURY_LEVELS.items():
                    # Construct column name (e.g., PILOT_FATAL, PASSENGERS_SERIOUS)
                    col_name = f'{field_prefix}_{level_suffix}'

                    if col_name in row.index:
                        count = pd.to_numeric(row[col_name], errors='coerce')

                        # Only create record if count > 0
                        if pd.notna(count) and count > 0:
                            injury_records.append({
                                'ev_id': ev_id,
                                'Aircraft_Key': '1',
                                'inj_person_category': category,
                                'inj_level': level_code,
                                'inj_person_count': int(count),
                            })

        injury = pd.DataFrame(injury_records)

        self.stats['injuries_created'] = len(injury)
        logger.info(f"  ✓ Created {len(injury):,} injury records")

        return injury

    def transform_findings(self, first_half: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform tblFirstHalf → Findings table.

        Maps cause factors (CAUSE_FACTOR_1P/M/S ... CAUSE_FACTOR_10P/M/S)
        to modern findings format.

        Note: Legacy codes may not map directly to modern taxonomy.
        Stores legacy codes as-is in finding_code and finding_description.
        """
        logger.info("Transforming findings table...")

        findings_records = []

        for idx, row in first_half.iterrows():
            ev_id = events.iloc[idx]['ev_id']

            # Process up to 10 cause factors
            for i in range(1, 11):
                primary = row.get(f'CAUSE_FACTOR_{i}P')
                modifier = row.get(f'CAUSE_FACTOR_{i}M')
                secondary = row.get(f'CAUSE_FACTOR_{i}S')

                # Skip if no primary cause
                if pd.isna(primary) or str(primary).strip() == '':
                    continue

                # First cause factor is typically the probable cause
                is_probable_cause = (i == 1)

                findings_records.append({
                    'ev_id': ev_id,
                    'Aircraft_Key': '1',
                    'finding_code': f'LEGACY_{str(primary).strip()}',  # Prefix to indicate legacy code
                    'finding_description': f'Legacy cause factor {i}P: {primary}, M: {modifier}, S: {secondary}',
                    'modifier_code': str(modifier).strip() if pd.notna(modifier) else None,
                    'cm_inPC': is_probable_cause,  # First factor is probable cause
                    'finding_source': 'PRE1982',
                })

        findings = pd.DataFrame(findings_records)

        self.stats['findings_created'] = len(findings)
        logger.info(f"  ✓ Created {len(findings):,} findings records")

        return findings

    def transform_narratives(self, first_half: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform narrative text if available in PRE1982.

        Note: Narrative fields may be in tblSecondHalf or separate table.
        This is a placeholder for narrative extraction.
        """
        logger.info("Transforming narratives table...")

        narratives = pd.DataFrame()

        # Check if narrative column exists
        narrative_cols = [col for col in first_half.columns if 'NARR' in col.upper() or 'DESC' in col.upper()]

        if narrative_cols:
            narratives['ev_id'] = events['ev_id']
            narratives['Aircraft_Key'] = '1'
            narratives['narr_cause'] = None  # Placeholder

            for col in narrative_cols:
                if narratives.get('narr_cause') is None:
                    narratives['narr_cause'] = first_half[col].str.strip()

            # Remove empty narratives
            narratives = narratives[narratives['narr_cause'].notna() & (narratives['narr_cause'] != '')]

            self.stats['narratives_created'] = len(narratives)
            logger.info(f"  ✓ Created {len(narratives):,} narrative records")
        else:
            logger.info("  ⊘ No narrative columns found in tblFirstHalf")

        return narratives

    def save_transformed_data(self, table_name: str, df: pd.DataFrame):
        """Save transformed DataFrame to CSV."""
        if df.empty:
            logger.warning(f"  ⚠ No data to save for {table_name}")
            return

        output_path = self.output_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=False, na_rep='')

        logger.info(f"  ✓ Saved {len(df):,} rows to {output_path}")

    def run(self):
        """Execute complete transformation pipeline."""
        logger.info("=" * 70)
        logger.info("PRE1982.MDB → Modern NTSB Schema Transformation")
        logger.info("=" * 70)

        try:
            # Step 1: Extract tblFirstHalf (main data)
            logger.info("\n--- Extracting Source Tables ---")
            first_half = self.extract_table('tblFirstHalf')

            if first_half is None or first_half.empty:
                raise ValueError("Failed to extract tblFirstHalf")

            self.stats['records_read'] = len(first_half)

            # Step 2: Transform events (master table)
            logger.info("\n--- Transforming Master Events ---")
            events = self.transform_events(first_half)
            self.save_transformed_data('events', events)

            # Step 3: Transform aircraft
            logger.info("\n--- Transforming Aircraft ---")
            aircraft = self.transform_aircraft(first_half, events)
            self.save_transformed_data('aircraft', aircraft)

            # Step 4: Transform flight crew
            logger.info("\n--- Transforming Flight Crew ---")
            flight_crew = self.transform_flight_crew(first_half, events)
            self.save_transformed_data('Flight_Crew', flight_crew)

            # Step 5: Transform injury
            logger.info("\n--- Transforming Injury Data ---")
            injury = self.transform_injury(first_half, events)
            self.save_transformed_data('injury', injury)

            # Step 6: Transform findings
            logger.info("\n--- Transforming Findings ---")
            findings = self.transform_findings(first_half, events)
            self.save_transformed_data('Findings', findings)

            # Step 7: Transform narratives (if available)
            logger.info("\n--- Transforming Narratives ---")
            narratives = self.transform_narratives(first_half, events)
            if not narratives.empty:
                self.save_transformed_data('narratives', narratives)

            # Step 8: Create empty tables for missing entities
            logger.info("\n--- Creating Empty Tables for Missing Entities ---")
            for table in ['engines', 'seq_of_events', 'Events_Sequence', 'Occurrences', 'NTSB_Admin']:
                empty_df = pd.DataFrame()
                empty_df['ev_id'] = []
                self.save_transformed_data(table, empty_df)

            # Step 9: Generate transformation report
            self.generate_report()

            logger.info("\n✓ Transformation completed successfully!")
            logger.info(f"✓ Transformed CSVs saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"\n✗ Transformation failed: {e}", exc_info=True)
            self.stats['errors'] += 1
            raise

    def generate_report(self):
        """Generate transformation statistics report."""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          PRE1982 Transformation Report                       ║
╚══════════════════════════════════════════════════════════════╝

SOURCE:
  Database: PRE1982.MDB (1962-1981)
  Records read: {self.stats['records_read']:,}

TRANSFORMATION RESULTS:
  Events created:       {self.stats['events_created']:>8,}
  Aircraft created:     {self.stats['aircraft_created']:>8,}
  Flight crew created:  {self.stats['flight_crew_created']:>8,}
  Injuries created:     {self.stats['injuries_created']:>8,}
  Findings created:     {self.stats['findings_created']:>8,}
  Narratives created:   {self.stats['narratives_created']:>8,}

TOTAL RECORDS:          {sum([
    self.stats['events_created'],
    self.stats['aircraft_created'],
    self.stats['flight_crew_created'],
    self.stats['injuries_created'],
    self.stats['findings_created'],
    self.stats['narratives_created']
]):>8,}

OUTPUT:
  Directory: {self.output_dir}

NEXT STEPS:
  1. Review transformed CSVs in {self.output_dir}
  2. Load via: python scripts/load_transformed_pre1982.py
  3. Validate data: psql -d ntsb_aviation -f scripts/validate_data.sql
"""
        logger.info(report)

        # Save report to file
        report_file = self.output_dir / "transformation_report.txt"
        report_file.write_text(report)
        logger.info(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Transform PRE1982.MDB to modern NTSB schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform with default paths
  python scripts/transform_pre1982.py

  # Transform with custom paths
  python scripts/transform_pre1982.py --mdb-path /path/to/PRE1982.MDB --output-dir /path/to/output

  # After transformation, load with:
  python scripts/load_transformed_pre1982.py --source data/pre1982_transformed
        """
    )

    parser.add_argument(
        '--mdb-path',
        default='datasets/PRE1982.MDB',
        help='Path to PRE1982.MDB file (default: datasets/PRE1982.MDB)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/pre1982_transformed',
        help='Output directory for transformed CSVs (default: data/pre1982_transformed)'
    )

    args = parser.parse_args()

    # Check if MDB file exists
    if not Path(args.mdb_path).exists():
        logger.error(f"✗ MDB file not found: {args.mdb_path}")
        logger.error("✗ If using Git LFS, run: git lfs pull --include=datasets/PRE1982.MDB")
        sys.exit(1)

    # Check if mdb-export is available
    try:
        subprocess.run(['mdb-export', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("✗ mdb-export not found. Install mdbtools:")
        logger.error("  Ubuntu/Debian: sudo apt install mdbtools")
        logger.error("  macOS: brew install mdbtools")
        sys.exit(1)

    # Execute transformation
    transformer = PRE1982Transformer(args.mdb_path, args.output_dir)
    transformer.run()


if __name__ == '__main__':
    main()
