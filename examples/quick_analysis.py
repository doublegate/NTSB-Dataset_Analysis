#!/usr/bin/env python3
"""Quick analysis example for NTSB database"""

import pandas as pd
import duckdb
from pathlib import Path
import sys


# Example: Query events from CSV using DuckDB
def analyze_recent_events(year=2023):
    """
    Analyze recent events from NTSB database

    Args:
        year: Minimum year to filter events (default: 2023)

    Returns:
        DataFrame of recent events
    """
    # Validate year parameter
    if not isinstance(year, int) or year < 1962 or year > 2100:
        raise ValueError(f"Invalid year: {year}. Must be between 1962 and 2100")

    query = f"""
    SELECT
        ev_id,
        ev_date,
        ev_type,
        ev_state,
        ev_city,
        COALESCE(inj_tot_f, 0) as fatalities,
        COALESCE(inj_tot_s, 0) as serious_injuries
    FROM 'data/avall-events.csv'
    WHERE ev_year >= {year}
      AND ev_year IS NOT NULL
      AND ev_id IS NOT NULL
    ORDER BY ev_date DESC
    LIMIT 100
    """

    try:
        df = duckdb.query(query).to_df()

        if len(df) == 0:
            print(f"⚠️  No events found for year >= {year}")

        return df
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        raise


# Example: Read with pandas
def load_events(low_memory=True):
    """
    Load all events using pandas

    Args:
        low_memory: Use low_memory mode for large files

    Returns:
        DataFrame of all events
    """
    try:
        df = pd.read_csv("data/avall-events.csv", low_memory=low_memory)
        print(f"📊 Loaded {len(df):,} events from database")
        return df
    except FileNotFoundError:
        print("❌ Error: data/avall-events.csv not found")
        print(
            "   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb"
        )
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


if __name__ == "__main__":
    print("📊 NTSB Aviation Accident Database - Quick Analysis")
    print("=" * 60)

    # Check if data exists
    if not Path("data/avall-events.csv").exists():
        print("\n❌ Error: data/avall-events.csv not found")
        print(
            "   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb"
        )
        sys.exit(1)

    try:
        print("\n📊 Loading recent events (2023+)...")
        df = analyze_recent_events(2023)

        if len(df) > 0:
            print(f"✅ Found {len(df)} events")
            print("\nFirst 5 events:")
            print(df.head())

            # Basic statistics
            print("\nStatistics:")
            print(f"  Total fatalities: {df['fatalities'].sum():.0f}")
            print(f"  Total serious injuries: {df['serious_injuries'].sum():.0f}")
            print(f"  Fatal accidents: {(df['fatalities'] > 0).sum()}")
            print(f"  Non-fatal accidents: {(df['fatalities'] == 0).sum()}")

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)
