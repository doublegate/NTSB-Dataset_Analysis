#!/usr/bin/env python3
"""Quick analysis example for NTSB database"""

import pandas as pd
import duckdb

# Example: Query events from CSV using DuckDB
def analyze_recent_events(year=2023):
    query = f"""
    SELECT
        ev_id,
        ev_date,
        ev_type,
        ev_state,
        ev_city,
        inj_tot_f as fatalities,
        inj_tot_s as serious_injuries
    FROM 'data/events.csv'
    WHERE ev_year >= {year}
    ORDER BY ev_date DESC
    LIMIT 100
    """

    return duckdb.query(query).to_df()

# Example: Read with pandas
def load_events():
    return pd.read_csv('data/events.csv')

if __name__ == '__main__':
    print("📊 Loading recent events...")
    df = analyze_recent_events(2023)
    print(f"Found {len(df)} events")
    print(df.head())
