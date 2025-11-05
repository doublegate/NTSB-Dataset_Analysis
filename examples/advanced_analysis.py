#!/usr/bin/env python3
"""Advanced analysis examples for NTSB database"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path


def load_data_with_duckdb(table_name: str, year_filter: int = None) -> pd.DataFrame:
    """Load data efficiently using DuckDB"""
    query = f"SELECT * FROM 'data/avall-{table_name}.csv'"
    if year_filter:
        query += f" WHERE ev_year >= {year_filter}"
    return duckdb.query(query).to_df()


def analyze_trends_by_year():
    """Analyze accident trends over time"""
    print("📊 Analyzing trends by year...")

    query = """
    SELECT
        ev_year,
        COUNT(*) as total_events,
        SUM(inj_tot_f) as total_fatalities,
        SUM(inj_tot_s) as total_serious_injuries,
        AVG(inj_tot_f) as avg_fatalities_per_event
    FROM 'data/avall-events.csv'
    WHERE ev_year IS NOT NULL
    GROUP BY ev_year
    ORDER BY ev_year DESC
    LIMIT 20
    """

    df = duckdb.query(query).to_df()
    print(df.to_string(index=False))
    return df


def analyze_by_aircraft_type():
    """Analyze accidents by aircraft make and model"""
    print("\n📊 Top 20 Aircraft by Accident Count...")

    query = """
    SELECT
        a.acft_make as make,
        a.acft_model as model,
        COUNT(*) as accident_count,
        SUM(e.inj_tot_f) as total_fatalities
    FROM 'data/avall-events.csv' e
    JOIN 'data/avall-aircraft.csv' a ON e.ev_id = a.ev_id
    WHERE a.acft_make IS NOT NULL AND a.acft_make != ''
    GROUP BY a.acft_make, a.acft_model
    ORDER BY accident_count DESC
    LIMIT 20
    """

    df = duckdb.query(query).to_df()
    print(df.to_string(index=False))
    return df


def analyze_geographic_patterns():
    """Analyze accidents by geographic location"""
    print("\n📊 Accidents by State (Top 20)...")

    query = """
    SELECT
        ev_state,
        COUNT(*) as total_events,
        SUM(inj_tot_f) as total_fatalities,
        ROUND(AVG(inj_tot_f), 2) as avg_fatalities
    FROM 'data/avall-events.csv'
    WHERE ev_state IS NOT NULL AND ev_state != ''
    GROUP BY ev_state
    ORDER BY total_events DESC
    LIMIT 20
    """

    df = duckdb.query(query).to_df()
    print(df.to_string(index=False))
    return df


def analyze_phase_of_flight():
    """Analyze accidents by phase of operation"""
    print("\n📊 Accidents by Flight Phase...")

    # Note: Phase codes are in the Occurrences table
    query = """
    SELECT
        occurrence_code,
        COUNT(*) as count
    FROM 'data/Occurrences.csv'
    WHERE occurrence_code >= 500 AND occurrence_code <= 610
    GROUP BY occurrence_code
    ORDER BY count DESC
    """

    try:
        df = duckdb.query(query).to_df()
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure Occurrences.csv has been extracted")
        return None


def analyze_causes():
    """Analyze most common findings/causes"""
    print("\n📊 Most Common Findings...")

    query = """
    SELECT
        cause_factor,
        COUNT(*) as count
    FROM 'data/Findings.csv'
    WHERE cause_factor IS NOT NULL
    GROUP BY cause_factor
    ORDER BY count DESC
    LIMIT 20
    """

    try:
        df = duckdb.query(query).to_df()
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure Findings.csv has been extracted")
        return None


def fatal_vs_nonfatal_comparison():
    """Compare fatal vs non-fatal accidents"""
    print("\n📊 Fatal vs Non-Fatal Accidents...")

    query = """
    SELECT
        CASE
            WHEN inj_tot_f > 0 THEN 'Fatal'
            ELSE 'Non-Fatal'
        END as accident_type,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM 'data/avall-events.csv'
    GROUP BY accident_type
    """

    df = duckdb.query(query).to_df()
    print(df.to_string(index=False))
    return df


def seasonal_analysis():
    """Analyze accidents by month/season"""
    print("\n📊 Accidents by Month...")

    query = """
    SELECT
        CAST(SUBSTR(ev_date, 6, 2) AS INTEGER) as month,
        COUNT(*) as accident_count
    FROM 'data/avall-events.csv'
    WHERE ev_date IS NOT NULL AND LENGTH(ev_date) >= 7
    GROUP BY month
    ORDER BY month
    """

    df = duckdb.query(query).to_df()

    # Add month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    df['month_name'] = df['month'].map(month_names)

    print(df[['month', 'month_name', 'accident_count']].to_string(index=False))
    return df


def export_summary_report(output_file: str = 'outputs/summary_report.csv'):
    """Generate and export comprehensive summary report"""
    print(f"\n📝 Generating summary report...")

    query = """
    SELECT
        e.ev_year,
        e.ev_state,
        a.acft_make,
        a.acft_model,
        e.ev_type,
        e.inj_tot_f as fatalities,
        e.inj_tot_s as serious_injuries,
        e.inj_tot_m as minor_injuries,
        e.inj_tot_n as no_injuries
    FROM 'data/avall-events.csv' e
    LEFT JOIN 'data/avall-aircraft.csv' a ON e.ev_id = a.ev_id
    WHERE e.ev_year >= 2020
    ORDER BY e.ev_date DESC
    """

    df = duckdb.query(query).to_df()

    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    # Export to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Report exported to: {output_file}")
    print(f"   Total records: {len(df)}")

    return df


if __name__ == '__main__':
    print("🚀 NTSB Aviation Accident Database - Advanced Analysis")
    print("=" * 60)

    # Check if data exists
    if not Path('data/avall-events.csv').exists():
        print("\n❌ Error: data/avall-events.csv not found")
        print("   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb")
        exit(1)

    try:
        # Run analyses
        analyze_trends_by_year()
        analyze_geographic_patterns()
        analyze_by_aircraft_type()
        fatal_vs_nonfatal_comparison()
        seasonal_analysis()

        # Optional analyses (require additional tables)
        if Path('data/Occurrences.csv').exists():
            analyze_phase_of_flight()

        if Path('data/Findings.csv').exists():
            analyze_causes()

        # Export report
        export_summary_report()

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("   Make sure all required CSV files have been extracted")
