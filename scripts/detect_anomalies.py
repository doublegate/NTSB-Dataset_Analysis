#!/usr/bin/env python3
"""
Data Quality Anomaly Detection for NTSB Aviation Database.

Performs automated quality checks on loaded data and flags anomalies:
1. Missing critical fields (ev_id, ev_date, coordinates)
2. Outlier detection (coordinates outside valid bounds)
3. Statistical anomalies (unexpected drops in event counts)
4. Referential integrity (orphaned child records)
5. Duplicate detection (same ev_id loaded multiple times)

Author: NTSB ETL Team
Created: 2025-11-07
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

import psycopg2
from psycopg2.extras import RealDictCursor


# =============================================================================
# DATABASE CONNECTION
# =============================================================================


def get_db_connection():
    """Connect to NTSB PostgreSQL database using environment variables."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("NTSB_DB_HOST", "localhost"),
            port=int(os.getenv("NTSB_DB_PORT", "5432")),
            database=os.getenv("NTSB_DB_NAME", "ntsb_aviation"),
            user=os.getenv("NTSB_DB_USER", "parobek"),
            password=os.getenv("NTSB_DB_PASSWORD", ""),
            cursor_factory=RealDictCursor,
        )
        return conn
    except psycopg2.Error as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


# =============================================================================
# ANOMALY CHECK FUNCTIONS
# =============================================================================


def check_missing_critical_fields(conn, lookback_days: int = 35) -> Dict[str, Any]:
    """
    Check 1: Missing critical fields in recent events.

    Args:
        conn: Database connection
        lookback_days: Number of days to look back for recent data

    Returns:
        Dictionary with check results
    """
    print("\n🔍 Check 1: Missing Critical Fields")

    with conn.cursor() as cursor:
        query = """
        SELECT
            COUNT(*) as total_events,
            COUNT(*) FILTER (WHERE ev_id IS NULL) as missing_ev_id,
            COUNT(*) FILTER (WHERE ev_date IS NULL) as missing_ev_date,
            COUNT(*) FILTER (WHERE dec_latitude IS NULL) as missing_latitude,
            COUNT(*) FILTER (WHERE dec_longitude IS NULL) as missing_longitude,
            COUNT(*) FILTER (WHERE ev_type IS NULL) as missing_ev_type,
            COUNT(*) FILTER (WHERE acft_make IS NULL AND acft_model IS NULL) as missing_aircraft_info
        FROM events
        WHERE ev_date >= CURRENT_DATE - INTERVAL '%s days'
           OR created_at >= CURRENT_DATE - INTERVAL '%s days';
        """ % (lookback_days, lookback_days)

        cursor.execute(query)
        result = cursor.fetchone()

    # Calculate anomaly threshold (>1% missing is anomalous)
    total = result["total_events"]
    threshold = total * 0.01

    anomalies = []
    if result["missing_ev_id"] > 0:
        anomalies.append(f"Missing ev_id: {result['missing_ev_id']} events")
    if result["missing_ev_date"] > 0:
        anomalies.append(f"Missing ev_date: {result['missing_ev_date']} events")
    if result["missing_latitude"] > threshold:
        anomalies.append(
            f"Missing latitude: {result['missing_latitude']} events ({result['missing_latitude'] / total * 100:.1f}%)"
        )
    if result["missing_longitude"] > threshold:
        anomalies.append(
            f"Missing longitude: {result['missing_longitude']} events ({result['missing_longitude'] / total * 100:.1f}%)"
        )

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "⚠️  WARNING"

    print(f"{status} - Total events checked: {total}")
    if anomalies:
        for anomaly in anomalies:
            print(f"  - {anomaly}")

    return {
        "check": "missing_critical_fields",
        "passed": passed,
        "severity": "WARNING" if not passed else "INFO",
        "total_events": total,
        "anomalies": anomalies,
        "details": dict(result),
    }


def check_coordinate_outliers(conn, lookback_days: int = 35) -> Dict[str, Any]:
    """
    Check 2: Coordinate outliers (lat/lon outside valid bounds).

    Args:
        conn: Database connection
        lookback_days: Number of days to look back

    Returns:
        Dictionary with check results
    """
    print("\n🔍 Check 2: Coordinate Outliers")

    with conn.cursor() as cursor:
        query = """
        SELECT
            COUNT(*) as total_with_coords,
            COUNT(*) FILTER (WHERE dec_latitude < -90 OR dec_latitude > 90) as invalid_latitude,
            COUNT(*) FILTER (WHERE dec_longitude < -180 OR dec_longitude > 180) as invalid_longitude,
            COUNT(*) FILTER (WHERE dec_latitude = 0 AND dec_longitude = 0) as null_island
        FROM events
        WHERE (ev_date >= CURRENT_DATE - INTERVAL '%s days'
           OR created_at >= CURRENT_DATE - INTERVAL '%s days')
          AND dec_latitude IS NOT NULL
          AND dec_longitude IS NOT NULL;
        """ % (lookback_days, lookback_days)

        cursor.execute(query)
        result = cursor.fetchone()

    anomalies = []
    if result["invalid_latitude"] > 0:
        anomalies.append(f"Invalid latitude: {result['invalid_latitude']} events")
    if result["invalid_longitude"] > 0:
        anomalies.append(f"Invalid longitude: {result['invalid_longitude']} events")
    if result["null_island"] > 10:  # More than 10 events at 0,0 is suspicious
        anomalies.append(
            f"'Null Island' (0,0): {result['null_island']} events (suspicious)"
        )

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "⚠️  WARNING"

    print(f"{status} - Total events with coordinates: {result['total_with_coords']}")
    if anomalies:
        for anomaly in anomalies:
            print(f"  - {anomaly}")

    return {
        "check": "coordinate_outliers",
        "passed": passed,
        "severity": "WARNING" if not passed else "INFO",
        "total_checked": result["total_with_coords"],
        "anomalies": anomalies,
        "details": dict(result),
    }


def check_event_count_drop(conn) -> Dict[str, Any]:
    """
    Check 3: Statistical anomalies (sudden drop in monthly event counts).

    Args:
        conn: Database connection

    Returns:
        Dictionary with check results
    """
    print("\n🔍 Check 3: Statistical Anomalies (Event Count Drop)")

    with conn.cursor() as cursor:
        # Get last 12 months of event counts
        query = """
        SELECT
            DATE_TRUNC('month', ev_date) as month,
            COUNT(*) as event_count
        FROM events
        WHERE ev_date >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY DATE_TRUNC('month', ev_date)
        ORDER BY month DESC
        LIMIT 12;
        """

        cursor.execute(query)
        monthly_counts = cursor.fetchall()

    if len(monthly_counts) < 2:
        return {
            "check": "event_count_drop",
            "passed": True,
            "severity": "INFO",
            "anomalies": [],
            "details": {"message": "Not enough data for statistical analysis"},
        }

    # Calculate average and check for sudden drops
    counts = [row["event_count"] for row in monthly_counts]
    avg_count = sum(counts) / len(counts)
    latest_count = counts[0]

    # Anomaly if latest month is <50% of average
    threshold = avg_count * 0.5
    anomalies = []

    if latest_count < threshold:
        anomalies.append(
            f"Latest month ({monthly_counts[0]['month'].strftime('%Y-%m')}) has {latest_count} events "
            f"(avg: {avg_count:.0f}, drop of {(1 - latest_count / avg_count) * 100:.1f}%)"
        )

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "⚠️  WARNING"

    print(
        f"{status} - Latest month: {latest_count} events, 12-month avg: {avg_count:.0f}"
    )
    if anomalies:
        for anomaly in anomalies:
            print(f"  - {anomaly}")

    return {
        "check": "event_count_drop",
        "passed": passed,
        "severity": "WARNING" if not passed else "INFO",
        "latest_count": latest_count,
        "average_count": avg_count,
        "anomalies": anomalies,
        "details": {"monthly_counts": [dict(row) for row in monthly_counts]},
    }


def check_referential_integrity(conn, lookback_days: int = 35) -> Dict[str, Any]:
    """
    Check 4: Referential integrity (orphaned child records).

    Args:
        conn: Database connection
        lookback_days: Number of days to look back

    Returns:
        Dictionary with check results
    """
    print("\n🔍 Check 4: Referential Integrity (Orphaned Records)")

    anomalies = []

    with conn.cursor() as cursor:
        # Check orphaned aircraft records
        cursor.execute("""
            SELECT COUNT(*) as orphaned_aircraft
            FROM aircraft a
            LEFT JOIN events e ON a.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """)
        orphaned_aircraft = cursor.fetchone()["orphaned_aircraft"]

        # Check orphaned findings
        cursor.execute("""
            SELECT COUNT(*) as orphaned_findings
            FROM findings f
            LEFT JOIN events e ON f.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """)
        orphaned_findings = cursor.fetchone()["orphaned_findings"]

        # Check orphaned narratives
        cursor.execute("""
            SELECT COUNT(*) as orphaned_narratives
            FROM narratives n
            LEFT JOIN events e ON n.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """)
        orphaned_narratives = cursor.fetchone()["orphaned_narratives"]

    if orphaned_aircraft > 0:
        anomalies.append(f"Orphaned aircraft: {orphaned_aircraft} records")
    if orphaned_findings > 0:
        anomalies.append(f"Orphaned findings: {orphaned_findings} records")
    if orphaned_narratives > 0:
        anomalies.append(f"Orphaned narratives: {orphaned_narratives} records")

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "⚠️  WARNING"

    print(f"{status}")
    if anomalies:
        for anomaly in anomalies:
            print(f"  - {anomaly}")
    else:
        print("  - No orphaned records found")

    return {
        "check": "referential_integrity",
        "passed": passed,
        "severity": "WARNING" if not passed else "INFO",
        "anomalies": anomalies,
        "details": {
            "orphaned_aircraft": orphaned_aircraft,
            "orphaned_findings": orphaned_findings,
            "orphaned_narratives": orphaned_narratives,
        },
    }


def check_duplicates(conn, lookback_days: int = 35) -> Dict[str, Any]:
    """
    Check 5: Duplicate detection (same ev_id appears multiple times).

    Args:
        conn: Database connection
        lookback_days: Number of days to look back

    Returns:
        Dictionary with check results
    """
    print("\n🔍 Check 5: Duplicate Detection")

    with conn.cursor() as cursor:
        query = """
        SELECT
            ev_id,
            COUNT(*) as duplicate_count
        FROM events
        WHERE ev_date >= CURRENT_DATE - INTERVAL '%s days'
           OR created_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY ev_id
        HAVING COUNT(*) > 1;
        """ % (lookback_days, lookback_days)

        cursor.execute(query)
        duplicates = cursor.fetchall()

    anomalies = []
    if duplicates:
        for dup in duplicates:
            anomalies.append(
                f"Duplicate ev_id '{dup['ev_id']}': {dup['duplicate_count']} occurrences"
            )

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "❌ CRITICAL"

    print(f"{status} - Duplicates found: {len(duplicates)}")
    if anomalies:
        for anomaly in anomalies[:10]:  # Show first 10
            print(f"  - {anomaly}")
        if len(anomalies) > 10:
            print(f"  - ... and {len(anomalies) - 10} more")

    return {
        "check": "duplicate_detection",
        "passed": passed,
        "severity": "CRITICAL" if not passed else "INFO",
        "duplicate_count": len(duplicates),
        "anomalies": anomalies,
        "details": {"duplicates": [dict(d) for d in duplicates[:20]]},  # Limit to 20
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_all_checks(lookback_days: int = 35) -> List[Dict[str, Any]]:
    """
    Run all anomaly detection checks.

    Args:
        lookback_days: Number of days to look back for recent data

    Returns:
        List of check results
    """
    print("=" * 80)
    print("NTSB Aviation Data Quality Anomaly Detection")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback Period: {lookback_days} days")

    conn = get_db_connection()
    results = []

    try:
        results.append(check_missing_critical_fields(conn, lookback_days))
        results.append(check_coordinate_outliers(conn, lookback_days))
        results.append(check_event_count_drop(conn))
        results.append(check_referential_integrity(conn, lookback_days))
        results.append(check_duplicates(conn, lookback_days))
    finally:
        conn.close()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed_checks = sum(1 for r in results if r["passed"])
    total_checks = len(results)
    total_anomalies = sum(len(r.get("anomalies", [])) for r in results)

    print(f"✅ Passed: {passed_checks}/{total_checks}")
    print(f"⚠️  Anomalies Found: {total_anomalies}")

    # Determine overall status
    critical_failures = [
        r for r in results if not r["passed"] and r.get("severity") == "CRITICAL"
    ]
    warnings = [
        r for r in results if not r["passed"] and r.get("severity") == "WARNING"
    ]

    if critical_failures:
        print(
            f"\n❌ CRITICAL: {len(critical_failures)} check(s) failed with critical severity"
        )
        return_code = 2
    elif warnings:
        print(f"\n⚠️  WARNING: {len(warnings)} check(s) failed with warning severity")
        return_code = 1
    else:
        print("\n✅ SUCCESS: All checks passed!")
        return_code = 0

    # Add summary to results
    results.append(
        {
            "check": "summary",
            "passed": return_code == 0,
            "severity": "CRITICAL"
            if critical_failures
            else "WARNING"
            if warnings
            else "INFO",
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "total_anomalies": total_anomalies,
            "return_code": return_code,
        }
    )

    return results


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="NTSB Aviation Data Quality Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=35,
        help="Number of days to look back for recent data (default: 35)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results (optional)",
    )

    args = parser.parse_args()

    # Run checks
    results = run_all_checks(lookback_days=args.lookback_days)

    # Write results to file if specified
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": args.lookback_days,
            "checks": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n📄 Results written to: {args.output}")

    # Exit with appropriate code
    summary = next(r for r in results if r["check"] == "summary")
    sys.exit(summary["return_code"])


if __name__ == "__main__":
    main()
