#!/usr/bin/env python3
"""
validate_csv.py - Validate extracted CSV files from NTSB MDB database

Phase 1 Sprint 1: Data Validation
Version: 1.0.0
Date: 2025-11-05

Usage:
    python scripts/validate_csv.py
    python scripts/validate_csv.py --data-dir data/
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import json


class CSVValidator:
    """Validate CSV files extracted from NTSB MDB databases"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        self.tables = [
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

    def validate_file_existence(self) -> dict:
        """Check if all expected CSV files exist"""
        print("=" * 60)
        print("STEP 1: File Existence Check")
        print("=" * 60)

        results = {}
        for table in self.tables:
            csv_file = self.data_dir / f"avall-{table}.csv"
            exists = csv_file.exists()
            size_mb = csv_file.stat().st_size / (1024 * 1024) if exists else 0

            results[table] = {
                "exists": exists,
                "path": str(csv_file),
                "size_mb": round(size_mb, 2),
            }

            status = "✓" if exists else "✗"
            print(f"{status} {table:20} {size_mb:8.2f} MB")

        print()
        return results

    def validate_data_quality(self) -> dict:
        """Validate data quality for each CSV file"""
        print("=" * 60)
        print("STEP 2: Data Quality Validation")
        print("=" * 60)

        results = {}
        for table in self.tables:
            csv_file = self.data_dir / f"avall-{table}.csv"

            if not csv_file.exists():
                print(f"✗ {table:20} File not found")
                continue

            try:
                # Read CSV with low_memory=False to avoid dtype warnings
                df = pd.read_csv(csv_file, low_memory=False)

                row_count = len(df)
                col_count = len(df.columns)
                null_pct = (
                    (df.isnull().sum().sum() / (row_count * col_count) * 100)
                    if row_count > 0 and col_count > 0
                    else 0
                )

                results[table] = {
                    "rows": row_count,
                    "columns": col_count,
                    "null_pct": round(null_pct, 2),
                    "memory_mb": round(
                        df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                    ),
                    "columns_list": list(df.columns),
                }

                # Check for key columns
                key_checks = self._validate_key_columns(table, df)
                results[table]["key_checks"] = key_checks

                print(
                    f"✓ {table:20} {row_count:>8,} rows, {col_count:>3} cols, "
                    f"{null_pct:>5.1f}% null"
                )

            except Exception as e:
                print(f"✗ {table:20} Error: {str(e)}")
                results[table] = {"error": str(e)}

        print()
        return results

    def _validate_key_columns(self, table: str, df: pd.DataFrame) -> dict:
        """Validate presence of key columns for each table"""
        checks = {}

        # Define expected key columns for each table
        key_columns = {
            "events": ["ev_id", "ev_date"],
            "aircraft": ["Aircraft_Key", "ev_id"],
            "Flight_Crew": ["ev_id"],
            "injury": ["ev_id"],
            "Findings": ["ev_id"],
            "Occurrences": ["ev_id"],
            "seq_of_events": ["ev_id"],
            "Events_Sequence": ["ev_id"],
            "engines": ["ev_id", "Aircraft_Key"],
            "narratives": ["ev_id"],
            "NTSB_Admin": ["ev_id"],
        }

        if table in key_columns:
            for col in key_columns[table]:
                checks[col] = col in df.columns

        return checks

    def validate_events_data(self) -> dict:
        """Detailed validation for events table"""
        print("=" * 60)
        print("STEP 3: Events Table Detailed Validation")
        print("=" * 60)

        csv_file = self.data_dir / "avall-events.csv"
        if not csv_file.exists():
            print("✗ events.csv not found")
            return {}

        df = pd.read_csv(csv_file, low_memory=False)

        results = {}

        # 1. Check ev_id uniqueness
        ev_id_unique = df["ev_id"].is_unique if "ev_id" in df.columns else False
        results["ev_id_unique"] = ev_id_unique
        print(f"  ev_id unique: {ev_id_unique}")

        # 2. Check date range
        if "ev_date" in df.columns:
            df["ev_date"] = pd.to_datetime(df["ev_date"], errors="coerce")
            min_date = df["ev_date"].min()
            max_date = df["ev_date"].max()
            results["date_range"] = {
                "min": str(min_date),
                "max": str(max_date),
            }
            print(f"  Date range: {min_date} to {max_date}")

        # 3. Check coordinate validity
        if "dec_latitude" in df.columns and "dec_longitude" in df.columns:
            valid_coords = (
                (df["dec_latitude"].between(-90, 90))
                & (df["dec_longitude"].between(-180, 180))
            ).sum()
            total_coords = (
                df[["dec_latitude", "dec_longitude"]].notna().all(axis=1).sum()
            )
            results["coordinates"] = {
                "valid": int(valid_coords),
                "total": int(total_coords),
                "pct": round(
                    (valid_coords / total_coords * 100) if total_coords > 0 else 0, 2
                ),
            }
            print(
                f"  Valid coordinates: {valid_coords:,} / {total_coords:,} "
                f"({results['coordinates']['pct']:.1f}%)"
            )

        # 4. Check injury severity distribution
        if "ev_highest_injury" in df.columns:
            injury_dist = df["ev_highest_injury"].value_counts().to_dict()
            results["injury_distribution"] = injury_dist
            print("  Injury severity distribution:")
            for level, count in injury_dist.items():
                print(f"    {level}: {count:,}")

        print()
        return results

    def validate_relationships(self) -> dict:
        """Validate foreign key relationships"""
        print("=" * 60)
        print("STEP 4: Foreign Key Relationship Validation")
        print("=" * 60)

        results = {}

        # Check if events.csv exists (primary table)
        events_file = self.data_dir / "avall-events.csv"
        if not events_file.exists():
            print("✗ Cannot validate relationships: events.csv not found")
            return {}

        events_df = pd.read_csv(events_file, low_memory=False)
        event_ids = set(events_df["ev_id"]) if "ev_id" in events_df.columns else set()
        print(f"  Base: {len(event_ids):,} unique ev_ids in events table")

        # Check each related table
        related_tables = [
            "aircraft",
            "Flight_Crew",
            "injury",
            "Findings",
            "Events_Sequence",
            "engines",
            "narratives",
            "NTSB_Admin",
        ]

        for table in related_tables:
            csv_file = self.data_dir / f"avall-{table}.csv"
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file, low_memory=False)
            if "ev_id" not in df.columns:
                continue

            table_event_ids = set(df["ev_id"].dropna())
            orphaned = table_event_ids - event_ids
            orphaned_pct = (
                (len(orphaned) / len(table_event_ids) * 100) if table_event_ids else 0
            )

            results[table] = {
                "total_records": len(df),
                "unique_ev_ids": len(table_event_ids),
                "orphaned_ev_ids": len(orphaned),
                "orphaned_pct": round(orphaned_pct, 2),
            }

            status = "✓" if len(orphaned) == 0 else "⚠"
            print(
                f"{status} {table:20} {len(table_event_ids):>6,} ev_ids, "
                f"{len(orphaned):>5} orphaned ({orphaned_pct:.1f}%)"
            )

        print()
        return results

    def generate_summary_report(self, all_results: dict) -> str:
        """Generate summary report"""
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        total_files = len(self.tables)
        files_found = sum(
            1 for r in all_results["file_existence"].values() if r.get("exists", False)
        )
        total_rows = sum(r.get("rows", 0) for r in all_results["data_quality"].values())
        total_size_mb = sum(
            r.get("size_mb", 0) for r in all_results["file_existence"].values()
        )

        print(f"Files found: {files_found}/{total_files}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total size: {total_size_mb:.1f} MB")
        print()

        # Calculate data quality score
        quality_scores = []
        for table_results in all_results["data_quality"].values():
            if "rows" in table_results and table_results["rows"] > 0:
                null_pct = table_results.get("null_pct", 100)
                quality_score = 100 - null_pct
                quality_scores.append(quality_score)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        print(f"Average data quality: {avg_quality:.1f}% (100% - avg null%)")
        print()

        # Check for issues
        issues = []
        if files_found < total_files:
            issues.append(f"Missing {total_files - files_found} CSV files")

        for table, rel_data in all_results["relationships"].items():
            if rel_data.get("orphaned_ev_ids", 0) > 0:
                issues.append(
                    f"{table}: {rel_data['orphaned_ev_ids']} orphaned records"
                )

        if issues:
            print("⚠ Issues Found:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✓ No critical issues found")

        print()

        # Ready for PostgreSQL?
        if (
            files_found >= 9 and avg_quality >= 80
        ):  # At least 9 key tables, 80%+ quality
            print("✓ Data quality sufficient for PostgreSQL migration")
            return "READY"
        else:
            print("⚠ Data quality issues - review before migration")
            return "NEEDS_REVIEW"

    def run_validation(self) -> dict:
        """Run all validation steps"""
        print("\n📊 CSV Validation Report")
        print(f"Data directory: {self.data_dir.absolute()}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(self.data_dir.absolute()),
            "file_existence": self.validate_file_existence(),
            "data_quality": self.validate_data_quality(),
            "events_detail": self.validate_events_data(),
            "relationships": self.validate_relationships(),
        }

        status = self.generate_summary_report(all_results)
        all_results["validation_status"] = status

        # Save to JSON
        report_file = self.data_dir.parent / "csv_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"📄 Full report saved to: {report_file}")
        print()

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate CSV files extracted from NTSB MDB database"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing CSV files (default: data/)",
    )

    args = parser.parse_args()

    validator = CSVValidator(data_dir=args.data_dir)
    results = validator.run_validation()

    # Return exit code based on validation status
    if results["validation_status"] == "READY":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
