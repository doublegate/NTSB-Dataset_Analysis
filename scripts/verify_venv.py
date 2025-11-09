#!/usr/bin/env python
"""
Verify .venv Python 3.13 Virtual Environment Setup

This script verifies that all required packages are installed and working
correctly in the Python 3.13 virtual environment.

Usage:
    source .venv/bin/activate && python scripts/verify_venv.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
import importlib
from typing import List, Tuple


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{text.center(60)}")
    print("=" * 60)


def check_python_version() -> bool:
    """Verify Python version is 3.13.x."""
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    if version.major == 3 and version.minor == 13:
        print("✅ Python 3.13.x detected")
        return True
    else:
        print(
            f"❌ Expected Python 3.13.x, found {version.major}.{version.minor}.{version.micro}"
        )
        return False


def check_package_imports(packages: List[Tuple[str, str]]) -> bool:
    """Check if packages can be imported and get versions."""
    print_header("Package Import Check")
    all_passed = True

    for module_name, display_name in packages:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {display_name:25s} {version}")
        except ImportError as e:
            print(f"❌ {display_name:25s} FAILED: {e}")
            all_passed = False

    return all_passed


def check_database_connectivity() -> bool:
    """Test database connections with psycopg2 and SQLAlchemy."""
    print_header("Database Connectivity Check")
    all_passed = True

    # Test psycopg2
    try:
        import psycopg2

        conn = psycopg2.connect("dbname=ntsb_aviation user=parobek")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        events = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM aircraft")
        aircraft = cur.fetchone()[0]
        print(f"✅ psycopg2: {events:,} events, {aircraft:,} aircraft")
        conn.close()
    except Exception as e:
        print(f"❌ psycopg2 connection failed: {e}")
        all_passed = False

    # Test SQLAlchemy
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine("postgresql://parobek@localhost/ntsb_aviation")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            pg_version = result.scalar()
            print(f"✅ SQLAlchemy: PostgreSQL {pg_version.split()[1]}")
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        all_passed = False

    return all_passed


def check_api_application() -> bool:
    """Test FastAPI application import."""
    print_header("API Application Check")

    try:
        sys.path.insert(0, "api")
        from app.main import app

        print("✅ API import successful")
        print(f"   Title: {app.title}")
        print(f"   Routes: {len(app.routes)} endpoints")
        print(f"   Version: {app.version}")
        return True
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False


def main() -> int:
    """Run all verification checks."""
    print("\n" + "🔍 NTSB Database - Virtual Environment Verification".center(60))

    # Define packages to check
    core_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("sqlalchemy", "SQLAlchemy"),
        ("psycopg2", "psycopg2"),
        ("geoalchemy2", "GeoAlchemy2"),
        ("shapely", "Shapely"),
    ]

    additional_packages = [
        ("pytest", "pytest"),
        ("ruff", "ruff"),
        ("black", "black"),
        ("httpx", "httpx"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("jupyter", "jupyter"),
    ]

    # Run checks
    results = []
    results.append(("Python Version", check_python_version()))
    results.append(("Core Packages", check_package_imports(core_packages)))
    results.append(("Additional Packages", check_package_imports(additional_packages)))
    results.append(("Database Connectivity", check_database_connectivity()))
    results.append(("API Application", check_api_application()))

    # Summary
    print_header("Verification Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:25s} {status}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Virtual environment is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
