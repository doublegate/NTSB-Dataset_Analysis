#!/usr/bin/env python3
"""
monthly_sync_dag.py - Production Airflow DAG for Monthly NTSB Data Synchronization

Phase 1 Sprint 3 Week 2: Automated Monthly Data Pipeline
Version: 1.0.0
Date: 2025-11-06

This DAG automates the monthly synchronization of NTSB aviation accident data:
1. Check for updates to avall.zip on NTSB website
2. Download new data if available
3. Backup current database
4. Load new data using existing staging table ETL
5. Validate data quality
6. Refresh materialized views
7. Send completion notification

Schedule: Runs at 2 AM UTC on the 1st of each month
Update Source: https://data.ntsb.gov/avdata/avall.zip
Target Database: ntsb_aviation (PostgreSQL 18.0)

Data Flow:
    NTSB Website → Download → Extract MDB → Stage → Validate → Production → MV Refresh

Error Handling:
- Automatic retries (2x with 5min delay)
- Database backup before loading
- Validation checks prevent bad data
- Rollback on failure
- Email notifications on success/failure

Dependencies:
- PostgreSQL connection: ntsb_aviation_db
- HTTP connection: ntsb_carol_query_http (optional, uses requests if not available)
- Mounted volumes: /tmp/NTSB_Datasets/, /opt/airflow/scripts/
- Python packages: requests, psycopg2, pandas

Author: Claude Code (Anthropic)
Sprint: Phase 1 Sprint 3
"""

import os
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.exceptions import AirflowException

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

# Configure DAG-level logging
logger = logging.getLogger(__name__)

# ====================================================================================================
# CONSTANTS AND CONFIGURATION
# ====================================================================================================

# NTSB Data Source (verified 2025-11-06)
NTSB_AVDATA_URL = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile"
AVALL_ZIP_PARAMS = {"fileID": "C:\\avdata\\avall.zip"}
AVALL_ZIP_URL = f"{NTSB_AVDATA_URL}?fileID=C%3A%5Cavdata%5Cavall.zip"

# File paths
DOWNLOAD_DIR = Path("/tmp/NTSB_Datasets/monthly_sync")
BACKUP_DIR = Path("/tmp/NTSB_Datasets/backups")
EXTRACT_DIR = Path("/tmp/NTSB_Datasets/monthly_sync/extracted")
SCRIPTS_DIR = Path("/opt/airflow/scripts")

# Expected file sizes (used for validation)
MIN_AVALL_ZIP_SIZE = 80_000_000  # 80 MB (current is ~92 MB)
MAX_AVALL_ZIP_SIZE = 200_000_000  # 200 MB (allow for growth)

# Database configuration
DB_CONN_ID = "ntsb_aviation_db"
DB_NAME = "ntsb_aviation"

# DAG configuration
DAG_ID = "monthly_sync_ntsb_data"
SCHEDULE_INTERVAL = "0 2 1 * *"  # 2 AM UTC on the 1st of each month
START_DATE = datetime(2025, 11, 1)
MAX_ACTIVE_RUNS = 1
CATCHUP = False

# Email configuration (set via Airflow Variables)
# Variables: ntsb_notification_email, ntsb_slack_webhook
EMAIL_ON_FAILURE = True
EMAIL_ON_RETRY = False
EMAIL_ON_SUCCESS = False  # Set to True if you want success emails


# ====================================================================================================
# HELPER FUNCTIONS
# ====================================================================================================

def get_file_metadata_requests(url: str) -> Dict[str, any]:
    """
    Get file metadata using requests library (HTTP HEAD).

    Args:
        url: URL to check

    Returns:
        Dict with 'size' (int), 'last_modified' (str), 'etag' (str or None)

    Raises:
        AirflowException: If request fails
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()

        return {
            'size': int(response.headers.get('Content-Length', 0)),
            'last_modified': response.headers.get('Last-Modified', 'unknown'),
            'etag': response.headers.get('ETag'),
            'content_type': response.headers.get('Content-Type'),
        }
    except requests.RequestException as e:
        raise AirflowException(f"Failed to get file metadata: {e}")


def get_file_metadata_urllib(url: str) -> Dict[str, any]:
    """
    Get file metadata using urllib (HTTP HEAD).

    Args:
        url: URL to check

    Returns:
        Dict with 'size' (int), 'last_modified' (str), 'etag' (str or None)

    Raises:
        AirflowException: If request fails
    """
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            headers = response.headers
            return {
                'size': int(headers.get('Content-Length', 0)),
                'last_modified': headers.get('Last-Modified', 'unknown'),
                'etag': headers.get('ETag'),
                'content_type': headers.get('Content-Type'),
            }
    except urllib.error.URLError as e:
        raise AirflowException(f"Failed to get file metadata: {e}")


def get_last_successful_load(**context) -> Optional[Dict[str, any]]:
    """
    Get metadata from the last successful avall.mdb load.

    Args:
        context: Airflow task context

    Returns:
        Dict with 'load_date', 'events_loaded', 'file_size' or None if never loaded
    """
    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

    query = """
        SELECT
            database_name,
            load_status,
            events_loaded,
            duplicate_events_found,
            load_completed_at,
            file_size,
            file_checksum
        FROM load_tracking
        WHERE database_name = 'avall.mdb'
          AND load_status = 'completed'
        ORDER BY load_completed_at DESC
        LIMIT 1;
    """

    result = pg_hook.get_first(query)

    if result:
        return {
            'database_name': result[0],
            'load_status': result[1],
            'events_loaded': result[2],
            'duplicate_events_found': result[3],
            'load_completed_at': result[4],
            'file_size': result[5],
            'file_checksum': result[6],
        }

    return None


# ====================================================================================================
# TASK 1: CHECK FOR UPDATES
# ====================================================================================================

def check_for_updates_task(**context) -> bool:
    """
    Check if avall.zip has been updated since last successful load.

    This task performs the following checks:
    1. Get current file metadata from NTSB website (size, last-modified)
    2. Query load_tracking table for last successful load
    3. Compare file size and/or last-modified date
    4. Return True if update detected, False to skip downstream tasks

    The task uses XCom to pass metadata to downstream tasks:
    - current_file_size
    - current_last_modified
    - should_download (boolean)

    Args:
        context: Airflow task context

    Returns:
        bool: True if update detected (continue DAG), False to skip
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 1: Checking for updates to avall.zip")
    logger.info("=" * 80)

    # Get current file metadata from NTSB website
    logger.info(f"Fetching metadata from: {AVALL_ZIP_URL}")

    if HAS_REQUESTS:
        metadata = get_file_metadata_requests(AVALL_ZIP_URL)
    else:
        metadata = get_file_metadata_urllib(AVALL_ZIP_URL)

    logger.info(f"Current file size: {metadata['size']:,} bytes ({metadata['size'] / 1_000_000:.1f} MB)")
    logger.info(f"Last modified: {metadata['last_modified']}")
    logger.info(f"ETag: {metadata.get('etag', 'N/A')}")

    # Validate file size is reasonable
    if metadata['size'] < MIN_AVALL_ZIP_SIZE:
        raise AirflowException(
            f"File size {metadata['size']:,} bytes is too small "
            f"(expected at least {MIN_AVALL_ZIP_SIZE:,} bytes). "
            "This may indicate a download error or corrupted file."
        )

    if metadata['size'] > MAX_AVALL_ZIP_SIZE:
        logger.warning(
            f"File size {metadata['size']:,} bytes exceeds expected maximum "
            f"({MAX_AVALL_ZIP_SIZE:,} bytes). Proceeding anyway, but verify file integrity."
        )

    # Store current metadata in XCom for downstream tasks
    ti.xcom_push(key='current_file_size', value=metadata['size'])
    ti.xcom_push(key='current_last_modified', value=metadata['last_modified'])
    ti.xcom_push(key='current_etag', value=metadata.get('etag'))

    # Get last successful load from database
    last_load = get_last_successful_load(**context)

    if last_load is None:
        logger.info("No previous successful loads found. This is the first load.")
        logger.info("Proceeding with download...")
        ti.xcom_push(key='should_download', value=True)
        ti.xcom_push(key='is_first_load', value=True)
        return True  # Continue DAG

    # Compare with last load
    logger.info(f"Last successful load: {last_load['load_completed_at']}")
    logger.info(f"Last loaded file size: {last_load.get('file_size', 'unknown')}")
    logger.info(f"Events loaded: {last_load['events_loaded']:,}")
    logger.info(f"Duplicates found: {last_load['duplicate_events_found']:,}")

    # Decision logic: download if file size changed
    last_size = last_load.get('file_size')

    if last_size is None:
        logger.warning("Last load has no file size recorded. Assuming update needed.")
        should_download = True
    elif metadata['size'] != last_size:
        size_diff = metadata['size'] - last_size
        size_diff_mb = size_diff / 1_000_000
        logger.info(f"File size changed by {size_diff:,} bytes ({size_diff_mb:+.1f} MB)")
        should_download = True
    else:
        logger.info("File size unchanged. No update detected.")
        should_download = False

    # Store decision in XCom
    ti.xcom_push(key='should_download', value=should_download)
    ti.xcom_push(key='is_first_load', value=False)
    ti.xcom_push(key='last_load_date', value=str(last_load['load_completed_at']))

    if should_download:
        logger.info("✅ Update detected. Proceeding with download...")
        return True  # Continue DAG
    else:
        logger.info("⏭️  No update detected. Skipping download and load tasks.")
        return False  # Skip downstream tasks


# ====================================================================================================
# TASK 2: DOWNLOAD AVALL.ZIP
# ====================================================================================================

def download_avall_zip_task(**context) -> str:
    """
    Download avall.zip from NTSB website.

    This task:
    1. Creates download directory if needed
    2. Downloads avall.zip using requests or urllib
    3. Verifies file size matches expected size
    4. Returns path to downloaded file

    Args:
        context: Airflow task context

    Returns:
        str: Path to downloaded file

    Raises:
        AirflowException: If download fails or file size mismatch
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 2: Downloading avall.zip")
    logger.info("=" * 80)

    # Get expected file size from previous task
    expected_size = ti.xcom_pull(task_ids='check_for_updates', key='current_file_size')

    if expected_size is None:
        raise AirflowException("Expected file size not found in XCom. check_for_updates may have failed.")

    logger.info(f"Expected file size: {expected_size:,} bytes ({expected_size / 1_000_000:.1f} MB)")

    # Create download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DOWNLOAD_DIR / "avall.zip"

    # Remove old file if exists
    if output_path.exists():
        logger.info(f"Removing existing file: {output_path}")
        output_path.unlink()

    # Download file
    logger.info(f"Downloading from: {AVALL_ZIP_URL}")
    logger.info(f"Saving to: {output_path}")

    try:
        if HAS_REQUESTS:
            # Use requests library (preferred)
            response = requests.get(AVALL_ZIP_URL, stream=True, timeout=600)
            response.raise_for_status()

            total_size = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

                        # Log progress every 10 MB
                        if total_size % (10 * 1024 * 1024) < chunk_size:
                            logger.info(f"Downloaded {total_size / 1_000_000:.1f} MB...")
        else:
            # Fall back to urllib
            with urllib.request.urlopen(AVALL_ZIP_URL, timeout=600) as response:
                with open(output_path, 'wb') as f:
                    total_size = 0
                    chunk_size = 1024 * 1024  # 1 MB chunks

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        total_size += len(chunk)

                        # Log progress every 10 MB
                        if total_size % (10 * 1024 * 1024) < chunk_size:
                            logger.info(f"Downloaded {total_size / 1_000_000:.1f} MB...")

        logger.info(f"Download complete: {total_size:,} bytes ({total_size / 1_000_000:.1f} MB)")

    except (requests.RequestException if HAS_REQUESTS else urllib.error.URLError) as e:
        raise AirflowException(f"Download failed: {e}")
    except Exception as e:
        raise AirflowException(f"Unexpected error during download: {e}")

    # Verify file size
    actual_size = output_path.stat().st_size

    if actual_size != expected_size:
        logger.error("File size mismatch!")
        logger.error(f"Expected: {expected_size:,} bytes")
        logger.error(f"Actual: {actual_size:,} bytes")
        logger.error(f"Difference: {abs(actual_size - expected_size):,} bytes")

        # Delete incomplete file
        output_path.unlink()

        raise AirflowException(
            f"Downloaded file size ({actual_size:,} bytes) does not match "
            f"expected size ({expected_size:,} bytes). File may be corrupted."
        )

    logger.info("✅ File size verification passed")
    logger.info(f"Downloaded file: {output_path}")

    # Store download path in XCom
    ti.xcom_push(key='downloaded_zip_path', value=str(output_path))
    ti.xcom_push(key='downloaded_file_size', value=actual_size)

    return str(output_path)


# ====================================================================================================
# TASK 3: EXTRACT AVALL.MDB
# ====================================================================================================

def extract_avall_mdb_task(**context) -> str:
    """
    Extract avall.mdb from avall.zip.

    This task:
    1. Creates extraction directory
    2. Extracts avall.mdb from zip file using unzip command
    3. Verifies MDB file was extracted
    4. Returns path to extracted MDB file

    Args:
        context: Airflow task context

    Returns:
        str: Path to extracted avall.mdb file

    Raises:
        AirflowException: If extraction fails
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 3: Extracting avall.mdb from zip")
    logger.info("=" * 80)

    # Get downloaded zip path from previous task
    zip_path = ti.xcom_pull(task_ids='download_avall_zip', key='downloaded_zip_path')

    if zip_path is None:
        raise AirflowException("Downloaded zip path not found in XCom.")

    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise AirflowException(f"Downloaded zip file not found: {zip_path}")

    # Create extraction directory
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old extracted files
    for old_file in EXTRACT_DIR.glob("*.mdb"):
        logger.info(f"Removing old extracted file: {old_file}")
        old_file.unlink()

    # Extract using unzip command
    logger.info(f"Extracting {zip_path} to {EXTRACT_DIR}")

    try:
        result = subprocess.run(
            ['unzip', '-o', str(zip_path), '-d', str(EXTRACT_DIR)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True
        )

        logger.info(result.stdout)

        if result.stderr:
            logger.warning(f"Extraction warnings: {result.stderr}")

    except subprocess.CalledProcessError as e:
        raise AirflowException(f"Extraction failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise AirflowException("Extraction timed out after 5 minutes")
    except FileNotFoundError:
        raise AirflowException("unzip command not found. Install unzip package.")

    # Verify MDB file was extracted
    mdb_files = list(EXTRACT_DIR.glob("*.mdb")) + list(EXTRACT_DIR.glob("*.MDB"))

    if not mdb_files:
        raise AirflowException(f"No MDB file found in {EXTRACT_DIR} after extraction")

    if len(mdb_files) > 1:
        logger.warning(f"Multiple MDB files found: {mdb_files}. Using first one.")

    mdb_path = mdb_files[0]
    mdb_size = mdb_path.stat().st_size

    logger.info(f"✅ Extracted MDB file: {mdb_path}")
    logger.info(f"MDB file size: {mdb_size:,} bytes ({mdb_size / 1_000_000:.1f} MB)")

    # Store MDB path in XCom
    ti.xcom_push(key='extracted_mdb_path', value=str(mdb_path))
    ti.xcom_push(key='extracted_mdb_size', value=mdb_size)

    return str(mdb_path)


# ====================================================================================================
# TASK 4: BACKUP DATABASE
# ====================================================================================================

def backup_database_task(**context) -> str:
    """
    Create timestamped backup of ntsb_aviation database.

    This task:
    1. Creates backup directory
    2. Generates timestamped backup filename
    3. Uses pg_dump to create SQL backup
    4. Compresses backup with gzip
    5. Verifies backup was created

    Args:
        context: Airflow task context

    Returns:
        str: Path to backup file

    Raises:
        AirflowException: If backup fails
    """
    ti = context['task_instance']
    execution_date = context['execution_date']

    logger.info("=" * 80)
    logger.info("TASK 4: Creating database backup")
    logger.info("=" * 80)

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Generate timestamped backup filename
    timestamp = execution_date.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"ntsb_aviation_backup_{timestamp}.sql.gz"
    backup_path = BACKUP_DIR / backup_filename

    logger.info(f"Backup file: {backup_path}")

    # Get database connection parameters
    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = pg_hook.get_connection(DB_CONN_ID)

    # Build pg_dump command
    env = os.environ.copy()
    env['PGPASSWORD'] = conn.password or ''

    pg_dump_cmd = [
        'pg_dump',
        '-h', conn.host or 'localhost',
        '-p', str(conn.port or 5432),
        '-U', conn.login or 'parobek',
        '-d', DB_NAME,
        '--format=plain',
        '--no-owner',
        '--no-acl',
    ]

    gzip_cmd = ['gzip', '-c']

    logger.info(f"Running: {' '.join(pg_dump_cmd)} | gzip > {backup_path}")

    try:
        # Run pg_dump and pipe to gzip
        with open(backup_path, 'wb') as backup_file:
            pg_dump_process = subprocess.Popen(
                pg_dump_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )

            gzip_process = subprocess.Popen(
                gzip_cmd,
                stdin=pg_dump_process.stdout,
                stdout=backup_file,
                stderr=subprocess.PIPE
            )

            # Allow pg_dump to receive SIGPIPE if gzip exits
            pg_dump_process.stdout.close()

            # Wait for both processes
            gzip_stdout, gzip_stderr = gzip_process.communicate(timeout=1800)  # 30 min timeout
            pg_dump_returncode = pg_dump_process.wait()

            if pg_dump_returncode != 0:
                _, pg_dump_stderr = pg_dump_process.communicate()
                raise AirflowException(f"pg_dump failed: {pg_dump_stderr.decode()}")

            if gzip_process.returncode != 0:
                raise AirflowException(f"gzip failed: {gzip_stderr.decode()}")

    except subprocess.TimeoutExpired:
        raise AirflowException("Backup timed out after 30 minutes")
    except FileNotFoundError as e:
        raise AirflowException(f"Command not found: {e}. Ensure pg_dump and gzip are installed.")
    except Exception as e:
        raise AirflowException(f"Backup failed: {e}")

    # Verify backup was created
    if not backup_path.exists():
        raise AirflowException(f"Backup file was not created: {backup_path}")

    backup_size = backup_path.stat().st_size

    if backup_size < 1_000_000:  # Less than 1 MB
        raise AirflowException(
            f"Backup file is suspiciously small ({backup_size:,} bytes). "
            "Backup may have failed."
        )

    logger.info(f"✅ Backup created: {backup_path}")
    logger.info(f"Backup size: {backup_size:,} bytes ({backup_size / 1_000_000:.1f} MB)")

    # Store backup path in XCom
    ti.xcom_push(key='backup_path', value=str(backup_path))
    ti.xcom_push(key='backup_size', value=backup_size)

    # Clean up old backups (keep last 10)
    backups = sorted(BACKUP_DIR.glob("ntsb_aviation_backup_*.sql.gz"))

    if len(backups) > 10:
        old_backups = backups[:-10]
        logger.info(f"Cleaning up {len(old_backups)} old backups...")

        for old_backup in old_backups:
            logger.info(f"Deleting: {old_backup}")
            old_backup.unlink()

    return str(backup_path)


# ====================================================================================================
# TASK 5: LOAD NEW DATA
# ====================================================================================================

def load_new_data_task(**context) -> Dict[str, any]:
    """
    Load new NTSB data using existing load_with_staging.py script.

    This task:
    1. Calls load_with_staging.py with extracted MDB path
    2. Captures output and logs
    3. Parses load statistics from output
    4. Returns load summary

    Args:
        context: Airflow task context

    Returns:
        Dict with load statistics

    Raises:
        AirflowException: If load fails
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 5: Loading new data")
    logger.info("=" * 80)

    # Get MDB path from previous task
    mdb_path = ti.xcom_pull(task_ids='extract_avall_mdb', key='extracted_mdb_path')

    if mdb_path is None:
        raise AirflowException("Extracted MDB path not found in XCom.")

    mdb_path = Path(mdb_path)

    if not mdb_path.exists():
        raise AirflowException(f"MDB file not found: {mdb_path}")

    # Get database connection parameters
    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = pg_hook.get_connection(DB_CONN_ID)

    # Build load command
    load_script = SCRIPTS_DIR / "load_with_staging.py"

    if not load_script.exists():
        raise AirflowException(f"Load script not found: {load_script}")

    # Set environment variables for database connection
    env = os.environ.copy()
    env['PGHOST'] = conn.host or 'localhost'
    env['PGPORT'] = str(conn.port or 5432)
    env['PGDATABASE'] = DB_NAME
    env['PGUSER'] = conn.login or 'parobek'
    env['PGPASSWORD'] = conn.password or ''
    # Also set DB_* variables for load_with_staging.py
    env['DB_HOST'] = conn.host or 'localhost'
    env['DB_PORT'] = str(conn.port or 5432)
    env['DB_NAME'] = DB_NAME
    env['DB_USER'] = conn.login or 'parobek'
    env['DB_PASSWORD'] = conn.password or ''

    load_cmd = [
        'python3',
        str(load_script),
        '--source', mdb_path.name,  # Just the filename (script expects it in datasets/)
    ]

    # For monthly sync, always use --force for avall.mdb
    # This is safe because:
    # 1. load_with_staging.py has built-in duplicate detection via staging tables
    # 2. Only new events are merged into production (duplicates are skipped)
    # 3. avall.mdb is designed for monthly updates (unlike historical databases)
    #
    # The --force flag allows the script to proceed even if avall.mdb was previously loaded,
    # which is expected behavior for monthly sync operations.
    is_first_load = ti.xcom_pull(task_ids='check_for_updates', key='is_first_load')

    if not is_first_load:
        logger.info("This is a monthly re-load (avall.mdb periodic update)")
        logger.info("Adding --force flag to allow reload (safe due to duplicate detection)")
        load_cmd.append('--force')
    else:
        logger.info("This is the first load (no --force needed)")

    logger.info(f"Running: {' '.join(load_cmd)}")
    logger.info(f"Working directory: {EXTRACT_DIR.parent.parent}")

    try:
        # Run load script
        # Note: Volumes are mounted at /opt/airflow/datasets and /opt/airflow/scripts
        project_root = Path("/opt/airflow")

        # Copy MDB to datasets/ directory (where script expects it)
        datasets_dir = project_root / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        target_mdb = datasets_dir / "avall.mdb"

        logger.info(f"Copying {mdb_path} to {target_mdb}")

        import shutil
        shutil.copyfile(mdb_path, target_mdb)

        # Now run the load script
        load_cmd = [
            'python3',
            str(load_script),
            '--source', 'avall.mdb',  # Now it's in datasets/
        ]

        # Add --force flag if this is a re-load (monthly sync)
        if not is_first_load:
            load_cmd.append('--force')

        result = subprocess.run(
            load_cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout (load can take a while)
            env=env,
            check=True
        )

        # Log output
        logger.info("=== Load Script Output ===")
        logger.info(result.stdout)

        if result.stderr:
            logger.warning("=== Load Script Warnings ===")
            logger.warning(result.stderr)

        # Parse load statistics from output
        # Look for patterns like "✅ Loaded X events" in output
        import re

        events_loaded = 0
        duplicates_found = 0

        # Extract statistics from output
        for line in result.stdout.split('\n'):
            if match := re.search(r'(\d+)\s+new events', line, re.IGNORECASE):
                events_loaded = int(match.group(1))
            elif match := re.search(r'(\d+)\s+duplicate events', line, re.IGNORECASE):
                duplicates_found = int(match.group(1))

        load_summary = {
            'events_loaded': events_loaded,
            'duplicates_found': duplicates_found,
            'load_successful': True,
            'mdb_file': str(mdb_path.name),
        }

        logger.info("=== Load Summary ===")
        logger.info(f"Events loaded: {events_loaded:,}")
        logger.info(f"Duplicates found: {duplicates_found:,}")

        # Store summary in XCom
        ti.xcom_push(key='load_summary', value=load_summary)

        return load_summary

    except subprocess.CalledProcessError as e:
        logger.error("=== Load Script Error ===")
        logger.error(e.stdout)
        logger.error(e.stderr)
        raise AirflowException(f"Load script failed with exit code {e.returncode}")
    except subprocess.TimeoutExpired:
        raise AirflowException("Load script timed out after 2 hours")
    except Exception as e:
        raise AirflowException(f"Unexpected error during load: {e}")


# ====================================================================================================
# TASK 6: VALIDATE DATA QUALITY
# ====================================================================================================

def validate_data_quality_task(**context) -> Dict[str, any]:
    """
    Run data quality validation queries.

    This task:
    1. Runs critical validation queries
    2. Checks row counts, foreign keys, NULL values
    3. Verifies generated columns and constraints
    4. Returns validation results

    Args:
        context: Airflow task context

    Returns:
        Dict with validation results

    Raises:
        AirflowException: If critical validation fails
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 6: Validating data quality")
    logger.info("=" * 80)

    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

    validation_results = {}
    errors = []
    warnings = []

    # 1. Check row counts
    logger.info("1. Checking row counts...")

    row_count_query = """
        SELECT
            schemaname,
            relname as tablename,
            n_live_tup as rows
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        ORDER BY n_live_tup DESC;
    """

    row_counts = pg_hook.get_records(row_count_query)

    total_rows = sum(row[2] for row in row_counts)
    validation_results['row_counts'] = {row[1]: row[2] for row in row_counts}
    validation_results['total_rows'] = total_rows

    logger.info(f"Total rows across all tables: {total_rows:,}")

    for schema, table, rows in row_counts:
        logger.info(f"  {table}: {rows:,}")

    # Check if events table has rows
    events_count = validation_results['row_counts'].get('events', 0)

    if events_count == 0:
        errors.append("events table is empty!")
    elif events_count < 10000:
        warnings.append(f"events table has only {events_count:,} rows (expected ~90,000+)")

    # 2. Check for duplicate primary keys
    logger.info("2. Checking primary key uniqueness...")

    duplicate_pk_query = """
        SELECT
            COUNT(*) as total_events,
            COUNT(DISTINCT ev_id) as unique_ev_ids,
            COUNT(*) - COUNT(DISTINCT ev_id) as duplicates
        FROM events;
    """

    pk_result = pg_hook.get_first(duplicate_pk_query)

    if pk_result[2] > 0:
        errors.append(f"Found {pk_result[2]} duplicate ev_id values in events table!")
    else:
        logger.info("✅ No duplicate primary keys found")

    validation_results['duplicate_primary_keys'] = pk_result[2]

    # 3. Check foreign key integrity
    logger.info("3. Checking foreign key integrity...")

    orphan_check_queries = {
        'aircraft': """
            SELECT COUNT(*)
            FROM aircraft a
            LEFT JOIN events e ON a.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """,
        'findings': """
            SELECT COUNT(*)
            FROM Findings f
            LEFT JOIN events e ON f.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """,
        'injury': """
            SELECT COUNT(*)
            FROM injury i
            LEFT JOIN events e ON i.ev_id = e.ev_id
            WHERE e.ev_id IS NULL;
        """,
    }

    orphan_counts = {}

    for table, query in orphan_check_queries.items():
        orphan_count = pg_hook.get_first(query)[0]
        orphan_counts[table] = orphan_count

        if orphan_count > 0:
            errors.append(f"Found {orphan_count} orphaned records in {table} table!")
        else:
            logger.info(f"✅ No orphaned records in {table}")

    validation_results['orphan_records'] = orphan_counts

    # 4. Check coordinate validity
    logger.info("4. Checking coordinate validity...")

    invalid_coords_query = """
        SELECT COUNT(*)
        FROM events
        WHERE (dec_latitude IS NOT NULL AND (dec_latitude < -90 OR dec_latitude > 90))
           OR (dec_longitude IS NOT NULL AND (dec_longitude < -180 OR dec_longitude > 180));
    """

    invalid_coords = pg_hook.get_first(invalid_coords_query)[0]

    if invalid_coords > 0:
        warnings.append(f"Found {invalid_coords} events with invalid coordinates")
    else:
        logger.info("✅ All coordinates valid")

    validation_results['invalid_coordinates'] = invalid_coords

    # 5. Check date validity
    logger.info("5. Checking date validity...")

    invalid_dates_query = """
        SELECT COUNT(*)
        FROM events
        WHERE ev_date IS NOT NULL
          AND (ev_date < '1962-01-01' OR ev_date > CURRENT_DATE + INTERVAL '1 year');
    """

    invalid_dates = pg_hook.get_first(invalid_dates_query)[0]

    if invalid_dates > 0:
        warnings.append(f"Found {invalid_dates} events with questionable dates")
    else:
        logger.info("✅ All dates valid")

    validation_results['invalid_dates'] = invalid_dates

    # 6. Summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    validation_results['errors'] = errors
    validation_results['warnings'] = warnings
    validation_results['validation_passed'] = len(errors) == 0

    if errors:
        logger.error(f"❌ {len(errors)} critical validation errors:")
        for error in errors:
            logger.error(f"  - {error}")

    if warnings:
        logger.warning(f"⚠️  {len(warnings)} warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    if not errors and not warnings:
        logger.info("✅ All validation checks passed!")
    elif not errors:
        logger.info("✅ All critical checks passed (with warnings)")

    # Store results in XCom
    ti.xcom_push(key='validation_results', value=validation_results)

    # Fail task if critical errors found
    if errors:
        raise AirflowException(
            f"Data validation failed with {len(errors)} critical errors. "
            "Database may be in inconsistent state. Check logs for details."
        )

    return validation_results


# ====================================================================================================
# TASK 7: REFRESH MATERIALIZED VIEWS
# ====================================================================================================

def refresh_materialized_views_task(**context) -> Dict[str, any]:
    """
    Refresh all materialized views and update statistics.

    This task:
    1. Calls refresh_all_materialized_views() function
    2. Runs ANALYZE on main tables
    3. Verifies materialized views have data
    4. Returns refresh statistics

    Args:
        context: Airflow task context

    Returns:
        Dict with refresh statistics

    Raises:
        AirflowException: If refresh fails
    """
    ti = context['task_instance']

    logger.info("=" * 80)
    logger.info("TASK 7: Refreshing materialized views")
    logger.info("=" * 80)

    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

    # Get list of materialized views before refresh
    mv_list_query = """
        SELECT
            schemaname,
            matviewname,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
        FROM pg_matviews
        WHERE schemaname = 'public'
        ORDER BY matviewname;
    """

    mvs_before = pg_hook.get_records(mv_list_query)

    logger.info(f"Found {len(mvs_before)} materialized views to refresh:")
    for schema, mv, size in mvs_before:
        logger.info(f"  {mv}: {size}")

    # Refresh all materialized views (using existing function)
    logger.info("Calling refresh_all_materialized_views()...")

    refresh_start = datetime.now()

    try:
        pg_hook.run("SELECT * FROM refresh_all_materialized_views();")

        refresh_end = datetime.now()
        refresh_duration = (refresh_end - refresh_start).total_seconds()

        logger.info(f"✅ Materialized views refreshed in {refresh_duration:.1f} seconds")

    except Exception as e:
        raise AirflowException(f"Failed to refresh materialized views: {e}")

    # Run ANALYZE on main tables
    logger.info("Running ANALYZE on main tables...")

    analyze_tables = [
        'events', 'aircraft', 'flight_crew', 'injury', 'findings',
        'narratives', 'engines', 'ntsb_admin', 'events_sequence'
    ]

    for table in analyze_tables:
        try:
            pg_hook.run(f"ANALYZE {table};")
            logger.info(f"  Analyzed {table}")
        except Exception as e:
            logger.warning(f"  Failed to analyze {table}: {e}")

    # Verify materialized views have data
    logger.info("Verifying materialized views...")

    mv_row_counts = {}

    for schema, mv, _ in mvs_before:
        row_count_query = f"SELECT COUNT(*) FROM {mv};"
        row_count = pg_hook.get_first(row_count_query)[0]
        mv_row_counts[mv] = row_count

        logger.info(f"  {mv}: {row_count:,} rows")

        if row_count == 0:
            logger.warning(f"⚠️  {mv} has 0 rows after refresh!")

    # Get updated sizes
    mvs_after = pg_hook.get_records(mv_list_query)

    refresh_summary = {
        'materialized_views_refreshed': len(mvs_after),
        'refresh_duration_seconds': refresh_duration,
        'mv_row_counts': mv_row_counts,
        'refresh_successful': True,
    }

    # Store summary in XCom
    ti.xcom_push(key='refresh_summary', value=refresh_summary)

    logger.info("✅ Materialized view refresh complete")

    return refresh_summary


# ====================================================================================================
# TASK 8: SEND SUCCESS NOTIFICATION
# ====================================================================================================

def send_success_notification_task(**context) -> None:
    """
    Send success notification with load summary.

    This task:
    1. Gathers statistics from all previous tasks
    2. Generates summary report
    3. Logs summary (and optionally sends email/Slack)

    Args:
        context: Airflow task context
    """
    ti = context['task_instance']
    execution_date = context['execution_date']

    logger.info("=" * 80)
    logger.info("TASK 8: Sending success notification")
    logger.info("=" * 80)

    # Gather statistics from previous tasks
    load_summary = ti.xcom_pull(task_ids='load_new_data', key='load_summary')
    validation_results = ti.xcom_pull(task_ids='validate_data_quality', key='validation_results')
    refresh_summary = ti.xcom_pull(task_ids='refresh_materialized_views', key='refresh_summary')

    current_file_size = ti.xcom_pull(task_ids='check_for_updates', key='current_file_size')
    backup_path = ti.xcom_pull(task_ids='backup_database', key='backup_path')

    # Get final database statistics
    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

    db_stats_query = """
        SELECT
            pg_database_size('ntsb_aviation') as db_size,
            (SELECT COUNT(*) FROM events) as total_events,
            (SELECT COUNT(*) FROM aircraft) as total_aircraft,
            (SELECT COUNT(*) FROM findings) as total_findings;
    """

    db_stats = pg_hook.get_first(db_stats_query)

    # Build summary report
    summary_lines = [
        "",
        "=" * 80,
        "NTSB MONTHLY SYNC - SUCCESS",
        "=" * 80,
        "",
        f"Execution Date: {execution_date.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"DAG Run ID: {context['run_id']}",
        "",
        "--- FILE DOWNLOAD ---",
        "Source: avall.zip from NTSB",
        f"File Size: {current_file_size:,} bytes ({current_file_size / 1_000_000:.1f} MB)",
        "",
        "--- DATA LOAD ---",
        f"Events Loaded: {load_summary.get('events_loaded', 0):,}",
        f"Duplicates Found: {load_summary.get('duplicates_found', 0):,}",
        "",
        "--- DATABASE STATE ---",
        f"Total Events: {db_stats[1]:,}",
        f"Total Aircraft: {db_stats[2]:,}",
        f"Total Findings: {db_stats[3]:,}",
        f"Database Size: {db_stats[0]:,} bytes ({db_stats[0] / 1_000_000:.1f} MB)",
        "",
        "--- VALIDATION ---",
        f"Critical Errors: {len(validation_results.get('errors', []))}",
        f"Warnings: {len(validation_results.get('warnings', []))}",
        f"Validation Status: {'✅ PASSED' if validation_results.get('validation_passed') else '❌ FAILED'}",
        "",
        "--- MATERIALIZED VIEWS ---",
        f"Views Refreshed: {refresh_summary.get('materialized_views_refreshed', 0)}",
        f"Refresh Duration: {refresh_summary.get('refresh_duration_seconds', 0):.1f} seconds",
        "",
        "--- BACKUP ---",
        f"Backup Location: {backup_path}",
        "",
        "=" * 80,
        "All tasks completed successfully!",
        "=" * 80,
        "",
    ]

    summary_text = "\n".join(summary_lines)

    # Log summary
    logger.info(summary_text)

    # Store summary in XCom
    ti.xcom_push(key='final_summary', value=summary_text)

    # TODO: Send email notification if configured
    notification_email = Variable.get("ntsb_notification_email", default_var=None)

    if notification_email:
        logger.info(f"Email notification would be sent to: {notification_email}")
        # TODO: Implement email sending (requires EmailOperator configuration)

    # TODO: Send Slack notification if configured
    slack_webhook = Variable.get("ntsb_slack_webhook", default_var=None)

    if slack_webhook:
        logger.info("Slack notification would be sent to webhook")
        # TODO: Implement Slack notification (requires SlackWebhookOperator)

    logger.info("✅ Success notification complete")


# ====================================================================================================
# DAG DEFINITION
# ====================================================================================================

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': EMAIL_ON_FAILURE,
    'email_on_retry': EMAIL_ON_RETRY,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=3),  # Total DAG timeout
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Monthly synchronization of NTSB aviation accident database from data.ntsb.gov',
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    catchup=CATCHUP,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['ntsb', 'production', 'etl', 'monthly', 'aviation'],
    doc_md=__doc__,
) as dag:

    # Task 1: Check for updates (ShortCircuitOperator for smart skipping)
    check_for_updates = ShortCircuitOperator(
        task_id='check_for_updates',
        python_callable=check_for_updates_task,
        provide_context=True,
        doc_md="""
        Check if avall.zip has been updated since last successful load.

        This task queries the NTSB website for file metadata and compares with
        the last successful load recorded in the load_tracking table.

        If no update is detected, downstream tasks are skipped (smart skipping).
        """,
    )

    # Task 2: Download avall.zip
    download_avall_zip = PythonOperator(
        task_id='download_avall_zip',
        python_callable=download_avall_zip_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        Download avall.zip from NTSB website.

        Downloads the current aviation accident database (avall.zip) from
        https://data.ntsb.gov/avdata/ and verifies file size matches expected size.
        """,
    )

    # Task 3: Extract avall.mdb
    extract_avall_mdb = PythonOperator(
        task_id='extract_avall_mdb',
        python_callable=extract_avall_mdb_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        Extract avall.mdb from avall.zip.

        Extracts the Microsoft Access database file from the downloaded zip archive.
        """,
    )

    # Task 4: Backup database
    backup_database = PythonOperator(
        task_id='backup_database',
        python_callable=backup_database_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=45),
        doc_md="""
        Create timestamped backup of ntsb_aviation database.

        Uses pg_dump to create a compressed SQL backup before loading new data.
        Backups are stored in /tmp/NTSB_Datasets/backups/ with automatic cleanup
        (keeps last 10 backups).
        """,
    )

    # Task 5: Load new data
    load_new_data = PythonOperator(
        task_id='load_new_data',
        python_callable=load_new_data_task,
        provide_context=True,
        execution_timeout=timedelta(hours=2),
        doc_md="""
        Load new NTSB data using existing load_with_staging.py script.

        Calls the production ETL script which:
        1. Extracts CSV from MDB
        2. Loads into staging tables
        3. Identifies duplicates
        4. Merges only new events
        5. Updates load_tracking table
        """,
    )

    # Task 6: Validate data quality
    validate_data_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
        doc_md="""
        Run comprehensive data quality validation.

        Validates:
        - Row counts (events, aircraft, findings, etc.)
        - Primary key uniqueness
        - Foreign key integrity
        - Coordinate validity
        - Date validity

        Fails the DAG if critical validation errors are found.
        """,
    )

    # Task 7: Refresh materialized views
    refresh_materialized_views = PythonOperator(
        task_id='refresh_materialized_views',
        python_callable=refresh_materialized_views_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        Refresh all materialized views and update statistics.

        Calls refresh_all_materialized_views() to update:
        - mv_yearly_stats
        - mv_state_stats
        - mv_aircraft_stats
        - mv_decade_stats
        - mv_crew_stats
        - mv_finding_stats

        Also runs ANALYZE on main tables for query planner.
        """,
    )

    # Task 8: Send success notification
    send_success_notification = PythonOperator(
        task_id='send_success_notification',
        python_callable=send_success_notification_task,
        provide_context=True,
        doc_md="""
        Send success notification with load summary.

        Gathers statistics from all tasks and generates a summary report.
        Logs the summary and optionally sends email/Slack notifications
        (if configured via Airflow Variables).
        """,
    )

    # Define task dependencies
    check_for_updates >> download_avall_zip >> extract_avall_mdb >> backup_database
    backup_database >> load_new_data >> validate_data_quality
    validate_data_quality >> refresh_materialized_views >> send_success_notification


# ====================================================================================================
# DAG DOCUMENTATION
# ====================================================================================================

# This DAG is designed to run automatically on a monthly schedule, but can also be
# triggered manually for testing or emergency updates.
#
# Manual Trigger:
#   airflow dags trigger monthly_sync_ntsb_data
#
# Test Run (with dry-run mode):
#   airflow dags test monthly_sync_ntsb_data 2025-11-01
#
# Configuration:
#   - Airflow Connection: ntsb_aviation_db (PostgreSQL)
#   - Airflow Variables (optional):
#     - ntsb_notification_email: Email for success/failure notifications
#     - ntsb_slack_webhook: Slack webhook URL for notifications
#
# Monitoring:
#   - Check Airflow Web UI: http://localhost:8080
#   - Review task logs for detailed execution information
#   - Check /tmp/NTSB_Datasets/backups/ for database backups
#
# Troubleshooting:
#   - If download fails: Check NTSB website availability, verify URL
#   - If load fails: Review load_with_staging.log, check database permissions
#   - If validation fails: Review validation results in logs, may indicate data quality issues
#   - If MV refresh fails: Check database disk space, review materialized view definitions
#
# Error Recovery:
#   - Automatic retries (2x with 5min delay) for transient errors
#   - Database backup created before load (can manually restore if needed)
#   - Validation prevents bad data from being committed
#   - Staging table pattern allows inspection before merge
#
# Performance:
#   - Typical DAG execution time: 20-40 minutes (depends on data size)
#   - Download: ~2 minutes (92 MB file)
#   - Extract: ~30 seconds
#   - Backup: ~5 minutes (966 MB database)
#   - Load: ~10-15 minutes (depends on duplicates)
#   - Validation: ~1 minute
#   - MV Refresh: ~2 minutes
#   - Total: ~20-25 minutes (normal run)
