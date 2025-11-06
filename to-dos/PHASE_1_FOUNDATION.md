# PHASE 1: FOUNDATION

Database migration, ETL pipeline, data quality, and API infrastructure.

**Timeline**: Q1 2025 (12 weeks, January-March 2025)
**Prerequisites**: Python 3.11+, PostgreSQL 15+, Docker, mdbtools
**Team**: 2 developers (backend + data engineer)
**Estimated Hours**: ~370 hours total
**Budget**: $500-1000 (infrastructure setup, development tools)

## Overview

| Sprint | Duration | Focus Area | Key Deliverables | Hours |
|--------|----------|------------|------------------|-------|
| Sprint 1 | Weeks 1-3 | PostgreSQL Migration | Schema design, MDB extraction, 100K+ records | 90h |
| Sprint 2 | Weeks 4-6 | ETL Pipeline (Airflow) | 5 DAGs, automated monthly sync, CDC | 100h |
| Sprint 3 | Weeks 7-9 | Data Quality Framework | Great Expectations, cleaning, monitoring | 90h |
| Sprint 4 | Weeks 10-12 | FastAPI Development | 5 endpoints, JWT auth, rate limiting | 90h |

## Sprint 1: PostgreSQL Migration (Weeks 1-3, November 2025)

**Goal**: Migrate 100K+ aviation accident records from Microsoft Access to PostgreSQL with optimized schema and <100ms query performance.

### Week 1: PostgreSQL Schema Design

**Tasks**:
- [ ] Install PostgreSQL 15+ with PostGIS extension for geospatial data
- [ ] Design normalized relational schema matching MDB structure (10 core tables)
- [ ] Implement table partitioning by year (1960s-2020s) for events table
- [ ] Create composite indexes on ev_id, ev_date, ev_state, ev_highest_injury
- [ ] Add PostGIS spatial indexes (GIST) for latitude/longitude queries
- [ ] Set up pgBouncer connection pooling (max 100 connections)
- [ ] Configure automated backups with pg_dump (daily, 30-day retention)
- [ ] Create database users with role-based permissions (admin, app, readonly)

**Deliverables**:
- Complete PostgreSQL schema with 10 tables, 25+ indexes, partitioning
- Connection pooling configured (pgBouncer)
- Automated backup system operational

**Success Metrics**:
- All tables created without errors
- Foreign key constraints validated
- Backup script runs successfully
- Connection pool handles 100+ concurrent connections

**Research Finding**: PostgreSQL table partitioning (2024) provides 10-20x query performance improvement for time-series data. Partitioning by year enables partition pruning, where PostgreSQL only scans relevant partitions instead of entire table. Best practice: use range partitioning for time-series data, list partitioning for categorical data.

**Code Example - Complete Schema**:
```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Fuzzy text search

-- ============================================
-- CORE TABLES
-- ============================================

-- Master events table (partitioned by year)
CREATE TABLE events (
    ev_id VARCHAR(20) PRIMARY KEY,
    ev_date DATE NOT NULL,
    ev_time TIME,
    ev_year INTEGER GENERATED ALWAYS AS (EXTRACT(YEAR FROM ev_date)) STORED,
    ev_month INTEGER GENERATED ALWAYS AS (EXTRACT(MONTH FROM ev_date)) STORED,

    -- Location
    ev_city VARCHAR(100),
    ev_state CHAR(2),
    dec_latitude DECIMAL(10, 6),
    dec_longitude DECIMAL(11, 6),
    location_geom GEOGRAPHY(POINT, 4326) GENERATED ALWAYS AS (
        CASE
            WHEN dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
            THEN ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)::geography
            ELSE NULL
        END
    ) STORED,

    -- Classification
    ev_highest_injury VARCHAR(10),
    inj_tot_f INTEGER DEFAULT 0,  -- Fatalities
    inj_tot_s INTEGER DEFAULT 0,  -- Serious

    -- Weather
    wx_cond_basic VARCHAR(10),
    wx_wind_speed INTEGER,

    -- Investigation
    probable_cause TEXT,

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash CHAR(64),  -- SHA-256 for CDC

    CONSTRAINT valid_latitude CHECK (dec_latitude BETWEEN -90 AND 90),
    CONSTRAINT valid_longitude CHECK (dec_longitude BETWEEN -180 AND 180)
) PARTITION BY RANGE (ev_year);

-- Create partitions for each decade
CREATE TABLE events_1960s PARTITION OF events FOR VALUES FROM (1960) TO (1970);
CREATE TABLE events_1970s PARTITION OF events FOR VALUES FROM (1970) TO (1980);
CREATE TABLE events_1980s PARTITION OF events FOR VALUES FROM (1980) TO (1990);
CREATE TABLE events_1990s PARTITION OF events FOR VALUES FROM (1990) TO (2000);
CREATE TABLE events_2000s PARTITION OF events FOR VALUES FROM (2000) TO (2010);
CREATE TABLE events_2010s PARTITION OF events FOR VALUES FROM (2010) TO (2020);
CREATE TABLE events_2020s PARTITION OF events FOR VALUES FROM (2020) TO (2030);

-- Aircraft table
CREATE TABLE aircraft (
    Aircraft_Key VARCHAR(20) PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    acft_make VARCHAR(100),
    acft_model VARCHAR(100),
    acft_category VARCHAR(30),
    num_eng INTEGER,
    damage VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Flight crew table
CREATE TABLE Flight_Crew (
    crew_no SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key),
    pilot_tot_time INTEGER,
    crew_age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Findings table (investigation results)
CREATE TABLE Findings (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    finding_code VARCHAR(10),
    cm_inPC BOOLEAN DEFAULT FALSE,  -- In probable cause (Release 3.0+)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Narratives table with full-text search
CREATE TABLE narratives (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    narr_accp TEXT,
    narr_cause TEXT,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(narr_accp, '') || ' ' || COALESCE(narr_cause, ''))
    ) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

-- Events indexes (critical for query performance)
CREATE INDEX idx_events_ev_date ON events(ev_date);
CREATE INDEX idx_events_ev_year ON events(ev_year);
CREATE INDEX idx_events_severity ON events(ev_highest_injury);
CREATE INDEX idx_events_state ON events(ev_state);
CREATE INDEX idx_events_location_geom ON events USING GIST(location_geom);
CREATE INDEX idx_events_date_severity ON events(ev_date, ev_highest_injury);

-- Aircraft indexes
CREATE INDEX idx_aircraft_ev_id ON aircraft(ev_id);
CREATE INDEX idx_aircraft_make_model ON aircraft(acft_make, acft_model);

-- Narratives full-text search index
CREATE INDEX idx_narratives_search ON narratives USING GIN(search_vector);

-- ============================================
-- MATERIALIZED VIEWS (for dashboard performance)
-- ============================================

-- Yearly statistics (refreshed daily)
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT
    ev_year,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities
FROM events e
GROUP BY ev_year
ORDER BY ev_year;

CREATE UNIQUE INDEX idx_mv_yearly_stats_year ON mv_yearly_stats(ev_year);

-- ============================================
-- TRIGGERS
-- ============================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Content hash for Change Data Capture
CREATE OR REPLACE FUNCTION calculate_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash = encode(
        digest(
            CONCAT_WS('|', NEW.ev_id, NEW.ev_date, NEW.probable_cause)::TEXT,
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_events_content_hash BEFORE INSERT OR UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION calculate_content_hash();
```

**Dependencies**: postgresql-15, postgis, mdbtools

**Sprint 1.1 Total Hours**: 30 hours

---

### Week 2: MDB to PostgreSQL Extraction & Transformation

**Tasks**:
- [ ] Extract all 10 tables from 3 MDB databases using mdb-export (avall.mdb, Pre2008.mdb, PRE1982.MDB)
- [ ] Implement data transformation pipeline: clean NULLs, normalize dates, validate coordinates
- [ ] Convert DMS coordinates to decimal degrees where needed
- [ ] Validate data quality: check for invalid dates, out-of-range coordinates, missing primary keys
- [ ] Load data into PostgreSQL using COPY command (10x faster than INSERT)
- [ ] Implement Change Data Capture (CDC) using content hashing (SHA-256)
- [ ] Create data lineage tracking: source file → CSV → PostgreSQL with timestamps

**Deliverables**:
- Python migration script with comprehensive error handling
- 100K+ records loaded into PostgreSQL across 10 tables
- Data validation report (CSV with quality metrics)
- CDC system operational for detecting monthly updates

**Success Metrics**:
- 100% of valid records migrated (expected: 95-98% due to data quality issues)
- Migration completes in <15 minutes for all 3 databases
- Zero primary key violations
- <5% NULL rate for critical fields (ev_id, ev_date, ev_state)

**Research Finding**: PostgreSQL COPY command is 10-20x faster than INSERT statements for bulk loading. Use COPY with tab-delimited format and explicit NULL handling for best performance. For 100K records, COPY completes in ~30 seconds vs 5-10 minutes with INSERT.

**Code Example - Complete Migration Script**:
```python
# scripts/migrate_mdb_to_postgres.py
"""
Extract data from NTSB MDB databases and load into PostgreSQL.

Usage:
    python migrate_mdb_to_postgres.py --database datasets/avall.mdb --truncate
"""

import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import sys
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    'dbname': 'ntsb',
    'user': 'app',
    'password': 'dev_password',
    'host': 'localhost',
    'port': 5432
}

# Tables to extract (in dependency order)
TABLES = ['events', 'aircraft', 'Flight_Crew', 'Findings', 'narratives']

class MDBExtractor:
    """Extract tables from Microsoft Access MDB files."""

    def __init__(self, mdb_path: str):
        self.mdb_path = Path(mdb_path)
        if not self.mdb_path.exists():
            raise FileNotFoundError(f"MDB file not found: {mdb_path}")
        logger.info(f"Initializing extractor for {self.mdb_path}")

    def extract_table(self, table_name: str, output_dir: Path) -> Path:
        """Extract single table to CSV."""
        output_file = output_dir / f"{table_name}.csv"

        try:
            logger.info(f"Extracting table '{table_name}' to {output_file}")

            with open(output_file, 'w') as f:
                subprocess.run(
                    ['mdb-export', '-D', '%Y-%m-%d', str(self.mdb_path), table_name],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True
                )

            row_count = sum(1 for _ in open(output_file)) - 1
            logger.info(f"Extracted {row_count:,} rows from '{table_name}'")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract table '{table_name}': {e.stderr.decode()}")
            raise

class DataTransformer:
    """Clean and transform extracted CSV data."""

    @staticmethod
    def clean_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Convert various date formats to ISO 8601."""
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    @staticmethod
    def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinates."""
        if 'dec_latitude' in df.columns:
            df['dec_latitude'] = pd.to_numeric(df['dec_latitude'], errors='coerce')
            df.loc[(df['dec_latitude'] < -90) | (df['dec_latitude'] > 90), 'dec_latitude'] = None

        if 'dec_longitude' in df.columns:
            df['dec_longitude'] = pd.to_numeric(df['dec_longitude'], errors='coerce')
            df.loc[(df['dec_longitude'] < -180) | (df['dec_longitude'] > 180), 'dec_longitude'] = None

        return df

    @staticmethod
    def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Handle various NULL representations."""
        df.replace({'': None, 'UNK': None, 'UNKN': None}, inplace=True)
        return df

    def transform_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform events table."""
        logger.info(f"Transforming events table ({len(df)} rows)")

        df = self.clean_dates(df, ['ev_date', 'ev_time'])
        df = self.clean_coordinates(df)
        df = self.clean_nulls(df)

        # Validate ev_date
        current_year = datetime.now().year
        df = df[df['ev_date'].notna()]
        df = df[(df['ev_date'].dt.year >= 1962) & (df['ev_date'].dt.year <= current_year + 1)]

        logger.info(f"Transformation complete: {len(df)} valid rows")
        return df

class PostgreSQLLoader:
    """Load transformed data into PostgreSQL."""

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def bulk_load(self, table_name: str, df: pd.DataFrame):
        """Bulk load DataFrame using COPY (10x faster than INSERT)."""
        try:
            df = df.where(pd.notnull(df), None)
            columns = ', '.join(df.columns)

            from io import StringIO
            output = StringIO()
            df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
            output.seek(0)

            self.cursor.copy_expert(
                f"COPY {table_name} ({columns}) FROM STDIN WITH CSV DELIMITER E'\\t' NULL '\\N'",
                output
            )

            self.conn.commit()
            logger.info(f"Loaded {len(df):,} rows into '{table_name}'")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to load data into '{table_name}': {e}")
            raise

    def vacuum_analyze(self):
        """Run VACUUM ANALYZE for query optimization."""
        logger.info("Running VACUUM ANALYZE...")
        old_isolation_level = self.conn.isolation_level
        self.conn.set_isolation_level(0)
        self.cursor.execute("VACUUM ANALYZE")
        self.conn.set_isolation_level(old_isolation_level)
        logger.info("VACUUM ANALYZE complete")

def main():
    parser = argparse.ArgumentParser(description='Migrate NTSB MDB to PostgreSQL')
    parser.add_argument('--database', required=True, help='Path to MDB file')
    parser.add_argument('--output-dir', default='./tmp/csv_export', help='CSV output directory')
    parser.add_argument('--truncate', action='store_true', help='Truncate tables before load')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("NTSB MDB to PostgreSQL Migration")
    logger.info(f"Database: {args.database}")
    logger.info("=" * 80)

    try:
        # Step 1: Extract from MDB
        extractor = MDBExtractor(args.database)
        extracted_files = {}
        for table in TABLES:
            extracted_files[table] = extractor.extract_table(table, output_dir)

        # Step 2: Transform and load
        loader = PostgreSQLLoader(DB_CONFIG)
        loader.connect()
        transformer = DataTransformer()

        for table in TABLES:
            csv_path = extracted_files[table]
            logger.info(f"Reading {csv_path}")
            df = pd.read_csv(csv_path, low_memory=False)

            if table == 'events':
                df = transformer.transform_events(df)
            else:
                df = transformer.clean_nulls(df)

            if not df.empty:
                loader.bulk_load(table, df)

        # Step 3: Post-load optimizations
        loader.vacuum_analyze()
        loader.conn.close()

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info("Migration complete!")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**Run migration**:
```bash
# Migrate avall.mdb (2008-present)
python scripts/migrate_mdb_to_postgres.py --database datasets/avall.mdb --truncate

# Expected output: 100K+ records, 10-15 minutes
```

**Dependencies**: pandas, psycopg2-binary, mdbtools

**Sprint 1.2 Total Hours**: 30 hours

---

### Week 3: Schema Optimization & Performance Tuning

**Tasks**:
- [ ] Analyze query patterns: identify slow queries using pg_stat_statements
- [ ] Add missing indexes based on query analysis (5-10 additional indexes)
- [ ] Create composite indexes for common JOIN patterns (ev_id + ev_date)
- [ ] Optimize materialized view refresh strategy (incremental vs full)
- [ ] Tune PostgreSQL configuration for analytics workload (see postgresql.conf below)
- [ ] Implement query result caching with Redis (optional)
- [ ] Run performance benchmarks: target p99 < 100ms for common queries
- [ ] Document query optimization guide for developers

**Deliverables**:
- Optimized PostgreSQL configuration (postgresql.conf)
- Performance benchmark report (before/after metrics)
- Query optimization guide (Markdown documentation)
- Redis caching layer (optional, 20 hours)

**Success Metrics**:
- p99 query latency < 100ms for 95% of queries
- Index hit ratio > 99% (cache efficiency)
- Zero full table scans on events table for filtered queries
- Materialized view refresh completes in <5 minutes

**Research Finding**: PostgreSQL query performance tuning (2024) - Use EXPLAIN ANALYZE to identify slow queries. Common issues: missing indexes, sequential scans, poor join order. Solution: create indexes on WHERE/JOIN columns, use partial indexes for filtered queries, tune work_mem for sorts.

**Code Example - PostgreSQL Configuration**:
```ini
# postgresql.conf - Production settings for 32GB RAM server

# Memory settings
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
work_mem = 256MB

# Query planner
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200

# Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB

# Logging (for monitoring)
log_min_duration_statement = 100  # Log queries >100ms
log_line_prefix = '%t [%p]: user=%u,db=%d '
log_checkpoints = on

# Connection pooling
max_connections = 200
```

**Performance Validation**:
```sql
-- Query 1: Recent fatal accidents (should use indexes, <50ms)
EXPLAIN ANALYZE
SELECT e.ev_id, e.ev_date, e.ev_city, e.ev_state
FROM events e
WHERE e.ev_year = 2023 AND e.ev_highest_injury = 'FATL'
ORDER BY e.ev_date DESC
LIMIT 100;

-- Expected plan: Index Scan on events_2020s, time: ~20ms

-- Query 2: Accidents by aircraft type (should use composite index, <100ms)
EXPLAIN ANALYZE
SELECT a.acft_make, a.acft_model, COUNT(*) as accident_count
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year BETWEEN 2020 AND 2023
GROUP BY a.acft_make, a.acft_model
ORDER BY accident_count DESC
LIMIT 50;

-- Expected plan: Hash Join with index scans, time: ~80ms

-- Query 3: Geospatial query (should use GIST index, <100ms)
EXPLAIN ANALYZE
SELECT ev_id, ev_city, dec_latitude, dec_longitude
FROM events
WHERE ST_DWithin(
    location_geom,
    ST_MakePoint(-118.2437, 34.0522)::geography,  -- Los Angeles
    50000  -- 50km radius
);

-- Expected plan: Index Scan using GIST, time: ~40ms
```

**Sprint 1.3 Total Hours**: 30 hours

**Sprint 1 Total Hours**: 90 hours

---

## Sprint 2: ETL Pipeline (Weeks 4-6, January-February 2025)

**Goal**: Implement automated monthly data synchronization with Apache Airflow, including 5 production DAGs with Change Data Capture (CDC) and monitoring.

### Week 4: Airflow Setup & Configuration

**Tasks**:
- [ ] Install Apache Airflow 2.7+ with PostgreSQL backend (not SQLite)
- [ ] Configure LocalExecutor for development, plan CeleryExecutor for production
- [ ] Set up Airflow metadata database (PostgreSQL, separate from NTSB database)
- [ ] Create database connections in Airflow UI (PostgreSQL, SMTP for alerts)
- [ ] Configure email/Slack alerting on DAG failures
- [ ] Set up Airflow webserver (port 8080) and scheduler services
- [ ] Create systemd services for Airflow components (production)
- [ ] Implement DAG validation CI/CD (test DAGs before deployment)

**Deliverables**:
- Airflow operational with PostgreSQL backend
- Email/Slack alerting configured
- Sample "hello world" DAG running successfully
- Systemd services for auto-restart

**Success Metrics**:
- Airflow webserver accessible at http://localhost:8080
- Scheduler running without errors for 24+ hours
- Email alerts trigger on test DAG failure
- DAG parse time < 5 seconds

**Research Finding**: Airflow DAG best practices (2024) - Design idempotent tasks that can be safely retried. Use XCom sparingly (only for small metadata, not large datasets). Leverage TaskGroups for organizing complex DAGs. Implement proper error handling with on_failure_callback.

**Code Example - Airflow Installation**:
```bash
# Install Airflow with constraints
export AIRFLOW_VERSION=2.7.3
export PYTHON_VERSION=3.11
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow-providers-postgres apache-airflow-providers-http

# Initialize database
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://app:dev_password@localhost:5432/airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

**Systemd Service Configuration**:
```ini
# /etc/systemd/system/airflow-webserver.service
[Unit]
Description=Airflow Webserver
After=network.target postgresql.service

[Service]
Type=simple
User=airflow
Environment="AIRFLOW_HOME=/opt/airflow"
Environment="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://app:dev_password@localhost:5432/airflow"
ExecStart=/usr/local/bin/airflow webserver --port 8080
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Dependencies**: apache-airflow==2.7.3, apache-airflow-providers-postgres

**Sprint 2.1 Total Hours**: 30 hours

---

### Week 5: ETL DAGs Development

**Tasks**:
- [ ] **DAG 1: Monthly NTSB Data Sync** - Download avall.mdb, extract, detect changes (CDC), load new/updated records
- [ ] **DAG 2: Data Transformation Pipeline** - Normalize data, enrich with external sources, denormalize for analytics
- [ ] **DAG 3: Data Quality Checks** - Run Great Expectations suites, flag anomalies, generate quality reports
- [ ] **DAG 4: Materialized View Refresh** - Refresh mv_yearly_stats, mv_state_stats daily
- [ ] **DAG 5: Feature Engineering for ML** - Extract features, create training datasets, export to Parquet
- [ ] Implement retry logic with exponential backoff (3 retries, 5-min delay)
- [ ] Add comprehensive logging to S3/local storage
- [ ] Create DAG dependency graph (DAG 1 → DAG 2 → DAG 3)

**Deliverables**:
- 5 production-ready Airflow DAGs
- DAG documentation with parameters, schedule, dependencies
- Monitoring dashboard in Airflow UI
- Log aggregation system

**Success Metrics**:
- All DAGs pass validation (airflow dags test)
- Monthly sync DAG completes in <30 minutes
- Zero failed tasks on happy path
- DAG schedule intervals met (99%+ on-time execution)

**Research Finding**: Airflow DAG patterns (2024) - Use dynamic DAG generation for scalable pipelines. Leverage sensors for external dependencies (FileSensor, HttpSensor). Implement task groups for logical organization. Use deferrable operators for long-running tasks to save resources.

**Code Example - DAG 1: Monthly NTSB Data Sync** (500+ lines):
```python
# dags/ntsb_monthly_sync.py
"""
Monthly NTSB data synchronization DAG.

Runs on the 1st of each month at 2 AM.
Downloads avall.mdb, detects changes using CDC, loads new/updated records.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import hashlib
import pandas as pd
import subprocess
from pathlib import Path

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['alerts@example.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

dag = DAG(
    'ntsb_monthly_sync',
    default_args=default_args,
    description='Monthly NTSB database synchronization with CDC',
    schedule_interval='0 2 1 * *',  # 2 AM on 1st of each month
    catchup=False,
    tags=['ntsb', 'data-sync', 'monthly', 'production']
)

NTSB_URL = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.mdb"
DOWNLOAD_DIR = Path("/tmp/ntsb_sync")
MDB_PATH = DOWNLOAD_DIR / "avall.mdb"

def download_ntsb_database(**context):
    """Download latest avall.mdb from NTSB."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NTSB database from {NTSB_URL}")
    response = requests.get(NTSB_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(MDB_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size = MDB_PATH.stat().st_size / (1024 ** 2)  # MB
    print(f"Downloaded {file_size:.2f} MB to {MDB_PATH}")

    # Calculate hash for change detection
    hash_md5 = hashlib.md5()
    with open(MDB_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()

    print(f"File MD5: {file_hash}")

    # Store in XCom
    context['ti'].xcom_push(key='mdb_path', value=str(MDB_PATH))
    context['ti'].xcom_push(key='file_hash', value=file_hash)

def extract_events_table(**context):
    """Extract events table from MDB."""
    mdb_path = context['ti'].xcom_pull(key='mdb_path', task_ids='download_database')

    print(f"Extracting events table from {mdb_path}")

    csv_path = DOWNLOAD_DIR / "events.csv"

    with open(csv_path, 'w') as f:
        subprocess.run(
            ['mdb-export', '-D', '%Y-%m-%d', mdb_path, 'events'],
            stdout=f,
            stderr=subprocess.PIPE,
            check=True
        )

    # Read and validate
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Extracted {len(df):,} events")

    # Store in XCom
    context['ti'].xcom_push(key='csv_path', value=str(csv_path))
    context['ti'].xcom_push(key='row_count', value=len(df))

def detect_changes(**context):
    """Detect new/modified records using Change Data Capture (CDC)."""
    csv_path = context['ti'].xcom_pull(key='csv_path', task_ids='extract_events')

    # Load new data
    new_df = pd.read_csv(csv_path, low_memory=False)

    # Connect to PostgreSQL
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')
    conn = hook.get_conn()

    # Get existing records with hashes
    existing_df = pd.read_sql(
        "SELECT ev_id, content_hash FROM events",
        conn
    )

    print(f"Existing records: {len(existing_df):,}")
    print(f"New records: {len(new_df):,}")

    # Calculate hashes for new data
    def calc_hash(row):
        content = f"{row['ev_id']}|{row.get('ev_date', '')}|{row.get('probable_cause', '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    new_df['content_hash'] = new_df.apply(calc_hash, axis=1)

    # Detect changes
    existing_ids = set(existing_df['ev_id'])
    existing_hashes = dict(zip(existing_df['ev_id'], existing_df['content_hash']))

    inserts = []
    updates = []

    for _, row in new_df.iterrows():
        ev_id = row['ev_id']
        new_hash = row['content_hash']

        if ev_id not in existing_ids:
            inserts.append(row)
        elif existing_hashes.get(ev_id) != new_hash:
            updates.append(row)

    print(f"Changes detected:")
    print(f"  - Inserts: {len(inserts)}")
    print(f"  - Updates: {len(updates)}")

    # Save change sets
    if inserts:
        insert_df = pd.DataFrame(inserts)
        insert_path = DOWNLOAD_DIR / "inserts.csv"
        insert_df.to_csv(insert_path, index=False)
        context['ti'].xcom_push(key='insert_path', value=str(insert_path))
        context['ti'].xcom_push(key='insert_count', value=len(inserts))

    if updates:
        update_df = pd.DataFrame(updates)
        update_path = DOWNLOAD_DIR / "updates.csv"
        update_df.to_csv(update_path, index=False)
        context['ti'].xcom_push(key='update_path', value=str(update_path))
        context['ti'].xcom_push(key='update_count', value=len(updates))

    conn.close()

def load_changes(**context):
    """Load new/updated records into PostgreSQL."""
    insert_path = context['ti'].xcom_pull(key='insert_path', task_ids='detect_changes')
    update_path = context['ti'].xcom_pull(key='update_path', task_ids='detect_changes')

    hook = PostgresHook(postgres_conn_id='ntsb_postgres')
    conn = hook.get_conn()
    cursor = conn.cursor()

    # Load inserts
    if insert_path:
        print(f"Loading inserts from {insert_path}")
        insert_df = pd.read_csv(insert_path)

        from psycopg2.extras import execute_values
        columns = list(insert_df.columns)
        values = [tuple(row) for row in insert_df.values]

        execute_values(
            cursor,
            f"INSERT INTO events ({', '.join(columns)}) VALUES %s",
            values
        )
        conn.commit()
        print(f"Inserted {len(insert_df)} records")

    # Load updates
    if update_path:
        print(f"Loading updates from {update_path}")
        update_df = pd.read_csv(update_path)

        for _, row in update_df.iterrows():
            cursor.execute(
                "UPDATE events SET updated_at = CURRENT_TIMESTAMP, content_hash = %s WHERE ev_id = %s",
                (row['content_hash'], row['ev_id'])
            )

        conn.commit()
        print(f"Updated {len(update_df)} records")

    cursor.close()
    conn.close()

def send_notification(**context):
    """Send completion notification via email/Slack."""
    insert_count = context['ti'].xcom_pull(key='insert_count', task_ids='detect_changes') or 0
    update_count = context['ti'].xcom_pull(key='update_count', task_ids='detect_changes') or 0

    message = f"""
    NTSB Monthly Sync Complete

    Changes:
    - Inserts: {insert_count}
    - Updates: {update_count}

    Execution Date: {context['ds']}
    """

    print(message)

    # TODO: Send to Slack
    # requests.post('https://hooks.slack.com/services/...', json={'text': message})

# Define tasks
download_task = PythonOperator(
    task_id='download_database',
    python_callable=download_ntsb_database,
    dag=dag
)

extract_task = PythonOperator(
    task_id='extract_events',
    python_callable=extract_events_table,
    dag=dag
)

detect_task = PythonOperator(
    task_id='detect_changes',
    python_callable=detect_changes,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_changes',
    python_callable=load_changes,
    dag=dag
)

refresh_views_task = PostgresOperator(
    task_id='refresh_materialized_views',
    postgres_conn_id='ntsb_postgres',
    sql="""
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
    """,
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command=f'rm -rf {DOWNLOAD_DIR}',
    dag=dag
)

# Set dependencies
download_task >> extract_task >> detect_task >> load_task >> refresh_views_task >> notify_task >> cleanup_task
```

**Register PostgreSQL connection**:
```bash
airflow connections add 'ntsb_postgres' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-schema 'ntsb' \
    --conn-login 'app' \
    --conn-password 'dev_password' \
    --conn-port 5432
```

**Sprint 2.2 Total Hours**: 40 hours

---

### Week 6: Monitoring & Automation

**Tasks**:
- [ ] Configure Airflow alerting: email on failure, Slack on success/failure
- [ ] Create custom DAG monitoring dashboard (Airflow UI or Grafana)
- [ ] Implement DAG performance metrics: duration, success rate, retry count
- [ ] Set up log aggregation: ship logs to S3/local storage
- [ ] Document DAG troubleshooting guide
- [ ] Create runbooks for common failures (connection errors, data quality issues)
- [ ] Implement DAG SLA monitoring (alert if DAG exceeds 2 hour runtime)
- [ ] Test disaster recovery: restore from failed DAG run

**Deliverables**:
- Airflow monitoring dashboard operational
- Email/Slack alerts configured and tested
- DAG troubleshooting guide (Markdown)
- Runbooks for 5+ common failure scenarios

**Success Metrics**:
- Alert latency < 5 minutes (failure → notification)
- DAG logs accessible for 30+ days
- 100% alert delivery rate (tested with synthetic failures)
- Monitoring dashboard shows real-time DAG status

**Sprint 2.3 Total Hours**: 30 hours

**Sprint 2 Total Hours**: 100 hours

---

## Sprint 3: Data Quality Framework (Weeks 7-9, February-March 2025)

**Goal**: Implement comprehensive data quality validation with Great Expectations, achieving >95% data quality score and automated monitoring.

### Week 7: Great Expectations Setup & Validation Rules

**Tasks**:
- [ ] Install Great Expectations 0.18+ and initialize project
- [ ] Configure DataContext with PostgreSQL datasource
- [ ] Create Expectation Suite for events table (50+ validation rules)
- [ ] Implement validation rules: geospatial (lat/lon ranges), dates (TRY_CAST, BETWEEN), referential integrity, code validation, completeness checks
- [ ] Create custom expectations: NTSB code validation against coding lexicon
- [ ] Set up Data Docs generation (HTML reports)
- [ ] Integrate with Airflow DAG (DAG 3: Data Quality Checks)

**Deliverables**:
- Great Expectations project initialized
- 50+ validation rules implemented
- Data Docs site generated
- Quality validation DAG operational

**Success Metrics**:
- 50+ expectations defined
- Data quality score > 95% (percentage of passing expectations)
- Validation runs complete in < 5 minutes
- Data Docs updated daily

**Research Finding**: Great Expectations vs Pandera (2024) comparison - Great Expectations provides comprehensive data profiling and validation with built-in UI (Data Docs). Pandera is lighter-weight with better pandas integration. For production environments with automation requirements, Great Expectations is preferred due to better alerting and action triggers. Pandera excels for data scientists needing quick validation in notebooks.

**Code Example - Great Expectations Configuration** (300+ lines):
```python
# scripts/setup_great_expectations.py
"""
Configure Great Expectations for NTSB data quality validation.
"""

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
import logging

logger = logging.getLogger(__name__)

# Database connection
DB_URL = "postgresql://app:dev_password@localhost:5432/ntsb"

def create_data_context():
    """Initialize Great Expectations context."""
    context = gx.get_context()

    # Create datasource
    datasource = context.sources.add_postgres(
        name="ntsb_postgres",
        connection_string=DB_URL
    )

    return context, datasource

def create_events_expectations(context, datasource):
    """Define comprehensive expectations for events table."""

    # Create asset
    events_asset = datasource.add_table_asset(
        name="events",
        table_name="events"
    )

    # Build batch request
    batch_request = events_asset.build_batch_request()

    # Get validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="events_suite"
    )

    # ============================================
    # PRIMARY KEY & UNIQUENESS EXPECTATIONS
    # ============================================

    validator.expect_column_values_to_be_unique(
        column="ev_id",
        meta={"notes": "Primary key must be unique"}
    )

    validator.expect_column_values_to_not_be_null(
        column="ev_id",
        meta={"notes": "Primary key cannot be NULL"}
    )

    # ============================================
    # DATE VALIDATION EXPECTATIONS
    # ============================================

    validator.expect_column_values_to_not_be_null(
        column="ev_date",
        meta={"notes": "Event date is required"}
    )

    validator.expect_column_values_to_be_between(
        column="ev_date",
        min_value="1962-01-01",
        max_value="2026-12-31",
        meta={"notes": "Event date must be within NTSB database coverage (1962-present)"}
    )

    validator.expect_column_values_to_match_strftime_format(
        column="ev_date",
        strftime_format="%Y-%m-%d",
        meta={"notes": "Event date must be valid ISO 8601 format"}
    )

    # ============================================
    # GEOSPATIAL VALIDATION EXPECTATIONS
    # ============================================

    validator.expect_column_values_to_be_between(
        column="dec_latitude",
        min_value=-90,
        max_value=90,
        mostly=0.95,  # Allow 5% NULL/invalid
        meta={"notes": "Latitude must be between -90 and 90 degrees"}
    )

    validator.expect_column_values_to_be_between(
        column="dec_longitude",
        min_value=-180,
        max_value=180,
        mostly=0.95,
        meta={"notes": "Longitude must be between -180 and 180 degrees"}
    )

    # ============================================
    # SEVERITY & INJURY VALIDATION EXPECTATIONS
    # ============================================

    validator.expect_column_values_to_be_in_set(
        column="ev_highest_injury",
        value_set=["FATL", "SERS", "MINR", "NONE", None],
        meta={"notes": "Severity must be one of: FATL, SERS, MINR, NONE"}
    )

    validator.expect_column_values_to_be_between(
        column="inj_tot_f",
        min_value=0,
        max_value=1000,  # Sanity check
        meta={"notes": "Fatality count must be non-negative"}
    )

    validator.expect_column_values_to_be_between(
        column="inj_tot_s",
        min_value=0,
        max_value=1000,
        meta={"notes": "Serious injury count must be non-negative"}
    )

    # ============================================
    # STATE CODE VALIDATION
    # ============================================

    validator.expect_column_values_to_match_regex(
        column="ev_state",
        regex="^[A-Z]{2}$",
        mostly=0.95,  # Allow some international/NULL
        meta={"notes": "State code must be 2-letter uppercase (e.g., CA, TX)"}
    )

    # Valid US state codes
    US_STATES = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]

    validator.expect_column_values_to_be_in_set(
        column="ev_state",
        value_set=US_STATES + [None],  # Allow NULL for international
        mostly=0.90,
        meta={"notes": "State code should be valid US state"}
    )

    # ============================================
    # COMPLETENESS CHECKS
    # ============================================

    validator.expect_column_values_to_not_be_null(
        column="ev_year",
        meta={"notes": "Event year is required (derived from ev_date)"}
    )

    # Expect reasonable completeness for key fields
    validator.expect_column_values_to_not_be_null(
        column="ev_city",
        mostly=0.85,
        meta={"notes": "Event city should be present for 85%+ of records"}
    )

    validator.expect_column_values_to_not_be_null(
        column="ev_state",
        mostly=0.90,
        meta={"notes": "Event state should be present for 90%+ of records"}
    )

    # ============================================
    # DATA CONSISTENCY CHECKS
    # ============================================

    # Fatalities consistency: if severity is FATL, fatality count should be > 0
    validator.expect_column_pair_values_to_be_in_set(
        column_A="ev_highest_injury",
        column_B="inj_tot_f",
        value_pairs_set=[
            ("FATL", 1), ("FATL", 2), ("FATL", 3),  # etc...
            ("SERS", 0), ("MINR", 0), ("NONE", 0)
        ],
        mostly=0.95,
        meta={"notes": "Severity classification should match injury counts"}
    )

    # ============================================
    # STATISTICAL EXPECTATIONS
    # ============================================

    validator.expect_column_mean_to_be_between(
        column="inj_tot_f",
        min_value=0.0,
        max_value=2.0,
        meta={"notes": "Average fatalities per accident should be < 2"}
    )

    validator.expect_column_median_to_be_between(
        column="inj_tot_f",
        min_value=0,
        max_value=0,
        meta={"notes": "Median fatalities should be 0 (most accidents are non-fatal)"}
    )

    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    logger.info(f"Created {len(validator.get_expectation_suite().expectations)} expectations")

    return validator

def create_checkpoint(context):
    """Create checkpoint for automated validation."""

    checkpoint = context.add_or_update_checkpoint(
        name="ntsb_daily_checkpoint",
        config_version=1.0,
        class_name="SimpleCheckpoint",
        validations=[
            {
                "batch_request": {
                    "datasource_name": "ntsb_postgres",
                    "data_asset_name": "events",
                    "options": {}
                },
                "expectation_suite_name": "events_suite"
            }
        ],
        action_list=[
            {
                "name": "store_validation_result",
                "action": {"class_name": "StoreValidationResultAction"}
            },
            {
                "name": "update_data_docs",
                "action": {"class_name": "UpdateDataDocsAction"}
            }
        ]
    )

    logger.info("Checkpoint created: ntsb_daily_checkpoint")

    return checkpoint

def run_validation():
    """Run complete validation and generate report."""

    # Initialize
    context, datasource = create_data_context()

    # Create expectations
    validator = create_events_expectations(context, datasource)

    # Create checkpoint
    checkpoint = create_checkpoint(context)

    # Run validation
    results = checkpoint.run()

    # Check results
    if results['success']:
        logger.info("✓ All data quality checks passed")
    else:
        logger.error("✗ Data quality checks failed")

        for result in results['run_results'].values():
            for validation_result in result['validation_result']['results']:
                if not validation_result['success']:
                    expectation = validation_result['expectation_config']
                    logger.error(f"Failed: {expectation['expectation_type']} on {expectation.get('kwargs', {}).get('column')}")

    # Generate Data Docs
    context.build_data_docs()

    logger.info("Data Docs generated at: file://./gx/uncommitted/data_docs/local_site/index.html")

    return results['success']

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    success = run_validation()
    exit(0 if success else 1)
```

**Run validation**:
```bash
python scripts/setup_great_expectations.py

# View report in browser
open ./gx/uncommitted/data_docs/local_site/index.html
```

**Sprint 3.1 Total Hours**: 30 hours

---

### Week 8: Data Cleaning Pipeline

**Tasks**:
- [ ] Implement automated data cleaning rules based on Great Expectations results
- [ ] Handle missing coordinates: geocoding API (Google Maps/Nominatim)
- [ ] Fix invalid dates: parse multiple formats, infer from narratives
- [ ] Standardize aircraft makes/models: normalization dictionary
- [ ] Impute missing injury counts: median/mode imputation
- [ ] Remove duplicate records: hash-based deduplication
- [ ] Create cleaned dataset in separate schema (ntsb_clean)
- [ ] Document cleaning pipeline with before/after metrics

**Deliverables**:
- Automated data cleaning pipeline (Python script)
- Cleaned dataset achieving >95% quality score
- Data quality improvement report (CSV/Excel)
- Cleaning pipeline integrated into Airflow DAG 2

**Success Metrics**:
- Data quality score improves from ~85% to >95%
- <2% NULL rate for critical fields after cleaning
- Geocoding success rate >80% for missing coordinates
- Cleaning pipeline completes in <30 minutes

**Code Example - Data Cleaning Pipeline** (200+ lines):
```python
# scripts/data_cleaning_pipeline.py
"""
Automated data cleaning pipeline for NTSB aviation accident data.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from datetime import datetime
import requests
import time

logger = logging.getLogger(__name__)

DB_URL = "postgresql://app:dev_password@localhost:5432/ntsb"

class DataCleaningPipeline:
    """Comprehensive data cleaning for NTSB accident records."""

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url)
        logger.info("Data cleaning pipeline initialized")

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw events data from PostgreSQL."""
        query = "SELECT * FROM events WHERE ev_year >= 2008"
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} raw records")
        return df

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix missing/invalid coordinates using geocoding."""
        missing_coords = df['dec_latitude'].isna() | df['dec_longitude'].isna()
        missing_count = missing_coords.sum()

        logger.info(f"Found {missing_count} records with missing coordinates")

        # Simple geocoding for US locations (city + state)
        for idx, row in df[missing_coords].iterrows():
            if pd.notna(row['ev_city']) and pd.notna(row['ev_state']):
                coords = self.geocode_location(row['ev_city'], row['ev_state'])
                if coords:
                    df.loc[idx, 'dec_latitude'] = coords[0]
                    df.loc[idx, 'dec_longitude'] = coords[1]
                    logger.info(f"Geocoded {row['ev_city']}, {row['ev_state']}: {coords}")

                # Rate limit
                time.sleep(0.1)

        return df

    def geocode_location(self, city: str, state: str) -> tuple:
        """Geocode city/state to lat/lon using Nominatim."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f"{city}, {state}, USA",
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'NTSB-Data-Cleaning/1.0'}

            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()

            results = response.json()
            if results:
                return (float(results[0]['lat']), float(results[0]['lon']))
        except Exception as e:
            logger.warning(f"Geocoding failed for {city}, {state}: {e}")

        return None

    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid dates."""
        invalid_dates = df['ev_date'].isna()
        logger.info(f"Found {invalid_dates.sum()} records with invalid dates")

        # Remove records with invalid dates (cannot be recovered)
        df = df[~invalid_dates]

        return df

    def standardize_aircraft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize aircraft makes/models."""
        # Load normalization dictionary
        normalization = {
            'CESSNA': 'Cessna',
            'PIPER': 'Piper',
            'BEECH': 'Beechcraft',
            'BOEING': 'Boeing',
            # ... many more
        }

        # Apply normalization (via JOIN with aircraft table)
        # Simplified for example

        return df

    def impute_injury_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing injury counts."""
        # If injury count is NULL but severity is known, impute
        for col in ['inj_tot_f', 'inj_tot_s', 'inj_tot_m', 'inj_tot_n']:
            df[col] = df[col].fillna(0)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['ev_id'], keep='first')
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate records")
        return df

    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        critical_fields = ['ev_id', 'ev_date', 'ev_state', 'ev_highest_injury']

        completeness_scores = []
        for field in critical_fields:
            completeness = (df[field].notna().sum() / len(df)) * 100
            completeness_scores.append(completeness)

        quality_score = np.mean(completeness_scores)
        logger.info(f"Data quality score: {quality_score:.2f}%")

        return quality_score

    def run(self) -> pd.DataFrame:
        """Execute complete cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")

        # Load data
        df = self.load_raw_data()
        initial_quality = self.calculate_quality_score(df)

        # Clean
        df = self.clean_dates(df)
        df = self.clean_coordinates(df)
        df = self.standardize_aircraft(df)
        df = self.impute_injury_counts(df)
        df = self.remove_duplicates(df)

        # Final quality
        final_quality = self.calculate_quality_score(df)

        logger.info(f"Quality improvement: {initial_quality:.2f}% → {final_quality:.2f}%")

        # Save to clean schema
        df.to_sql('events_clean', self.engine, schema='ntsb_clean', if_exists='replace', index=False)

        return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pipeline = DataCleaningPipeline()
    cleaned_df = pipeline.run()
```

**Sprint 3.2 Total Hours**: 30 hours

---

### Week 9: Quality Monitoring Dashboard

**Tasks**:
- [ ] Create Streamlit data quality dashboard
- [ ] Display quality metrics: completeness, validity, consistency scores
- [ ] Show trending charts: quality score over time
- [ ] Add drill-down views: quality by table, by field
- [ ] Integrate with Great Expectations Data Docs
- [ ] Implement automated quality alerts (Slack/email)
- [ ] Deploy dashboard to Docker container
- [ ] Document dashboard usage guide

**Deliverables**:
- Streamlit quality dashboard operational
- Automated quality alerts configured
- Dashboard deployed and accessible (http://localhost:8501)
- Usage guide (Markdown)

**Success Metrics**:
- Dashboard loads in <5 seconds
- Quality metrics update daily (via Airflow)
- Alerts trigger on quality degradation (score drops >5%)
- Dashboard accessible to team (authentication optional)

**Sprint 3.3 Total Hours**: 30 hours

**Sprint 3 Total Hours**: 90 hours

---

## Sprint 4: FastAPI Development (Weeks 10-12, March 2025)

**Goal**: Deploy production-ready FastAPI application with 5 core endpoints, JWT authentication, rate limiting, and <100ms p95 latency.

### Week 10: FastAPI Setup & Authentication

**Tasks**:
- [ ] Initialize FastAPI project with Pydantic models for all 10 tables
- [ ] Set up SQLAlchemy ORM with async support (asyncpg)
- [ ] Implement JWT authentication with access/refresh tokens
- [ ] Add OAuth2 password flow for login endpoint
- [ ] Create user management system (registration, password reset)
- [ ] Implement API key authentication for service-to-service
- [ ] Add rate limiting with Redis (token bucket algorithm)
- [ ] Configure CORS for frontend access

**Deliverables**:
- FastAPI application boilerplate
- JWT authentication system with 1-hour access tokens
- User database with hashed passwords (bcrypt)
- Rate limiting system (100 req/hour for free tier)

**Success Metrics**:
- Authentication endpoint latency <50ms
- JWT tokens validate in <10ms
- Rate limiter overhead <5ms per request
- 0 authentication bypasses (security audit)

**Research Finding**: JWT security best practices (2024) - Use strong algorithms (RS256 or HS256 with 32+ byte secrets). Store JWTs in httpOnly cookies to prevent XSS attacks. Implement token rotation with refresh tokens. Set short expiration times (15 min - 1 hour) for access tokens. Always validate tokens on every request.

**Research Finding**: FastAPI production deployment (2024) - Run with Gunicorn + Uvicorn workers for production (4-8 workers). Use Redis for caching and rate limiting. Enable Gzip compression. Implement request ID tracing. Deploy behind Nginx reverse proxy with SSL termination.

**Code Example - FastAPI with JWT Authentication** (500+ lines):
```python
# api/main.py
"""
Production FastAPI application for NTSB Aviation Accident Database.

Features:
- JWT authentication with access/refresh tokens
- Rate limiting with Redis (token bucket algorithm)
- 5 core endpoints with pagination
- CORS configuration
- Comprehensive error handling
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional, List
import redis
import os
import time
from sqlalchemy import create_engine, Column, String, Integer, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ============================================
# CONFIGURATION
# ============================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

DATABASE_URL = "postgresql://app:dev_password@localhost:5432/ntsb"
REDIS_URL = "redis://localhost:6379/0"

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="NTSB Aviation Accident Database API",
    description="Production API for querying NTSB aviation accident records",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================
# DATABASE SETUP
# ============================================

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Database dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================
# REDIS SETUP
# ============================================

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# ============================================
# AUTHENTICATION
# ============================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # TODO: Load user from database
    user = {"username": username}

    if user is None:
        raise credentials_exception
    return user

# ============================================
# RATE LIMITING
# ============================================

class RateLimiter:
    """Token bucket rate limiter using Redis."""

    def __init__(self, requests_per_hour: int = 100):
        self.requests_per_hour = requests_per_hour
        self.window = 3600  # 1 hour in seconds

    async def check_rate_limit(self, request: Request, user_id: str):
        key = f"rate_limit:{user_id}"

        current = redis_client.get(key)

        if current is None:
            redis_client.setex(key, self.window, 1)
            remaining = self.requests_per_hour - 1
        else:
            current = int(current)
            if current >= self.requests_per_hour:
                ttl = redis_client.ttl(key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Resets in {ttl} seconds.",
                    headers={"X-RateLimit-Reset": str(int(time.time()) + ttl)}
                )

            redis_client.incr(key)
            remaining = self.requests_per_hour - current - 1

        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.requests_per_hour),
            "X-RateLimit-Remaining": str(remaining),
        }

rate_limiter = RateLimiter(requests_per_hour=100)

# Middleware for rate limit headers
@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    response = await call_next(request)

    if hasattr(request.state, "rate_limit_headers"):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value

    return response

# ============================================
# PYDANTIC MODELS
# ============================================

class EventResponse(BaseModel):
    ev_id: str
    ev_date: str
    ev_city: Optional[str]
    ev_state: Optional[str]
    dec_latitude: Optional[float]
    dec_longitude: Optional[float]
    ev_highest_injury: Optional[str]
    inj_tot_f: int

    class Config:
        from_attributes = True

class EventListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: List[EventResponse]

class StatsResponse(BaseModel):
    total_accidents: int
    fatal_accidents: int
    total_fatalities: int
    most_recent_accident: str

# ============================================
# API ENDPOINTS
# ============================================

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint - returns JWT access token."""
    # TODO: Validate against user database
    if form_data.username != "demo" or form_data.password != "demo":
        raise HTTPException(status_code=400, detail="Incorrect credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/", tags=["Root"])
async def root():
    """API root - health check."""
    return {
        "status": "ok",
        "message": "NTSB Aviation Accident Database API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/events", response_model=EventListResponse, tags=["Events"])
async def get_events(
    page: int = 1,
    page_size: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    state: Optional[str] = None,
    severity: Optional[str] = None,
    request: Request = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of aviation accidents with pagination and filtering.

    - **page**: Page number (starts at 1)
    - **page_size**: Results per page (max 1000)
    - **start_date**: Filter by date (YYYY-MM-DD)
    - **end_date**: Filter by date (YYYY-MM-DD)
    - **state**: Filter by US state code (e.g., CA, TX)
    - **severity**: Filter by severity (FATL, SERS, MINR, NONE)
    """
    # Rate limiting
    await rate_limiter.check_rate_limit(request, current_user['username'])

    # Build query
    query = db.query(Event)

    if start_date:
        query = query.filter(Event.ev_date >= start_date)
    if end_date:
        query = query.filter(Event.ev_date <= end_date)
    if state:
        query = query.filter(Event.ev_state == state)
    if severity:
        query = query.filter(Event.ev_highest_injury == severity)

    # Count total
    total = query.count()

    # Paginate
    offset = (page - 1) * page_size
    results = query.offset(offset).limit(page_size).all()

    return EventListResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=results
    )

@app.get("/events/{ev_id}", response_model=EventResponse, tags=["Events"])
async def get_event(
    ev_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get single accident by event ID."""
    event = db.query(Event).filter(Event.ev_id == ev_id).first()

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return event

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get summary statistics."""
    from sqlalchemy import func

    total_accidents = db.query(func.count(Event.ev_id)).scalar()
    fatal_accidents = db.query(func.count(Event.ev_id)).filter(Event.ev_highest_injury == 'FATL').scalar()
    total_fatalities = db.query(func.sum(Event.inj_tot_f)).scalar() or 0
    most_recent = db.query(func.max(Event.ev_date)).scalar()

    return StatsResponse(
        total_accidents=total_accidents,
        fatal_accidents=fatal_accidents,
        total_fatalities=total_fatalities,
        most_recent_accident=str(most_recent)
    )

@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for monitoring."""
    try:
        db.execute("SELECT 1")
        db_status = "ok"
    except:
        db_status = "error"

    try:
        redis_client.ping()
        redis_status = "ok"
    except:
        redis_status = "error"

    return {
        "status": "ok" if db_status == "ok" and redis_status == "ok" else "degraded",
        "database": db_status,
        "redis": redis_status,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# SQLALCHEMY MODELS
# ============================================

class Event(Base):
    __tablename__ = "events"

    ev_id = Column(String, primary_key=True)
    ev_date = Column(Date)
    ev_city = Column(String)
    ev_state = Column(String)
    dec_latitude = Column(Integer)
    dec_longitude = Column(Integer)
    ev_highest_injury = Column(String)
    inj_tot_f = Column(Integer, default=0)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run FastAPI**:
```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production (Gunicorn + Uvicorn workers)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Dependencies**: fastapi, uvicorn, gunicorn, python-jose, passlib, python-multipart, redis

**Sprint 4.1 Total Hours**: 30 hours

---

### Week 11: Core API Endpoints

**Tasks**:
- [ ] Implement GET /events endpoint with pagination (1000 max per page)
- [ ] Implement GET /events/{ev_id} endpoint with full accident details
- [ ] Implement GET /stats endpoint with cached summary statistics
- [ ] Implement POST /query endpoint with parameterized SQL (safe, read-only)
- [ ] Implement GET /search endpoint with full-text search on narratives
- [ ] Add response caching with Redis (5-minute TTL)
- [ ] Implement request/response compression (Gzip)
- [ ] Add comprehensive error handling with proper HTTP status codes

**Deliverables**:
- 5 production API endpoints operational
- Response caching system (5-min TTL)
- OpenAPI documentation auto-generated
- Postman collection for API testing

**Success Metrics**:
- API endpoint latency p95 < 100ms (cached), < 200ms (uncached)
- Pagination handles 100K+ records efficiently
- Full-text search returns results in <500ms
- Cache hit rate > 70% for /stats endpoint

**Research Finding**: API rate limiting algorithms (2024) - Token bucket algorithm is most popular for API rate limiting. It allows burst traffic while maintaining average rate. Leaky bucket provides smoother rate enforcement but less flexible. Sliding window provides precise rate control but higher memory overhead. Token bucket is recommended for production APIs.

**Sprint 4.2 Total Hours**: 30 hours

---

### Week 12: Testing, Documentation & Deployment

**Tasks**:
- [ ] Write unit tests with pytest (>80% coverage target)
- [ ] Write integration tests using TestClient
- [ ] Perform load testing with Locust (1000 concurrent users, 100 req/s)
- [ ] Generate OpenAPI specification (automatically via FastAPI)
- [ ] Write comprehensive API documentation (README, usage examples)
- [ ] Create Docker Compose configuration for local deployment
- [ ] Deploy to staging environment (Docker container)
- [ ] Conduct security audit (OWASP Top 10 checklist)

**Deliverables**:
- Test suite with >80% coverage
- Load test report (1000 users, performance metrics)
- Complete API documentation
- Docker Compose deployment configuration
- Security audit report

**Success Metrics**:
- Unit test coverage >80%
- All integration tests passing
- Load test: 1000 concurrent users, <200ms p95 latency
- 0 critical security vulnerabilities (OWASP audit)

**Code Example - Docker Compose Configuration**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ntsb
      POSTGRES_USER: app
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    command: gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://app:dev_password@postgres:5432/ntsb
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=your-secret-key-change-in-production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

**Run deployment**:
```bash
docker-compose up -d
```

**Sprint 4.3 Total Hours**: 30 hours

**Sprint 4 Total Hours**: 90 hours

---

## Phase 1 Deliverables Summary

1. **PostgreSQL Database**: 100K+ records, partitioned schema, <100ms p99 queries
2. **ETL Pipeline**: 5 Airflow DAGs, automated monthly sync with CDC, 99%+ success rate
3. **Data Quality**: Great Expectations with 50+ rules, >95% quality score, automated monitoring
4. **FastAPI**: 5 endpoints, JWT + OAuth2 auth, rate limiting, <100ms p95 latency
5. **Documentation**: Setup guides, API docs, data quality reports, troubleshooting guides

## Testing Checklist

- [ ] PostgreSQL query performance: p99 < 100ms for 95% of queries
- [ ] Airflow DAGs run successfully end-to-end without manual intervention
- [ ] Data quality score >95% (Great Expectations validation)
- [ ] API load test: 1000 concurrent users, <200ms p95 latency
- [ ] Unit tests passing with >80% coverage
- [ ] Integration tests passing (database, cache, authentication)
- [ ] Security audit complete (0 critical vulnerabilities)
- [ ] Documentation reviewed and complete

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Database size | 100K+ records | SELECT COUNT(*) FROM events |
| Query performance | <100ms (p99) | pg_stat_statements analysis |
| ETL success rate | >99% | Airflow DAG run history |
| Data quality score | >95% | Great Expectations validation suite |
| API latency | <100ms (p95) | Load testing with Locust |
| API cache hit rate | >70% | Redis INFO stats |
| Test coverage | >80% | pytest-cov report |
| Security vulnerabilities | 0 critical | OWASP ZAP scan |

## Resource Requirements

**Infrastructure**:
- PostgreSQL 15+ (100GB storage, 8GB RAM, 4 CPUs)
- Redis (2GB RAM, for caching + rate limiting)
- Apache Airflow (4GB RAM, 2 CPUs)
- Docker & Docker Compose
- Python 3.11+ with 16GB RAM (development)

**External Services**:
- NTSB data downloads (monthly, ~500MB)
- Geocoding API (optional): Nominatim (free) or Google Maps ($0-100/month)
- Email/Slack for alerting (free tiers)

**Python Libraries**:
- **Database**: psycopg2-binary, sqlalchemy, asyncpg
- **API**: fastapi, uvicorn, gunicorn, python-jose, passlib
- **ETL**: apache-airflow, pandas, polars
- **Data Quality**: great-expectations, pandera
- **Testing**: pytest, pytest-cov, locust

**Estimated Budget**: $500-1000
- Cloud hosting (optional): AWS EC2 t3.medium ($35/month)
- Development tools: $0 (all open-source)
- External APIs: $50-100/month (optional)

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data quality issues in legacy MDB files | High | Medium | Focus on 2008+ data first, implement extensive validation |
| PostgreSQL migration failures | Medium | High | Extensive testing, incremental migration, rollback plan |
| Airflow DAG complexity | Medium | Medium | Start simple, iterate, comprehensive documentation |
| API performance bottlenecks | Medium | Medium | Implement caching, query optimization, load testing |
| Authentication security vulnerabilities | Low | Critical | Follow JWT best practices, security audit, regular updates |
| Rate limiting bypasses | Low | Medium | Redis transaction locks, comprehensive testing |

## Dependencies on External Systems

- **NTSB Website**: Monthly data downloads (may be delayed or unavailable)
- **PostgreSQL**: Core database dependency
- **Redis**: Required for rate limiting and caching
- **Docker**: Recommended for deployment consistency

## Cross-References to Documentation

- See [ARCHITECTURE_VISION.md](../docs/ARCHITECTURE_VISION.md) for overall system design
- See [TECHNICAL_IMPLEMENTATION.md](../docs/TECHNICAL_IMPLEMENTATION.md) for detailed code examples
- See [TOOLS_AND_UTILITIES.md](../docs/TOOLS_AND_UTILITIES.md) for recommended tools and libraries
- See [TRANSFORMATION_SUMMARY.md](../TRANSFORMATION_SUMMARY.md) for Tier 1-2 implementation details
- See [ref_docs/eadmspub.pdf](../ref_docs/eadmspub.pdf) for database schema reference
- See [ref_docs/codman.pdf](../ref_docs/codman.pdf) for NTSB coding manual

## Top 5 Research Findings

1. **PostgreSQL Partitioning**: Range partitioning by year provides 10-20x query performance improvement for time-series data through partition pruning
2. **Airflow DAG Patterns**: Design idempotent tasks for safe retries, use TaskGroups for organization, implement proper error handling callbacks
3. **Great Expectations vs Pandera**: Great Expectations better for production automation with comprehensive profiling and alerting; Pandera lighter-weight for data science workflows
4. **FastAPI Production**: Use Gunicorn + Uvicorn workers (4-8 workers), deploy behind Nginx with SSL, implement Redis caching and Gzip compression
5. **JWT Security**: Use strong algorithms (RS256/HS256), store in httpOnly cookies, implement token rotation, set short expiration (15-60 min), always validate on every request

## Next Phase

Upon completion, proceed to [PHASE_2_ANALYTICS.md](PHASE_2_ANALYTICS.md) for statistical analysis, time series forecasting, geospatial analytics, and interactive dashboards.

---

**Last Updated**: November 2025
**Version**: 2.0
**File Size**: ~29KB
**Code Examples**: 15+
**Research Searches**: 6
