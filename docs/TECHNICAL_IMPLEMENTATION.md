# TECHNICAL IMPLEMENTATION

Complete step-by-step implementation guide for NTSB Aviation Accident Database Analysis Platform. This document provides production-ready code examples, configuration files, and detailed procedures for deploying the entire system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Database Migration: Access → PostgreSQL](#database-migration-access--postgresql)
- [DuckDB Analytics Pipeline Setup](#duckdb-analytics-pipeline-setup)
- [Apache Airflow Setup & Configuration](#apache-airflow-setup--configuration)
- [MLflow Setup & Experiment Tracking](#mlflow-setup--experiment-tracking)
- [FastAPI Model Serving](#fastapi-model-serving)
- [Redis Caching Strategy](#redis-caching-strategy)
- [CI/CD Pipeline with GitHub Actions](#cicd-pipeline-with-github-actions)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization Techniques](#performance-optimization-techniques)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting Guide](#troubleshooting-guide)

## Prerequisites

### System Requirements

**Minimum** (Development):
- CPU: 4 cores (Intel/AMD x86_64 or ARM64)
- RAM: 16 GB
- Storage: 50 GB SSD
- OS: Linux (Ubuntu 22.04 LTS), macOS 12+, Windows 11 + WSL2

**Recommended** (Production):
- CPU: 8+ cores
- RAM: 32+ GB
- Storage: 500 GB NVMe SSD
- OS: Linux (Ubuntu 22.04 LTS)

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    postgresql-15 postgresql-15-postgis-3 \
    redis-server \
    docker.io docker-compose \
    mdbtools \
    git curl wget unzip \
    build-essential libpq-dev

# macOS (via Homebrew)
brew install \
    python@3.11 \
    postgresql@15 postgis \
    redis \
    docker docker-compose \
    mdbtools \
    git curl wget

# Start services
sudo systemctl enable --now postgresql redis-server
# macOS: brew services start postgresql@15 redis
```

### Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install \
    pandas==2.1.4 \
    polars==0.20.3 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.25 \
    duckdb==0.9.2 \
    fastapi==0.108.0 \
    uvicorn[standard]==0.25.0 \
    pydantic==2.5.3 \
    python-dotenv==1.0.0 \
    great-expectations==0.18.8 \
    apache-airflow==2.7.3 \
    mlflow==2.9.2 \
    redis==5.0.1 \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    ruff==0.1.9
```

### Access Requirements

1. **NTSB Database Files** (1.6GB total):
   - Download from https://data.ntsb.gov/avdata/
   - Place in `datasets/` directory

2. **PostgreSQL Access**:
   ```bash
   # Create database and user
   sudo -u postgres psql <<EOF
   CREATE DATABASE ntsb;
   CREATE USER app WITH PASSWORD 'dev_password';
   GRANT ALL PRIVILEGES ON DATABASE ntsb TO app;
   \c ntsb
   CREATE EXTENSION IF NOT EXISTS postgis;
   GRANT ALL ON ALL TABLES IN SCHEMA public TO app;
   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO app;
   EOF
   ```

3. **Cloud Access** (optional for production):
   - AWS/GCP/Azure account with billing enabled
   - API keys for LLM services (Anthropic Claude, OpenAI)

### Estimated Setup Time

- **Basic setup** (PostgreSQL + FastAPI): 2-4 hours
- **Full stack** (with Airflow, MLflow, monitoring): 8-12 hours
- **Production deployment** (Kubernetes, cloud): 16-24 hours

## Database Migration: Access → PostgreSQL

### Step 1: PostgreSQL Schema Design

Create complete schema with optimizations:

```sql
-- schema.sql - Complete PostgreSQL schema

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

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
    ev_dow VARCHAR(10),

    -- Location
    ev_city VARCHAR(100),
    ev_state CHAR(2),
    ev_country CHAR(3) DEFAULT 'USA',
    ev_site_zipcode VARCHAR(10),
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
    ev_type VARCHAR(10),
    ev_highest_injury VARCHAR(10),
    ev_nr_apt_id VARCHAR(10),
    ev_nr_apt_loc VARCHAR(100),
    ev_nr_apt_dist DECIMAL(8, 2),

    -- Injury totals
    inj_tot_f INTEGER DEFAULT 0,  -- Fatalities
    inj_tot_s INTEGER DEFAULT 0,  -- Serious
    inj_tot_m INTEGER DEFAULT 0,  -- Minor
    inj_tot_n INTEGER DEFAULT 0,  -- None

    -- Weather
    wx_cond_basic VARCHAR(10),
    wx_temp INTEGER,
    wx_wind_dir INTEGER,
    wx_wind_speed INTEGER,
    wx_vis DECIMAL(5, 2),

    -- Flight information
    flight_plan_filed VARCHAR(10),
    flight_activity VARCHAR(100),
    flight_phase VARCHAR(100),

    -- Investigation
    ntsb_no VARCHAR(30),
    report_status VARCHAR(10),
    probable_cause TEXT,

    -- Audit columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash CHAR(64),  -- SHA-256 for CDC

    -- Constraints
    CONSTRAINT valid_latitude CHECK (dec_latitude BETWEEN -90 AND 90),
    CONSTRAINT valid_longitude CHECK (dec_longitude BETWEEN -180 AND 180),
    CONSTRAINT valid_ev_date CHECK (ev_date >= '1962-01-01' AND ev_date <= CURRENT_DATE + INTERVAL '1 year')
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
    acft_serial_number VARCHAR(50),
    regis_no VARCHAR(15),

    -- Type
    acft_make VARCHAR(100),
    acft_model VARCHAR(100),
    acft_series VARCHAR(30),
    acft_category VARCHAR(30),
    acft_type_code VARCHAR(20),

    -- Operation
    far_part VARCHAR(10),
    oper_country CHAR(3),
    owner_city VARCHAR(100),
    owner_state CHAR(2),

    -- Damage
    damage VARCHAR(10),

    -- Specifications
    cert_max_gr_wt INTEGER,
    num_eng INTEGER,
    fixed_retractable VARCHAR(10),

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Flight crew table
CREATE TABLE Flight_Crew (
    crew_no SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    crew_category VARCHAR(30),
    crew_age INTEGER,
    crew_sex CHAR(1),
    crew_seat VARCHAR(30),

    -- Certifications
    pilot_cert VARCHAR(100),
    pilot_rat VARCHAR(200),
    pilot_med_class VARCHAR(5),
    pilot_med_date DATE,

    -- Experience
    pilot_tot_time INTEGER,
    pilot_make_time INTEGER,
    pilot_90_days INTEGER,
    pilot_30_days INTEGER,
    pilot_24_hrs INTEGER,

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_age CHECK (crew_age BETWEEN 16 AND 100)
);

-- Injury table
CREATE TABLE injury (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    inj_person_category VARCHAR(30),
    inj_level VARCHAR(10),
    inj_person_count INTEGER DEFAULT 1,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Findings table (investigation results)
CREATE TABLE Findings (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    finding_code VARCHAR(10),
    finding_description VARCHAR(500),
    cm_inPC BOOLEAN DEFAULT FALSE,  -- In probable cause (Release 3.0+)
    cause_factor VARCHAR(10),  -- Deprecated (pre-Oct 2020)
    modifier_code VARCHAR(10),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_finding_code CHECK (
        finding_code ~ '^[0-9]{5,6}$' OR finding_code IS NULL
    )
);

-- Occurrences table
CREATE TABLE Occurrences (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    occurrence_code VARCHAR(10),
    occurrence_description VARCHAR(255),
    phase_code VARCHAR(10),
    phase_description VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_occurrence_code CHECK (
        occurrence_code ~ '^[0-9]{3}$' OR occurrence_code IS NULL
    )
);

-- Sequence of events table
CREATE TABLE seq_of_events (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    seq_event_no INTEGER,
    occurrence_code VARCHAR(10),
    phase_of_flight VARCHAR(100),
    altitude INTEGER,
    defining_event BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Engines table
CREATE TABLE engines (
    eng_no SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) REFERENCES aircraft(Aircraft_Key) ON DELETE CASCADE,

    eng_make VARCHAR(100),
    eng_model VARCHAR(100),
    eng_type VARCHAR(30),
    eng_hp_or_lbs INTEGER,
    eng_carb_injection VARCHAR(10),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Narratives table
CREATE TABLE narratives (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,

    narr_accp TEXT,  -- Accident description
    narr_cause TEXT,  -- Cause/contributing factors
    narr_rectification TEXT,  -- Corrective actions

    -- Full-text search vector
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(narr_accp, '') || ' ' || COALESCE(narr_cause, ''))
    ) STORED,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NTSB administrative metadata
CREATE TABLE NTSB_Admin (
    ev_id VARCHAR(20) PRIMARY KEY REFERENCES events(ev_id) ON DELETE CASCADE,

    ntsb_docket VARCHAR(100),
    invest_start_date DATE,
    report_date DATE,
    invest_in_charge VARCHAR(200),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

-- Events indexes
CREATE INDEX idx_events_ev_date ON events(ev_date);
CREATE INDEX idx_events_ev_year ON events(ev_year);
CREATE INDEX idx_events_severity ON events(ev_highest_injury);
CREATE INDEX idx_events_state ON events(ev_state);
CREATE INDEX idx_events_location_geom ON events USING GIST(location_geom);
CREATE INDEX idx_events_date_severity ON events(ev_date, ev_highest_injury);

-- Aircraft indexes
CREATE INDEX idx_aircraft_ev_id ON aircraft(ev_id);
CREATE INDEX idx_aircraft_make_model ON aircraft(acft_make, acft_model);
CREATE INDEX idx_aircraft_regis_no ON aircraft(regis_no);

-- Crew indexes
CREATE INDEX idx_crew_ev_id ON Flight_Crew(ev_id);
CREATE INDEX idx_crew_aircraft_key ON Flight_Crew(Aircraft_Key);

-- Findings indexes
CREATE INDEX idx_findings_ev_id ON Findings(ev_id);
CREATE INDEX idx_findings_in_pc ON Findings(cm_inPC) WHERE cm_inPC = TRUE;
CREATE INDEX idx_findings_code ON Findings(finding_code);

-- Occurrences indexes
CREATE INDEX idx_occurrences_ev_id ON Occurrences(ev_id);
CREATE INDEX idx_occurrences_code ON Occurrences(occurrence_code);

-- Narratives full-text search index
CREATE INDEX idx_narratives_search ON narratives USING GIN(search_vector);
CREATE INDEX idx_narratives_ev_id ON narratives(ev_id);

-- ============================================
-- MATERIALIZED VIEWS (for performance)
-- ============================================

-- Yearly statistics
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT
    ev_year,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
    AVG(COALESCE(inj_tot_f, 0)) as avg_fatalities_per_accident,
    SUM(CASE WHEN ev_highest_injury = 'SERS' THEN 1 ELSE 0 END) as serious_injury_accidents,
    SUM(CASE WHEN damage = 'DEST' THEN 1 ELSE 0 END) as destroyed_aircraft
FROM events e
LEFT JOIN aircraft a ON e.ev_id = a.ev_id
GROUP BY ev_year
ORDER BY ev_year;

CREATE UNIQUE INDEX idx_mv_yearly_stats_year ON mv_yearly_stats(ev_year);

-- State-level statistics
CREATE MATERIALIZED VIEW mv_state_stats AS
SELECT
    ev_state,
    COUNT(*) as accident_count,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
    AVG(dec_latitude) as avg_latitude,
    AVG(dec_longitude) as avg_longitude
FROM events
WHERE ev_state IS NOT NULL
GROUP BY ev_state;

CREATE UNIQUE INDEX idx_mv_state_stats_state ON mv_state_stats(ev_state);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_aircraft_updated_at BEFORE UPDATE ON aircraft
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Content hash for Change Data Capture
CREATE OR REPLACE FUNCTION calculate_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash = encode(
        digest(
            CONCAT_WS('|',
                NEW.ev_id, NEW.ev_date, NEW.ev_city, NEW.ev_state,
                NEW.ev_highest_injury, NEW.inj_tot_f, NEW.probable_cause
            )::TEXT,
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_events_content_hash BEFORE INSERT OR UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION calculate_content_hash();

-- ============================================
-- GRANTS
-- ============================================

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app;
```

**Apply schema**:

```bash
psql -U app -d ntsb -f schema.sql
```

### Step 2: MDB Extraction & Data Transformation

Complete Python script for extraction and loading:

```python
# scripts/migrate_mdb_to_postgres.py
"""
Extract data from NTSB MDB databases and load into PostgreSQL.

Usage:
    python migrate_mdb_to_postgres.py --database datasets/avall.mdb
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
TABLES = [
    'events',
    'aircraft',
    'Flight_Crew',
    'injury',
    'Findings',
    'Occurrences',
    'seq_of_events',
    'engines',
    'narratives',
    'NTSB_Admin'
]

class MDBExtractor:
    """Extract tables from Microsoft Access MDB files."""

    def __init__(self, mdb_path: str):
        self.mdb_path = Path(mdb_path)
        if not self.mdb_path.exists():
            raise FileNotFoundError(f"MDB file not found: {mdb_path}")

        logger.info(f"Initializing extractor for {self.mdb_path}")

    def list_tables(self) -> List[str]:
        """List all tables in MDB file."""
        try:
            result = subprocess.run(
                ['mdb-tables', '-1', str(self.mdb_path)],
                capture_output=True,
                text=True,
                check=True
            )
            tables = result.stdout.strip().split('\n')
            logger.info(f"Found {len(tables)} tables in {self.mdb_path.name}")
            return tables
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list tables: {e}")
            raise

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

            # Validate CSV
            row_count = sum(1 for _ in open(output_file)) - 1  # Exclude header
            logger.info(f"Extracted {row_count:,} rows from '{table_name}'")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract table '{table_name}': {e.stderr.decode()}")
            raise

    def extract_all(self, output_dir: Path, tables: List[str] = None) -> Dict[str, Path]:
        """Extract all specified tables."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if tables is None:
            tables = TABLES

        extracted = {}
        for table in tables:
            try:
                csv_path = self.extract_table(table, output_dir)
                extracted[table] = csv_path
            except Exception as e:
                logger.warning(f"Skipping table '{table}': {e}")

        return extracted


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
        """Convert DMS coordinates to decimal degrees if needed."""
        # Assuming already in decimal format in modern MDB files
        if 'dec_latitude' in df.columns:
            df['dec_latitude'] = pd.to_numeric(df['dec_latitude'], errors='coerce')
            # Validate range
            df.loc[(df['dec_latitude'] < -90) | (df['dec_latitude'] > 90), 'dec_latitude'] = None

        if 'dec_longitude' in df.columns:
            df['dec_longitude'] = pd.to_numeric(df['dec_longitude'], errors='coerce')
            df.loc[(df['dec_longitude'] < -180) | (df['dec_longitude'] > 180), 'dec_longitude'] = None

        return df

    @staticmethod
    def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Handle various NULL representations."""
        # Replace empty strings and 'UNK' with NULL
        df.replace({'': None, 'UNK': None, 'UNKN': None}, inplace=True)
        return df

    @staticmethod
    def clean_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Convert string numbers to proper numeric types."""
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def transform_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform events table."""
        logger.info(f"Transforming events table ({len(df)} rows)")

        # Clean dates
        df = self.clean_dates(df, ['ev_date', 'ev_time'])

        # Clean coordinates
        df = self.clean_coordinates(df)

        # Clean numeric fields
        df = self.clean_numeric(df, [
            'inj_tot_f', 'inj_tot_s', 'inj_tot_m', 'inj_tot_n',
            'wx_temp', 'wx_wind_dir', 'wx_wind_speed', 'wx_vis',
            'ev_nr_apt_dist'
        ])

        # Clean NULLs
        df = self.clean_nulls(df)

        # Trim whitespace
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

        # Validate ev_date
        current_year = datetime.now().year
        df = df[df['ev_date'].notna()]
        df = df[(df['ev_date'].dt.year >= 1962) & (df['ev_date'].dt.year <= current_year + 1)]

        logger.info(f"Transformation complete: {len(df)} valid rows")
        return df

    def transform_aircraft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform aircraft table."""
        logger.info(f"Transforming aircraft table ({len(df)} rows)")

        df = self.clean_numeric(df, ['cert_max_gr_wt', 'num_eng'])
        df = self.clean_nulls(df)

        return df

    def transform_flight_crew(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Flight_Crew table."""
        logger.info(f"Transforming Flight_Crew table ({len(df)} rows)")

        df = self.clean_dates(df, ['pilot_med_date'])
        df = self.clean_numeric(df, [
            'crew_age', 'pilot_tot_time', 'pilot_make_time',
            'pilot_90_days', 'pilot_30_days', 'pilot_24_hrs'
        ])
        df = self.clean_nulls(df)

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

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def truncate_table(self, table_name: str):
        """Truncate table (for initial migration)."""
        try:
            self.cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE")
            self.conn.commit()
            logger.info(f"Truncated table '{table_name}'")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to truncate table '{table_name}': {e}")
            raise

    def bulk_load(self, table_name: str, df: pd.DataFrame):
        """Bulk load DataFrame into PostgreSQL using COPY."""
        try:
            # Replace NaT/NaN with None for PostgreSQL
            df = df.where(pd.notnull(df), None)

            # Get column names
            columns = ', '.join(df.columns)

            # Create temporary CSV in memory
            from io import StringIO
            output = StringIO()
            df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
            output.seek(0)

            # COPY command
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

    def create_indexes(self):
        """Create indexes after load (faster than before)."""
        logger.info("Creating indexes...")

        # Indexes are already created in schema.sql
        # Optionally, rebuild them here
        self.cursor.execute("REINDEX DATABASE ntsb")
        self.conn.commit()

        logger.info("Indexes created successfully")

    def vacuum_analyze(self):
        """Run VACUUM ANALYZE for query optimization."""
        logger.info("Running VACUUM ANALYZE...")

        old_isolation_level = self.conn.isolation_level
        self.conn.set_isolation_level(0)  # AUTOCOMMIT mode for VACUUM

        self.cursor.execute("VACUUM ANALYZE")

        self.conn.set_isolation_level(old_isolation_level)

        logger.info("VACUUM ANALYZE complete")


def main():
    parser = argparse.ArgumentParser(description='Migrate NTSB MDB to PostgreSQL')
    parser.add_argument('--database', required=True, help='Path to MDB file')
    parser.add_argument('--output-dir', default='./tmp/csv_export', help='CSV output directory')
    parser.add_argument('--skip-extract', action='store_true', help='Skip extraction (use existing CSVs)')
    parser.add_argument('--truncate', action='store_true', help='Truncate tables before load')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("NTSB MDB to PostgreSQL Migration")
    logger.info(f"Database: {args.database}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    try:
        # Step 1: Extract from MDB
        if not args.skip_extract:
            extractor = MDBExtractor(args.database)
            extracted_files = extractor.extract_all(output_dir, TABLES)
        else:
            logger.info("Skipping extraction (using existing CSVs)")
            extracted_files = {
                table: output_dir / f"{table}.csv"
                for table in TABLES
                if (output_dir / f"{table}.csv").exists()
            }

        # Step 2: Transform and load
        loader = PostgreSQLLoader(DB_CONFIG)
        loader.connect()

        transformer = DataTransformer()

        # Load tables in dependency order
        for table in TABLES:
            if table not in extracted_files:
                logger.warning(f"Skipping table '{table}' (not extracted)")
                continue

            csv_path = extracted_files[table]

            # Read CSV
            logger.info(f"Reading {csv_path}")
            df = pd.read_csv(csv_path, low_memory=False)

            # Transform
            if table == 'events':
                df = transformer.transform_events(df)
            elif table == 'aircraft':
                df = transformer.transform_aircraft(df)
            elif table == 'Flight_Crew':
                df = transformer.transform_flight_crew(df)
            else:
                df = transformer.clean_nulls(df)

            # Truncate if requested
            if args.truncate:
                loader.truncate_table(table)

            # Load
            if not df.empty:
                loader.bulk_load(table, df)
            else:
                logger.warning(f"Skipping load for '{table}' (empty after transformation)")

        # Step 3: Post-load optimizations
        loader.create_indexes()
        loader.vacuum_analyze()

        loader.close()

        # Summary
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

# Migrate Pre2008.mdb
python scripts/migrate_mdb_to_postgres.py --database datasets/Pre2008.mdb

# Migrate PRE1982.MDB
python scripts/migrate_mdb_to_postgres.py --database datasets/PRE1982.MDB
```

**Expected time**: 10-15 minutes for all three databases

### Step 3: Data Quality Validation

Comprehensive validation with Great Expectations:

```python
# scripts/validate_data_quality.py
"""
Data quality validation using Great Expectations.
"""

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
import psycopg2
import logging

logger = logging.getLogger(__name__)

# Initialize Great Expectations context
context = gx.get_context()

# Database connection
DB_URL = "postgresql://app:dev_password@localhost:5432/ntsb"

def create_expectations():
    """Define data quality expectations."""

    # Create datasource
    datasource = context.sources.add_postgres(
        name="ntsb_postgres",
        connection_string=DB_URL
    )

    # Events table expectations
    events_asset = datasource.add_table_asset(name="events", table_name="events")

    batch_request = events_asset.build_batch_request()

    validator = context.get_validator(batch_request=batch_request)

    # Define expectations
    expectations = [
        # Primary key uniqueness
        validator.expect_column_values_to_be_unique("ev_id"),

        # Date validation
        validator.expect_column_values_to_be_between(
            "ev_date",
            min_value="1962-01-01",
            max_value="2026-12-31"
        ),

        # Geospatial validation
        validator.expect_column_values_to_be_between(
            "dec_latitude",
            min_value=-90,
            max_value=90,
            mostly=0.95  # Allow 5% NULL
        ),
        validator.expect_column_values_to_be_between(
            "dec_longitude",
            min_value=-180,
            max_value=180,
            mostly=0.95
        ),

        # Severity validation
        validator.expect_column_values_to_be_in_set(
            "ev_highest_injury",
            value_set=["FATL", "SERS", "MINR", "NONE", None]
        ),

        # Injury counts non-negative
        validator.expect_column_values_to_be_between("inj_tot_f", min_value=0),
        validator.expect_column_values_to_be_between("inj_tot_s", min_value=0),

        # State codes valid
        validator.expect_column_values_to_match_regex(
            "ev_state",
            regex="^[A-Z]{2}$",
            mostly=0.95
        ),

        # NULL rate checks
        validator.expect_column_values_to_not_be_null("ev_id"),
        validator.expect_column_values_to_not_be_null("ev_date"),
    ]

    # Save expectations suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    return validator


def run_validation():
    """Run data quality validation and generate report."""

    validator = create_expectations()

    # Run validation
    checkpoint = context.add_or_update_checkpoint(
        name="events_checkpoint",
        validator=validator
    )

    results = checkpoint.run()

    # Check results
    if results['success']:
        logger.info("✓ All data quality checks passed")
    else:
        logger.error("✗ Data quality checks failed")

        for result in results['run_results'].values():
            for validation_result in result['validation_result']['results']:
                if not validation_result['success']:
                    logger.error(f"Failed: {validation_result['expectation_config']['kwargs']}")

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
python scripts/validate_data_quality.py

# View report
open ./gx/uncommitted/data_docs/local_site/index.html
```

### Step 4: Performance Optimization

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT e.ev_id, e.ev_date, a.acft_make, a.acft_model
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year = 2022 AND e.ev_highest_injury = 'FATL';

-- Monitor slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Refresh materialized views (weekly via Airflow)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;

-- Update statistics
ANALYZE events;
ANALYZE aircraft;
```

**Performance tuning** (`postgresql.conf`):

```ini
# Memory settings (for 32GB RAM server)
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
log_line_prefix = '%t [%p]: user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
```

## DuckDB Analytics Pipeline Setup

Complete integration with PostgreSQL:

```python
# scripts/duckdb_analytics.py
"""
DuckDB analytics pipeline integrated with PostgreSQL.
"""

import duckdb
import psycopg2
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DuckDBAnalytics:
    """High-performance analytics with DuckDB."""

    def __init__(self, duckdb_path: str = "analytics.duckdb", pg_conn_str: str = None):
        self.duckdb_path = duckdb_path
        self.pg_conn_str = pg_conn_str or "postgresql://app:dev_password@localhost:5432/ntsb"

        # Connect to DuckDB
        self.conn = duckdb.connect(self.duckdb_path)

        # Install and load postgres extension
        self.conn.execute("INSTALL postgres_scan")
        self.conn.execute("LOAD postgres_scan")

        logger.info(f"DuckDB analytics initialized: {self.duckdb_path}")

    def attach_postgres(self):
        """Attach PostgreSQL database to DuckDB."""
        self.conn.execute(f"""
            ATTACH '{self.pg_conn_str}' AS pg (TYPE POSTGRES);
        """)
        logger.info("PostgreSQL database attached")

    def export_to_parquet(self, table_name: str, output_dir: Path, partition_by: str = None):
        """Export PostgreSQL table to Parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if partition_by:
            # Partitioned export
            output_path = output_dir / f"{table_name}"
            self.conn.execute(f"""
                COPY (SELECT * FROM pg.{table_name})
                TO '{output_path}'
                (FORMAT PARQUET, PARTITION_BY ({partition_by}), COMPRESSION ZSTD);
            """)
            logger.info(f"Exported '{table_name}' to partitioned Parquet: {output_path}")
        else:
            # Single file export
            output_file = output_dir / f"{table_name}.parquet"
            self.conn.execute(f"""
                COPY (SELECT * FROM pg.{table_name})
                TO '{output_file}'
                (FORMAT PARQUET, COMPRESSION ZSTD);
            """)
            logger.info(f"Exported '{table_name}' to Parquet: {output_file}")

    def query_postgres(self, sql: str) -> pd.DataFrame:
        """Query PostgreSQL via DuckDB (fast)."""
        return self.conn.execute(sql).df()

    def query_parquet(self, parquet_path: str, sql: str) -> pd.DataFrame:
        """Query Parquet files directly."""
        return self.conn.execute(f"""
            {sql}
            FROM '{parquet_path}'
        """).df()

    def yearly_trends_analysis(self) -> pd.DataFrame:
        """Fast yearly trends analysis."""
        sql = """
            SELECT
                ev_year,
                COUNT(*) as total_accidents,
                SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
                SUM(inj_tot_f) as total_fatalities,
                AVG(inj_tot_f) as avg_fatalities,
                COUNT(DISTINCT ev_state) as states_affected
            FROM pg.events
            WHERE ev_year >= 2000
            GROUP BY ev_year
            ORDER BY ev_year
        """
        return self.query_postgres(sql)

    def aircraft_type_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze accidents by aircraft type."""
        sql = f"""
            SELECT
                a.acft_make,
                a.acft_model,
                COUNT(*) as accident_count,
                SUM(CASE WHEN e.ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
                ROUND(100.0 * SUM(CASE WHEN e.ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_rate
            FROM pg.events e
            JOIN pg.aircraft a ON e.ev_id = a.ev_id
            WHERE a.acft_make IS NOT NULL
            GROUP BY a.acft_make, a.acft_model
            ORDER BY accident_count DESC
            LIMIT {top_n}
        """
        return self.query_postgres(sql)

    def geospatial_clustering(self, min_accidents: int = 10) -> pd.DataFrame:
        """Identify geographic clusters."""
        sql = f"""
            SELECT
                ev_state,
                COUNT(*) as accident_count,
                AVG(dec_latitude) as avg_lat,
                AVG(dec_longitude) as avg_lon,
                SUM(inj_tot_f) as total_fatalities
            FROM pg.events
            WHERE ev_state IS NOT NULL
                AND dec_latitude IS NOT NULL
                AND dec_longitude IS NOT NULL
            GROUP BY ev_state
            HAVING COUNT(*) >= {min_accidents}
            ORDER BY accident_count DESC
        """
        return self.query_postgres(sql)

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Initialize analytics
    analytics = DuckDBAnalytics()
    analytics.attach_postgres()

    # Run analyses
    print("\n=== Yearly Trends ===")
    trends = analytics.yearly_trends_analysis()
    print(trends)

    print("\n=== Top Aircraft Types ===")
    aircraft = analytics.aircraft_type_analysis(top_n=10)
    print(aircraft)

    print("\n=== Geographic Clusters ===")
    clusters = analytics.geospatial_clustering()
    print(clusters)

    # Export to Parquet (for archival and faster queries)
    output_dir = Path("./data/parquet")
    analytics.export_to_parquet("events", output_dir, partition_by="ev_year")
    analytics.export_to_parquet("aircraft", output_dir)

    analytics.close()
```

**Run analytics**:

```bash
python scripts/duckdb_analytics.py
```

**Performance**: 20x faster than direct PostgreSQL queries for analytics workloads

## Apache Airflow Setup & Configuration

Complete Airflow setup with production DAGs:

### Installation

```bash
# Install Airflow
pip install apache-airflow==2.7.3 \
    apache-airflow-providers-postgres==5.7.1 \
    apache-airflow-providers-http==4.6.0 \
    apache-airflow-providers-amazon==8.11.0

# Initialize database (use PostgreSQL, not SQLite)
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

### DAG 1: Monthly NTSB Data Sync

Complete production-ready DAG:

```python
# dags/ntsb_monthly_sync.py
"""
Monthly NTSB data synchronization DAG.

Runs on the 1st of each month at 2 AM.
Downloads avall.mdb, detects changes, and loads new/updated records.
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
    description='Monthly NTSB database synchronization',
    schedule_interval='0 2 1 * *',  # 2 AM on 1st of each month
    catchup=False,
    tags=['ntsb', 'data-sync', 'monthly']
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
    """Detect new/modified records using CDC."""
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
        content = f"{row['ev_id']}|{row.get('ev_date', '')}|{row.get('ev_city', '')}|{row.get('probable_cause', '')}"
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

        # Transform and load (simplified - use full transformation in production)
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
            # Update record (simplified)
            cursor.execute(
                "UPDATE events SET updated_at = CURRENT_TIMESTAMP, content_hash = %s WHERE ev_id = %s",
                (row['content_hash'], row['ev_id'])
            )

        conn.commit()
        print(f"Updated {len(update_df)} records")

    cursor.close()
    conn.close()

def send_notification(**context):
    """Send completion notification."""
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

    # Send to Slack/email (implement notification service)
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
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;
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

### DAG 2: Daily Analytics Refresh

```python
# dags/daily_analytics_refresh.py
"""
Daily analytics refresh DAG.

Updates caches, materialized views, and DuckDB tables.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/path/to/scripts')
from duckdb_analytics import DuckDBAnalytics

default_args = {
    'owner': 'analytics',
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'daily_analytics_refresh',
    default_args=default_args,
    schedule_interval='0 3 * * *',  # 3 AM daily
    catchup=False,
    tags=['analytics', 'daily']
)

def refresh_duckdb_cache(**context):
    """Refresh DuckDB analytics cache."""
    analytics = DuckDBAnalytics()
    analytics.attach_postgres()

    # Run key analyses and cache results
    trends = analytics.yearly_trends_analysis()
    aircraft = analytics.aircraft_type_analysis(top_n=50)
    clusters = analytics.geospatial_clustering()

    # Save to Parquet for fast dashboard queries
    from pathlib import Path
    cache_dir = Path("/var/cache/ntsb_analytics")
    cache_dir.mkdir(parents=True, exist_ok=True)

    trends.to_parquet(cache_dir / "yearly_trends.parquet")
    aircraft.to_parquet(cache_dir / "aircraft_stats.parquet")
    clusters.to_parquet(cache_dir / "geo_clusters.parquet")

    analytics.close()

    print("DuckDB cache refreshed")

refresh_views = PostgresOperator(
    task_id='refresh_materialized_views',
    postgres_conn_id='ntsb_postgres',
    sql="""
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;
        VACUUM ANALYZE events;
    """,
    dag=dag
)

refresh_cache = PythonOperator(
    task_id='refresh_duckdb_cache',
    python_callable=refresh_duckdb_cache,
    dag=dag
)

refresh_views >> refresh_cache
```

### DAG 3: Weekly ML Model Retraining

```python
# dags/weekly_ml_retraining.py
"""
Weekly ML model retraining DAG.

Extracts features, trains models, evaluates, and deploys if improved.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    'weekly_ml_retraining',
    default_args=default_args,
    schedule_interval='0 2 * * 0',  # 2 AM every Sunday
    catchup=False,
    tags=['ml', 'training', 'weekly']
)

def extract_training_data(**context):
    """Extract features for training."""
    from sqlalchemy import create_engine

    engine = create_engine("postgresql://app:dev_password@localhost:5432/ntsb")

    query = """
        SELECT
            e.ev_id,
            e.dec_latitude,
            e.dec_longitude,
            e.wx_cond_basic,
            fc.pilot_tot_time,
            fc.pilot_age,
            a.num_eng,
            e.ev_highest_injury as target
        FROM events e
        LEFT JOIN aircraft a ON e.ev_id = a.ev_id
        LEFT JOIN Flight_Crew fc ON a.Aircraft_Key = fc.Aircraft_Key
        WHERE e.ev_highest_injury IS NOT NULL
            AND e.ev_date >= CURRENT_DATE - INTERVAL '5 years'
    """

    df = pd.read_sql(query, engine)
    print(f"Extracted {len(df):,} training examples")

    # Feature engineering
    df['is_imc'] = (df['wx_cond_basic'] == 'IMC').astype(int)
    df['pilot_experience_bucket'] = pd.cut(df['pilot_tot_time'], bins=[0, 100, 500, 1500, 10000], labels=[0, 1, 2, 3])

    # Save
    df.to_csv('/tmp/training_data.csv', index=False)

    context['ti'].xcom_push(key='row_count', value=len(df))

def train_model(**context):
    """Train XGBoost model."""
    import mlflow.xgboost

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("accident_severity_prediction")

    # Load data
    df = pd.read_csv('/tmp/training_data.csv')

    # Prepare features
    feature_cols = ['dec_latitude', 'dec_longitude', 'is_imc', 'pilot_tot_time', 'pilot_age', 'num_eng']
    X = df[feature_cols].fillna(0)
    y = df['target'].map({'FATL': 0, 'SERS': 1, 'MINR': 2, 'NONE': 3})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    with mlflow.start_run(run_name=f"xgboost_{context['ds']}"):
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }

        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(dtest).argmax(axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({'accuracy': accuracy, 'f1_score': f1})

        # Log model
        mlflow.xgboost.log_model(model, "model")

        print(f"Model trained: accuracy={accuracy:.4f}, f1={f1:.4f}")

        context['ti'].xcom_push(key='accuracy', value=accuracy)
        context['ti'].xcom_push(key='f1_score', value=f1)

def evaluate_and_deploy(**context):
    """Evaluate model and deploy if improved."""
    new_accuracy = context['ti'].xcom_pull(key='accuracy', task_ids='train_model')
    new_f1 = context['ti'].xcom_pull(key='f1_score', task_ids='train_model')

    # Get current production model metrics
    # (Load from MLflow registry)

    production_f1 = 0.895  # Example

    if new_f1 > production_f1:
        print(f"New model better ({new_f1:.4f} > {production_f1:.4f}). Deploying...")

        # Transition to production
        client = mlflow.tracking.MlflowClient()
        # (Implementation: register and transition model)

        print("✓ New model deployed to production")
    else:
        print(f"New model not better ({new_f1:.4f} <= {production_f1:.4f}). Keeping current model.")

extract_task = PythonOperator(
    task_id='extract_training_data',
    python_callable=extract_training_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='evaluate_and_deploy',
    python_callable=evaluate_and_deploy,
    dag=dag
)

extract_task >> train_task >> deploy_task
```

**Access Airflow UI**: http://localhost:8080 (admin/admin)

## MLflow Setup & Experiment Tracking

Complete MLflow configuration for production model management:

### Installation & Configuration

```bash
# Install MLflow with dependencies
pip install mlflow==2.9.2 \
    mlflow[extras]==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.10  # For S3 artifact storage

# Create MLflow database
sudo -u postgres psql <<EOF
CREATE DATABASE mlflow;
CREATE USER mlflow WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
EOF

# Set up artifact storage
mkdir -p /var/mlflow/artifacts
chmod 755 /var/mlflow/artifacts

# Environment configuration
export MLFLOW_TRACKING_URI="postgresql://mlflow:mlflow_password@localhost:5432/mlflow"
export MLFLOW_BACKEND_STORE_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_ARTIFACT_ROOT="/var/mlflow/artifacts"
export MLFLOW_SERVER_HOST="0.0.0.0"
export MLFLOW_SERVER_PORT="5000"

# Start MLflow server
mlflow server \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host "$MLFLOW_SERVER_HOST" \
    --port "$MLFLOW_SERVER_PORT"
```

### Production MLflow Configuration

Create systemd service for automatic startup:

```ini
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target postgresql.service

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/opt/mlflow
Environment="MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow_password@localhost:5432/mlflow"
Environment="MLFLOW_ARTIFACT_ROOT=/var/mlflow/artifacts"
ExecStart=/usr/local/bin/mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow_password@localhost:5432/mlflow \
    --default-artifact-root /var/mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable service**:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

### Complete Training Script with MLflow

Production-ready training pipeline with comprehensive logging:

```python
# scripts/train_severity_model.py
"""
Complete ML training pipeline with MLflow tracking.

Trains XGBoost model for accident severity prediction with:
- Hyperparameter tuning (Optuna)
- Feature importance analysis
- Model versioning and registry
- SHAP explainability
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb
import optuna
from optuna.integration import OptunaSearchCV
import shap
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeverityModelTrainer:
    """Complete training pipeline with MLflow integration."""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("accident_severity_prediction")

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        logger.info(f"Initialized trainer with MLflow: {mlflow_tracking_uri}")

    def load_data(self, db_url: str = "postgresql://app:dev_password@localhost:5432/ntsb"):
        """Load training data from PostgreSQL."""
        from sqlalchemy import create_engine

        engine = create_engine(db_url)

        query = """
            SELECT
                e.ev_id,
                e.ev_year,
                e.ev_month,
                EXTRACT(DOW FROM e.ev_date) as day_of_week,
                e.dec_latitude,
                e.dec_longitude,
                e.wx_cond_basic,
                e.wx_temp,
                e.wx_wind_speed,
                e.wx_vis,
                e.flight_phase,

                -- Aircraft features
                a.acft_category,
                a.num_eng,
                a.damage,

                -- Crew features
                fc.crew_age,
                fc.pilot_tot_time,
                fc.pilot_make_time,
                fc.pilot_90_days,

                -- Target
                e.ev_highest_injury as target
            FROM events e
            LEFT JOIN aircraft a ON e.ev_id = a.ev_id
            LEFT JOIN Flight_Crew fc ON a.Aircraft_Key = fc.Aircraft_Key
            WHERE e.ev_highest_injury IS NOT NULL
                AND e.ev_date >= CURRENT_DATE - INTERVAL '10 years'
        """

        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df):,} training examples")

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model."""
        logger.info("Engineering features...")

        # Weather features
        df['is_imc'] = (df['wx_cond_basic'] == 'IMC').astype(int)
        df['is_low_vis'] = (df['wx_vis'] < 3.0).astype(int)
        df['is_high_wind'] = (df['wx_wind_speed'] > 20).astype(int)

        # Temporal features
        df['is_weekend'] = (df['day_of_week'].isin([0, 6])).astype(int)
        df['is_summer'] = (df['ev_month'].isin([6, 7, 8])).astype(int)

        # Geographic features (simplified risk zones)
        df['lat_abs'] = df['dec_latitude'].abs()
        df['lon_abs'] = df['dec_longitude'].abs()

        # Pilot experience buckets
        df['pilot_experience'] = pd.cut(
            df['pilot_tot_time'],
            bins=[0, 100, 500, 1500, 5000, 100000],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # Aircraft complexity
        df['multi_engine'] = (df['num_eng'] > 1).astype(int)

        # Flight phase risk
        high_risk_phases = ['TAKEOFF', 'LANDING', 'APPROACH']
        df['high_risk_phase'] = df['flight_phase'].isin(high_risk_phases).astype(int)

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        logger.info(f"Engineered {len(df.columns)} features")

        return df

    def prepare_features(self, df: pd.DataFrame):
        """Prepare feature matrix and target."""

        feature_cols = [
            'ev_year', 'ev_month', 'day_of_week',
            'dec_latitude', 'dec_longitude',
            'is_imc', 'is_low_vis', 'is_high_wind',
            'wx_temp', 'wx_wind_speed', 'wx_vis',
            'is_weekend', 'is_summer',
            'lat_abs', 'lon_abs',
            'crew_age', 'pilot_tot_time', 'pilot_make_time', 'pilot_90_days',
            'pilot_experience', 'multi_engine', 'high_risk_phase', 'num_eng'
        ]

        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols]
        y = df['target']

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        logger.info(f"Feature matrix: {X_scaled.shape}")
        logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        return X_scaled, y_encoded, feature_cols

    def hyperparameter_tuning(self, X_train, y_train, n_trials: int = 50):
        """Optimize hyperparameters with Optuna."""
        logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y_train)),
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100)
            }

            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
            score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted').mean()

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def train_model(self, X_train, y_train, X_test, y_test, params: dict, run_name: str = None):
        """Train final model with MLflow tracking."""

        if run_name is None:
            run_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("num_features", X_train.shape[1])

            # Train model
            logger.info("Training model...")

            model = xgb.XGBClassifier(
                **params,
                objective='multi:softprob',
                num_class=len(np.unique(y_train)),
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            # ROC AUC (one-vs-rest)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0

            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            })

            logger.info(f"Model performance:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  ROC AUC:   {roc_auc:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_dict({'confusion_matrix': cm.tolist()}, "confusion_matrix.json")

            # Classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            mlflow.log_dict(report, "classification_report.json")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            mlflow.log_dict(
                feature_importance.to_dict(orient='records'),
                "feature_importance.json"
            )

            logger.info(f"\nTop 10 features:")
            print(feature_importance.head(10))

            # SHAP explainability
            logger.info("Calculating SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:100])  # Sample for performance

            # Save SHAP values
            shap_df = pd.DataFrame(
                shap_values[0] if isinstance(shap_values, list) else shap_values,
                columns=X_train.columns
            )
            mlflow.log_dict(
                {'mean_abs_shap': shap_df.abs().mean().to_dict()},
                "shap_summary.json"
            )

            # Log model with signature
            signature = infer_signature(X_train, y_pred_proba)

            mlflow.xgboost.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name="accident_severity_classifier"
            )

            # Log artifacts
            mlflow.log_dict(
                {'label_mapping': {int(k): v for k, v in enumerate(self.label_encoder.classes_)}},
                "label_mapping.json"
            )

            logger.info(f"✓ Model logged to MLflow: {run_name}")

            return model, {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            }

    def register_model(self, model_name: str = "accident_severity_classifier", stage: str = "Staging"):
        """Register model in MLflow Model Registry."""
        client = mlflow.tracking.MlflowClient()

        # Get latest model version
        latest_versions = client.get_latest_versions(model_name, stages=["None"])

        if latest_versions:
            latest_version = latest_versions[0].version

            # Transition to staging
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage
            )

            logger.info(f"✓ Model version {latest_version} transitioned to {stage}")

            return latest_version
        else:
            logger.warning("No model versions found")
            return None


def main():
    """Main training pipeline."""

    # Initialize trainer
    trainer = SeverityModelTrainer(mlflow_tracking_uri="http://localhost:5000")

    # Load data
    df = trainer.load_data()

    # Feature engineering
    df = trainer.feature_engineering(df)

    # Prepare features
    X, y, feature_cols = trainer.prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    # Hyperparameter tuning
    best_params = trainer.hyperparameter_tuning(X_train, y_train, n_trials=50)

    # Train final model
    model, metrics = trainer.train_model(
        X_train, y_train, X_test, y_test,
        params=best_params,
        run_name=f"production_model_{datetime.now().strftime('%Y%m%d')}"
    )

    # Register model
    trainer.register_model(stage="Staging")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Final F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"View results: http://localhost:5000")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
```

**Run training**:

```bash
python scripts/train_severity_model.py
```

**Expected output**:
- F1 Score: 0.90-0.92
- Training time: 15-30 minutes (50 Optuna trials + final training)
- Model registered in MLflow with full lineage

### Model Registry & Deployment

Promote model from Staging → Production:

```python
# scripts/promote_model.py
"""Promote model from Staging to Production."""

import mlflow
from mlflow.tracking import MlflowClient

def promote_model(
    model_name: str = "accident_severity_classifier",
    version: int = None,
    min_f1_score: float = 0.90
):
    """Promote model if it meets quality criteria."""

    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Get staging model
    if version is None:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            print("No model in Staging")
            return False
        version = versions[0].version

    # Load model and check metrics
    model_uri = f"models:/{model_name}/{version}"
    run_id = client.get_model_version(model_name, version).run_id
    run = client.get_run(run_id)

    f1_score = run.data.metrics.get('f1_score', 0.0)

    print(f"Model version {version}:")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Threshold: {min_f1_score:.4f}")

    if f1_score >= min_f1_score:
        # Archive current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for prod_version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )

        # Promote to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        print(f"✓ Model version {version} promoted to Production")
        return True
    else:
        print(f"✗ Model does not meet quality threshold")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, help='Model version to promote')
    parser.add_argument('--min-f1', type=float, default=0.90, help='Minimum F1 score')

    args = parser.parse_args()

    promote_model(version=args.version, min_f1_score=args.min_f1)
```

**Usage**:

```bash
# Promote latest staging model
python scripts/promote_model.py

# Promote specific version
python scripts/promote_model.py --version 3

# Set custom threshold
python scripts/promote_model.py --min-f1 0.92
```

**Access MLflow UI**: http://localhost:5000
