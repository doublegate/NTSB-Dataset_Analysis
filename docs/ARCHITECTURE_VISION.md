# ARCHITECTURE VISION

Complete system architecture for NTSB Aviation Accident Database Analysis Platform. This document defines the production-ready architecture supporting 100K+ accident records, machine learning pipelines, AI-powered analysis, and real-time API services.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Complete System Architecture](#complete-system-architecture)
- [Data Ingestion Pipeline](#data-ingestion-pipeline)
- [ETL/ELT Architecture](#etlelt-architecture)
- [Data Warehouse Design](#data-warehouse-design)
- [Analytics Layer](#analytics-layer)
- [ML Model Serving Infrastructure](#ml-model-serving-infrastructure)
- [API Architecture](#api-architecture)
- [Dashboard & UI Architecture](#dashboard--ui-architecture)
- [Scalability Considerations](#scalability-considerations)
- [Cloud Deployment Comparison](#cloud-deployment-comparison)
- [Containerization Strategy](#containerization-strategy)
- [Monitoring & Observability](#monitoring--observability)
- [Disaster Recovery & Backup](#disaster-recovery--backup)
- [Security Architecture](#security-architecture)
- [Future Expansion Paths](#future-expansion-paths)

## Executive Summary

### System Vision

Transform 60+ years of NTSB aviation accident data (1.6GB, 100K+ events) into a production-ready analytics platform that enables:

- **Predictive Safety Intelligence**: 90%+ accuracy severity prediction using XGBoost ensemble models
- **AI-Powered Causal Analysis**: LLM-based narrative analysis with RAG and knowledge graphs
- **Real-Time Risk Assessment**: <100ms API latency for predictive analytics
- **Open Research Platform**: Public API with rate limiting for academic and industry use

### High-Level Architecture Overview

The platform follows a modern data lakehouse architecture with 7 distinct layers:

1. **Data Layer**: PostgreSQL (warehouse) + DuckDB (analytics) + Parquet (archive) + S3/MinIO (object storage)
2. **Processing Layer**: Apache Airflow orchestration with Spark/Dask for distributed computing
3. **Analytics Layer**: DuckDB for 20x faster OLAP queries, statistical analysis, geospatial processing
4. **ML Layer**: MLflow experiment tracking, XGBoost/Random Forest models, SHAP explainability
5. **AI Layer**: LLM integration (Claude/GPT), RAG with vector databases, Neo4j knowledge graphs
6. **Application Layer**: FastAPI REST endpoints, Streamlit/Dash dashboards, automated reporting
7. **Infrastructure Layer**: Docker containers, Kubernetes orchestration, Prometheus/Grafana monitoring

### Key Design Principles

1. **Scalability**: Horizontal scaling via Kubernetes, partitioned databases, distributed computing
2. **Reliability**: 99.9% uptime target, automated backups, multi-region replication
3. **Maintainability**: Modular architecture, comprehensive documentation, CI/CD automation
4. **Performance**: <100ms API latency (p95), <2s dashboard load times, sub-second SQL queries
5. **Security**: JWT authentication, TLS encryption, rate limiting, secrets management
6. **Cost-Efficiency**: Open-source tools, cloud-agnostic design, auto-scaling resources

## Complete System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        NTSB_MDB[NTSB MDB Databases<br/>avall.mdb 537MB<br/>Pre2008.mdb 893MB<br/>PRE1982.MDB 188MB]
        NTSB_API[NTSB Monthly Updates<br/>aviationquery.ntsb.gov]
        EXTERNAL[External Sources<br/>FAA, ASRS, Weather APIs]
    end

    subgraph "Data Lake Layer"
        S3[Object Storage<br/>S3/MinIO<br/>Raw MDB + Parquet]
        PARQUET[Parquet Archive<br/>Compressed 5-10x<br/>Partitioned by Year]
    end

    subgraph "Data Warehouse Layer"
        PG[(PostgreSQL 15+<br/>Primary Data Warehouse<br/>Normalized Schema<br/>Partitioned by Year)]
        DUCK[DuckDB Analytics Engine<br/>OLAP Queries<br/>20x faster than PG]
    end

    subgraph "Processing Layer"
        AIRFLOW[Apache Airflow<br/>Workflow Orchestration<br/>5+ DAGs]
        SPARK[Spark/Dask<br/>Distributed Processing<br/>Feature Engineering]
        CDC[Change Data Capture<br/>Incremental Updates<br/>Delta Detection]
    end

    subgraph "Analytics Layer"
        SQL_ANALYTICS[SQL Analytics<br/>DuckDB Queries<br/>Materialized Views]
        STATS[Statistical Analysis<br/>scipy, statsmodels<br/>Time Series ARIMA/Prophet]
        GIS[Geospatial Analysis<br/>geopandas, PostGIS<br/>DBSCAN Clustering]
    end

    subgraph "ML Layer"
        MLFLOW[MLflow Tracking<br/>Experiment Management<br/>Model Registry]
        TRAINING[Model Training<br/>XGBoost, Random Forest<br/>LSTM, Neural Networks]
        FEATURE_STORE[Feature Store<br/>Cached Features<br/>Real-time + Batch]
        SHAP_EXPLAIN[SHAP Explainability<br/>Model Interpretation<br/>Feature Importance]
    end

    subgraph "AI Layer"
        LLM[LLM Integration<br/>Claude Sonnet/GPT-4<br/>Report Generation]
        RAG[RAG System<br/>FAISS/Chroma Vector DB<br/>100K+ Documents]
        KG[Knowledge Graph<br/>Neo4j<br/>50K+ Entities]
        CAUSAL[Causal Inference<br/>DoWhy + LLM<br/>Root Cause Analysis]
    end

    subgraph "Application Layer"
        API[FastAPI REST API<br/>JWT Auth + Rate Limiting<br/>OpenAPI Docs]
        DASH_UI[Streamlit Dashboard<br/>Interactive Visualizations<br/>Real-time Updates]
        REPORTS[Automated Reports<br/>PDF/HTML Generation<br/>Email Delivery]
    end

    subgraph "Infrastructure Layer"
        DOCKER[Docker Containers<br/>Multi-stage Builds<br/>Image Registry]
        K8S[Kubernetes<br/>Auto-scaling<br/>Load Balancing]
        MONITOR[Prometheus + Grafana<br/>Metrics + Alerts<br/>99.9% SLA]
        REDIS[Redis Cache<br/>Query Results<br/>Rate Limiting]
    end

    NTSB_MDB --> S3
    NTSB_API --> AIRFLOW
    EXTERNAL --> AIRFLOW
    S3 --> AIRFLOW

    AIRFLOW --> CDC
    CDC --> PG
    AIRFLOW --> PARQUET
    PARQUET --> DUCK
    PG --> DUCK

    PG --> SPARK
    SPARK --> FEATURE_STORE

    DUCK --> SQL_ANALYTICS
    DUCK --> STATS
    PG --> GIS

    FEATURE_STORE --> TRAINING
    TRAINING --> MLFLOW
    MLFLOW --> SHAP_EXPLAIN

    SQL_ANALYTICS --> LLM
    STATS --> LLM
    TRAINING --> RAG
    PG --> KG
    KG --> CAUSAL

    MLFLOW --> API
    LLM --> API
    RAG --> API
    CAUSAL --> API

    API --> DASH_UI
    API --> REPORTS

    API --> REDIS
    DASH_UI --> DOCKER
    API --> DOCKER
    DOCKER --> K8S
    K8S --> MONITOR
```

## Data Ingestion Pipeline

### Automated NTSB Monthly Updates

**Source**: NTSB updates `avall.mdb` monthly with new accident records at https://data.ntsb.gov/avdata

**Architecture**:

```mermaid
flowchart LR
    SCHEDULE[Airflow Scheduler<br/>1st of each month]
    DOWNLOAD[Download avall.mdb<br/>~540MB]
    EXTRACT[Extract Tables<br/>mdb-export]
    VALIDATE[Data Validation<br/>Great Expectations<br/>50+ Rules]
    COMPARE[Delta Detection<br/>Compare with existing]
    LOAD[Incremental Load<br/>INSERT new records]
    NOTIFY[Notification<br/>Slack/Email]

    SCHEDULE --> DOWNLOAD
    DOWNLOAD --> EXTRACT
    EXTRACT --> VALIDATE
    VALIDATE --> COMPARE
    COMPARE --> LOAD
    LOAD --> NOTIFY

    VALIDATE -->|Validation Failed| NOTIFY
```

**Key Features**:
- **Incremental Updates**: Only new/modified records processed (CDC pattern)
- **Data Validation**: 50+ validation rules before load (dates, coordinates, codes)
- **Error Handling**: Automatic retry with exponential backoff (3 attempts)
- **Monitoring**: Airflow SLA monitoring with alerts on failures
- **Rollback**: Transaction-based loading with rollback on errors

### Change Data Capture (CDC) Strategy

**Problem**: NTSB updates existing preliminary reports to final status, requiring detection of changes.

**Solution**: Hash-based CDC with audit logging

```python
# Pseudocode for CDC logic
def detect_changes(new_records, existing_records):
    """
    Compare new records with existing using content hash.

    Returns:
        inserts: List of new records to INSERT
        updates: List of changed records to UPDATE
        unchanged: Count of unchanged records
    """
    inserts = []
    updates = []
    unchanged_count = 0

    for new_rec in new_records:
        content_hash = hashlib.sha256(
            json.dumps(new_rec, sort_keys=True).encode()
        ).hexdigest()

        existing = existing_records.get(new_rec['ev_id'])

        if not existing:
            inserts.append(new_rec)
        elif existing['content_hash'] != content_hash:
            updates.append({
                'ev_id': new_rec['ev_id'],
                'new_data': new_rec,
                'old_hash': existing['content_hash'],
                'new_hash': content_hash
            })
        else:
            unchanged_count += 1

    return inserts, updates, unchanged_count
```

**Performance**: Processes 30K events in <5 minutes (PostgreSQL bulk operations)

### Data Validation Gates

**Gate 1: Schema Validation**
- Column presence and data types
- Primary key uniqueness (ev_id, Aircraft_Key)
- Foreign key referential integrity

**Gate 2: Business Rules**
- Geospatial: `-90 <= lat <= 90`, `-180 <= lon <= 180`
- Date: `1962-01-01 <= ev_date <= TODAY`
- Injury counts: `inj_tot_f + inj_tot_s + inj_tot_m + inj_tot_n = total_persons`
- Code validation: Occurrence codes in 100-430 range, Finding codes in 10000-93300

**Gate 3: Statistical Anomalies**
- Outlier detection: Flag records with unusual values (e.g., altitude >50000 ft for GA)
- Temporal anomalies: Sudden spikes in accident rates
- Missing data: Flag records with >30% NULL fields

**Action on Failure**: Quarantine invalid records in `staging_quarantine` table for manual review

## ETL/ELT Architecture

### Extract Phase

**Source Systems**:
1. **MDB Databases** (3 files): Use `mdb-tools` for Linux/Mac, `pyodbc` for Windows
2. **NTSB API**: REST API for metadata and updates
3. **External APIs**: FAA registry, weather data (NOAA), airport data

**Extract Tools**:
```bash
# Linux/Mac: mdb-tools
mdb-export -D "%Y-%m-%d" database.mdb events > events.csv

# Windows: Python with pyodbc
import pyodbc
conn = pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb)};DBQ=database.mdb')
df = pd.read_sql('SELECT * FROM events', conn)
```

**Output**: CSV files staged in S3/MinIO object storage

### Transform Phase

**Transformations Applied**:

1. **Data Type Conversion**
   - Dates: Multiple formats → ISO 8601 (`YYYY-MM-DD`)
   - Coordinates: DMS format → Decimal degrees
   - Numeric: String numbers → INTEGER/DECIMAL

2. **Data Cleaning**
   - NULL handling: Empty strings → NULL, "UNK" → NULL
   - Whitespace: Trim leading/trailing spaces
   - Case standardization: Aircraft makes/models to Title Case

3. **Data Enrichment**
   - Temporal features: Extract year, month, day of week, quarter
   - Geospatial: Calculate nearest airport distance
   - Derived fields: Power-to-weight ratio, experience ratios

4. **Code Standardization**
   - Map deprecated codes to current codes
   - Expand abbreviations (e.g., "PLT" → "Pilot")

**Technology**: Spark for large datasets (>1M rows), Pandas for smaller batches

### Load Phase

**Loading Strategy**: Bulk loading with COPY command (10x faster than INSERT)

```sql
-- Fast bulk load from CSV
COPY events (ev_id, ev_date, ev_city, ...)
FROM '/tmp/events.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', NULL 'NULL');

-- Create indexes after load (faster than before)
CREATE INDEX idx_events_ev_id ON events(ev_id);
CREATE INDEX idx_events_ev_date ON events(ev_date);
CREATE INDEX idx_events_dec_lat_lon ON events USING GIST(
    ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)
);
```

**Performance Tuning**:
- Disable indexes during load, rebuild after
- Use `UNLOGGED` tables for staging (no WAL overhead)
- Increase `maintenance_work_mem` for index creation
- Parallel loading with multiple workers

**Expected Performance**: Load 100K events in <10 minutes

### ELT vs ETL Decision Matrix

| Criteria | ETL (Transform before load) | ELT (Load then transform) | Choice |
|----------|----------------------------|---------------------------|--------|
| Data Volume | <1M rows/batch | >1M rows/batch | **ELT** for monthly updates |
| Transformation Complexity | Simple (type conversion) | Complex (aggregations, joins) | **ETL** for initial migration |
| Target System Performance | Limited | High (PostgreSQL, Spark) | **ELT** leverages PG power |
| Data Quality Requirements | High (must be clean) | Medium (clean in warehouse) | **ETL** for critical fields |
| Development Speed | Slower (external tools) | Faster (SQL in warehouse) | **ELT** for rapid iteration |

**Recommendation**: Hybrid approach
- **ETL** for initial data migration and critical cleaning
- **ELT** for monthly updates and analytics transformations

## Data Warehouse Design

### Star Schema Design

**Fact Table**: `fact_accidents` (one row per accident event)

**Dimension Tables**:
1. `dim_time` - Date dimension with calendar attributes
2. `dim_location` - Geographic dimension (state, county, airport)
3. `dim_aircraft` - Aircraft type dimension (make, model, category)
4. `dim_crew` - Crew dimension (certifications, experience levels)
5. `dim_weather` - Weather condition dimension (VMC/IMC, temperature ranges)
6. `dim_causes` - Causal factor dimension (finding codes, occurrence codes)

```mermaid
erDiagram
    fact_accidents {
        BIGINT accident_key PK
        INTEGER time_key FK
        INTEGER location_key FK
        INTEGER aircraft_key FK
        INTEGER crew_key FK
        INTEGER weather_key FK
        STRING ev_id UK
        STRING severity
        INTEGER fatalities
        INTEGER injuries_serious
        INTEGER injuries_minor
        DECIMAL damage_cost_usd
        TIMESTAMP created_at
        TIMESTAMP updated_at
    }

    dim_time {
        INTEGER time_key PK
        DATE full_date UK
        INTEGER year
        INTEGER quarter
        INTEGER month
        INTEGER day
        INTEGER day_of_week
        BOOLEAN is_weekend
        BOOLEAN is_holiday
    }

    dim_location {
        INTEGER location_key PK
        STRING state
        STRING county
        STRING city
        STRING nearest_airport_id
        DECIMAL latitude
        DECIMAL longitude
        GEOGRAPHY point_geom
    }

    dim_aircraft {
        INTEGER aircraft_key PK
        STRING make
        STRING model
        STRING category
        STRING engine_type
        INTEGER num_engines
        INTEGER year_manufactured
    }

    dim_crew {
        INTEGER crew_key PK
        STRING certification_level
        STRING experience_bucket
        INTEGER total_hours_bucket
        INTEGER age_group
    }

    dim_weather {
        INTEGER weather_key PK
        STRING condition_basic
        STRING visibility_bucket
        STRING wind_speed_bucket
        STRING temperature_bucket
    }

    dim_causes {
        INTEGER cause_key PK
        STRING finding_code
        STRING finding_category
        STRING occurrence_code
        STRING phase_of_flight
        BOOLEAN in_probable_cause
    }

    fact_accidents ||--o{ dim_time : "occurs on"
    fact_accidents ||--o{ dim_location : "located at"
    fact_accidents ||--o{ dim_aircraft : "involves"
    fact_accidents ||--o{ dim_crew : "operated by"
    fact_accidents ||--o{ dim_weather : "occurred in"
    fact_accidents }o--o{ dim_causes : "caused by"
```

### Slowly Changing Dimensions (SCD)

**Type 1 SCD** (Overwrite): For corrections (e.g., fixing typos in aircraft make)
**Type 2 SCD** (Historical tracking): For aircraft modifications, crew certifications

```sql
-- Type 2 SCD example: dim_aircraft with history
CREATE TABLE dim_aircraft (
    aircraft_key SERIAL PRIMARY KEY,
    aircraft_natural_key VARCHAR(50) NOT NULL, -- regis_no
    make VARCHAR(100),
    model VARCHAR(100),
    category VARCHAR(50),
    num_engines INTEGER,

    -- SCD Type 2 columns
    valid_from DATE NOT NULL DEFAULT CURRENT_DATE,
    valid_to DATE DEFAULT '9999-12-31',
    is_current BOOLEAN DEFAULT TRUE,

    -- Audit columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for current records
CREATE INDEX idx_aircraft_current ON dim_aircraft(aircraft_natural_key)
WHERE is_current = TRUE;
```

### Partitioning Strategy

**Partition by Year** (events table):

```sql
-- Create partitioned events table
CREATE TABLE events (
    ev_id VARCHAR(20) PRIMARY KEY,
    ev_date DATE NOT NULL,
    ev_year INTEGER GENERATED ALWAYS AS (EXTRACT(YEAR FROM ev_date)) STORED,
    -- ... other columns
) PARTITION BY RANGE (ev_year);

-- Create partitions for each decade
CREATE TABLE events_1960s PARTITION OF events
    FOR VALUES FROM (1960) TO (1970);

CREATE TABLE events_1970s PARTITION OF events
    FOR VALUES FROM (1970) TO (1980);

-- ... up to current decade

CREATE TABLE events_2020s PARTITION OF events
    FOR VALUES FROM (2020) TO (2030);
```

**Benefits**:
- **Query Performance**: 10-20x faster for time-range queries
- **Maintenance**: Drop old partitions instead of DELETE (instant)
- **Archival**: Easily move old partitions to slower storage
- **Parallel Processing**: Queries scan only relevant partitions

**Partition Pruning**: PostgreSQL automatically eliminates irrelevant partitions

```sql
-- Query only scans events_2020s partition
EXPLAIN SELECT * FROM events WHERE ev_year = 2022;
-- Result: Seq Scan on events_2020s (no scan on other partitions)
```

### Indexing Strategy

**Primary Indexes**:
```sql
-- B-tree indexes for exact matches and ranges
CREATE INDEX idx_events_ev_id ON events(ev_id);
CREATE INDEX idx_events_ev_date ON events(ev_date);
CREATE INDEX idx_events_severity ON events(ev_highest_injury);
CREATE INDEX idx_aircraft_key ON aircraft(Aircraft_Key);

-- Composite index for common query patterns
CREATE INDEX idx_events_date_severity ON events(ev_date, ev_highest_injury);
CREATE INDEX idx_aircraft_make_model ON aircraft(acft_make, acft_model);

-- GiST index for geospatial queries (PostGIS)
CREATE INDEX idx_events_location ON events USING GIST(
    ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)
);

-- GIN index for full-text search on narratives
CREATE INDEX idx_narratives_fts ON narratives USING GIN(
    to_tsvector('english', narr_accp)
);

-- Partial index for frequent queries (active accidents only)
CREATE INDEX idx_events_active ON events(ev_id)
WHERE report_status = 'FINL';
```

**Index Maintenance**:
```sql
-- Weekly VACUUM ANALYZE (scheduled via cron/Airflow)
VACUUM ANALYZE events;

-- Monthly REINDEX for heavily updated tables
REINDEX TABLE events;

-- Monitor index usage (drop unused indexes)
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Aggregate Tables for Performance

**Pre-aggregated materialized views** for common queries:

```sql
-- Yearly accident statistics
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT
    ev_year,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(inj_tot_f) as total_fatalities,
    AVG(inj_tot_f) as avg_fatalities_per_accident
FROM events
GROUP BY ev_year
ORDER BY ev_year;

-- Refresh monthly via Airflow
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;

-- State-level heatmap data
CREATE MATERIALIZED VIEW mv_state_heatmap AS
SELECT
    ev_state,
    COUNT(*) as accident_count,
    ST_Centroid(ST_Collect(
        ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)
    )) as centroid
FROM events
WHERE ev_state IS NOT NULL
GROUP BY ev_state;

CREATE INDEX idx_state_heatmap_state ON mv_state_heatmap(ev_state);
```

**Performance Gain**: Materialized views reduce query time from 5s to <50ms

## Analytics Layer

### DuckDB for Fast Analytics

**Why DuckDB?**
- **20x faster** than PostgreSQL for OLAP queries (aggregations, joins)
- **In-process**: No network overhead (embedded database)
- **Columnar storage**: Efficient for analytics workloads
- **Parquet native**: Direct queries on Parquet files

**Integration Pattern**:

```python
import duckdb
import psycopg2

# Connect to both PostgreSQL and DuckDB
pg_conn = psycopg2.connect("postgresql://user:pass@localhost/ntsb")
duck_conn = duckdb.connect("analytics.duckdb")

# Install and load postgres_scan extension
duck_conn.execute("INSTALL postgres_scan")
duck_conn.execute("LOAD postgres_scan")

# Query PostgreSQL directly from DuckDB (zero-copy)
duck_conn.execute("""
    ATTACH 'postgresql://user:pass@localhost/ntsb' AS pg (TYPE POSTGRES);

    -- Fast analytics query
    SELECT
        ev_year,
        ev_highest_injury,
        COUNT(*) as count,
        AVG(inj_tot_f) as avg_fatalities
    FROM pg.events
    WHERE ev_year >= 2010
    GROUP BY ev_year, ev_highest_injury
    ORDER BY ev_year, ev_highest_injury;
""")

# Export to Parquet for archival
duck_conn.execute("""
    COPY (SELECT * FROM pg.events WHERE ev_year < 2000)
    TO 'archive/events_pre2000.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
""")
```

**Performance Comparison** (query: aggregate 100K events by year and severity):
- PostgreSQL: 4.2 seconds
- DuckDB (from PostgreSQL): 0.8 seconds
- DuckDB (from Parquet): 0.2 seconds

### Caching Strategy

**Multi-level caching** for dashboard queries:

```mermaid
flowchart LR
    USER[User Request]
    APP_CACHE[Application Cache<br/>Python functools.lru_cache<br/>TTL: 5 minutes]
    REDIS_CACHE[Redis Cache<br/>Query Results<br/>TTL: 1 hour]
    DUCK[DuckDB Analytics]
    PG[(PostgreSQL)]

    USER --> APP_CACHE
    APP_CACHE -->|Miss| REDIS_CACHE
    REDIS_CACHE -->|Miss| DUCK
    DUCK --> PG

    APP_CACHE -->|Hit| USER
    REDIS_CACHE -->|Hit| USER
```

**Redis Caching Example**:

```python
import redis
import json
import hashlib
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_query(ttl=3600):
    """
    Decorator to cache query results in Redis.

    Args:
        ttl: Time to live in seconds (default 1 hour)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = hashlib.md5(
                f"{func.__name__}:{args}:{kwargs}".encode()
            ).hexdigest()

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Execute query
            result = func(*args, **kwargs)

            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )

            return result
        return wrapper
    return decorator

# Usage
@cache_query(ttl=3600)
def get_yearly_stats(start_year, end_year):
    """Fetch yearly statistics (cached for 1 hour)"""
    return duck_conn.execute(f"""
        SELECT ev_year, COUNT(*) as count
        FROM events
        WHERE ev_year BETWEEN {start_year} AND {end_year}
        GROUP BY ev_year
    """).fetchdf()
```

**Cache Invalidation**: Invalidate on new data load (Airflow DAG triggers cache flush)

### Query Optimization Techniques

**1. Predicate Pushdown**: Filter early in query execution

```sql
-- Bad: Filter after join
SELECT e.ev_id, a.acft_make
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year = 2022;

-- Good: Filter before join (PostgreSQL planner optimizes this)
SELECT e.ev_id, a.acft_make
FROM (SELECT * FROM events WHERE ev_year = 2022) e
JOIN aircraft a ON e.ev_id = a.ev_id;
```

**2. Columnar Projection**: Select only needed columns

```sql
-- Bad: SELECT *
SELECT * FROM events WHERE ev_year = 2022;

-- Good: Select only needed columns
SELECT ev_id, ev_date, ev_city, ev_state FROM events WHERE ev_year = 2022;
```

**3. Efficient Joins**: Use EXPLAIN ANALYZE to verify join order

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT e.ev_id, a.acft_make, f.finding_description
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
JOIN Findings f ON e.ev_id = f.ev_id
WHERE e.ev_year = 2022 AND f.cm_inPC = TRUE;
```

## ML Model Serving Infrastructure

### MLflow Architecture

```mermaid
graph TB
    subgraph "Training Environment"
        NOTEBOOK[Jupyter Notebooks<br/>Experiment Development]
        TRAINING[Training Scripts<br/>Python/PyTorch]
    end

    subgraph "MLflow Server"
        TRACKING[Tracking Server<br/>Experiments + Metrics]
        REGISTRY[Model Registry<br/>Versioning + Staging]
        BACKEND[(PostgreSQL Backend<br/>Experiment Metadata)]
        ARTIFACTS[Artifact Store<br/>S3/MinIO<br/>Models + Plots]
    end

    subgraph "Serving Infrastructure"
        API[FastAPI Serving<br/>REST Endpoints]
        MODEL_CACHE[Model Cache<br/>In-Memory Models]
        BATCH[Batch Inference<br/>Airflow DAG]
    end

    subgraph "Monitoring"
        DRIFT[Data Drift Detection<br/>Evidently AI]
        PERF[Performance Monitoring<br/>Latency + Accuracy]
    end

    NOTEBOOK --> TRACKING
    TRAINING --> TRACKING
    TRACKING --> BACKEND
    TRACKING --> ARTIFACTS

    REGISTRY --> ARTIFACTS
    REGISTRY --> API
    REGISTRY --> BATCH

    API --> MODEL_CACHE
    API --> DRIFT
    API --> PERF

    BATCH --> DRIFT
```

### Model Registry Workflow

**Stages**: `None` → `Staging` → `Production` → `Archived`

```python
import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow
mlflow.set_tracking_uri("postgresql://user:pass@localhost/mlflow")
mlflow.set_experiment("accident_severity_prediction")

client = MlflowClient()

# Register model
with mlflow.start_run(run_name="xgboost_v2.3"):
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=1000)

    # Log parameters, metrics, and model
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": 0.912,
        "f1_score": 0.895,
        "roc_auc": 0.948
    })
    mlflow.xgboost.log_model(model, "model")

    # Register to model registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(model_uri, "AccidentSeverityClassifier")

    # Transition to Staging for testing
    client.transition_model_version_stage(
        name="AccidentSeverityClassifier",
        version=mv.version,
        stage="Staging"
    )

# After validation, promote to Production
client.transition_model_version_stage(
    name="AccidentSeverityClassifier",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True  # Archive old production models
)
```

### Batch Inference Pipeline

**Airflow DAG** for weekly batch predictions:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email': ['alerts@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'batch_severity_prediction',
    default_args=default_args,
    schedule_interval='0 2 * * 0',  # 2 AM every Sunday
    catchup=False
)

def load_model():
    """Load production model from MLflow registry"""
    model_uri = "models:/AccidentSeverityClassifier/Production"
    return mlflow.pyfunc.load_model(model_uri)

def extract_features(**context):
    """Extract features for unpredicted accidents"""
    import pandas as pd
    import psycopg2

    conn = psycopg2.connect("postgresql://user:pass@localhost/ntsb")

    # Get accidents without predictions
    query = """
        SELECT e.ev_id, e.ev_date, e.dec_latitude, e.dec_longitude,
               a.acft_make, a.acft_model, fc.pilot_tot_time,
               e.wx_cond_basic, e.wx_temp
        FROM events e
        LEFT JOIN predictions p ON e.ev_id = p.ev_id
        JOIN aircraft a ON e.ev_id = a.ev_id
        LEFT JOIN Flight_Crew fc ON a.Aircraft_Key = fc.Aircraft_Key
        WHERE p.ev_id IS NULL AND e.ev_date >= CURRENT_DATE - INTERVAL '7 days'
    """

    df = pd.read_sql(query, conn)
    context['ti'].xcom_push(key='features', value=df.to_json())
    conn.close()

def predict_severity(**context):
    """Generate predictions"""
    import pandas as pd

    # Load features from XCom
    features_json = context['ti'].xcom_pull(key='features', task_ids='extract_features')
    df = pd.read_json(features_json)

    if df.empty:
        print("No new accidents to predict")
        return

    # Load model
    model = load_model()

    # Feature engineering
    X = feature_engineering(df)  # Defined elsewhere

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Prepare results
    results = df[['ev_id']].copy()
    results['predicted_severity'] = predictions
    results['confidence'] = probabilities.max(axis=1)
    results['predicted_at'] = datetime.utcnow()

    context['ti'].xcom_push(key='predictions', value=results.to_json())

def store_predictions(**context):
    """Store predictions in database"""
    import pandas as pd
    import psycopg2

    predictions_json = context['ti'].xcom_pull(key='predictions', task_ids='predict_severity')
    if not predictions_json:
        return

    df = pd.read_json(predictions_json)

    conn = psycopg2.connect("postgresql://user:pass@localhost/ntsb")
    cursor = conn.cursor()

    # Bulk insert
    from psycopg2.extras import execute_values
    execute_values(
        cursor,
        """
        INSERT INTO predictions (ev_id, predicted_severity, confidence, predicted_at, model_version)
        VALUES %s
        """,
        [(row.ev_id, row.predicted_severity, row.confidence, row.predicted_at, 'v2.3')
         for row in df.itertuples()]
    )

    conn.commit()
    cursor.close()
    conn.close()

# Define tasks
extract_task = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features,
    dag=dag
)

predict_task = PythonOperator(
    task_id='predict_severity',
    python_callable=predict_severity,
    dag=dag
)

store_task = PythonOperator(
    task_id='store_predictions',
    python_callable=store_predictions,
    dag=dag
)

# Set dependencies
extract_task >> predict_task >> store_task
```

### Real-Time Inference API

**FastAPI endpoint** for real-time predictions:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import mlflow
import numpy as np
from typing import List, Optional
from functools import lru_cache

app = FastAPI(title="NTSB ML Prediction API", version="1.0.0")

# Pydantic models
class AccidentFeatures(BaseModel):
    ev_date: str = Field(..., example="2024-01-15")
    dec_latitude: float = Field(..., ge=-90, le=90)
    dec_longitude: float = Field(..., ge=-180, le=180)
    acft_make: str = Field(..., example="Cessna")
    acft_model: str = Field(..., example="172")
    pilot_tot_time: Optional[int] = Field(None, ge=0)
    wx_cond_basic: str = Field(..., example="VMC")
    wx_temp: Optional[int] = None

class PredictionResponse(BaseModel):
    ev_id: Optional[str] = None
    predicted_severity: str
    confidence: float
    probabilities: dict
    model_version: str

@lru_cache(maxsize=1)
def load_production_model():
    """Load production model (cached)"""
    model_uri = "models:/AccidentSeverityClassifier/Production"
    return mlflow.pyfunc.load_model(model_uri)

@app.post("/predict/severity", response_model=PredictionResponse)
async def predict_severity(features: AccidentFeatures):
    """
    Predict accident severity from input features.

    Returns:
        - predicted_severity: Fatal/Serious/Minor/None
        - confidence: Probability of predicted class (0-1)
        - probabilities: All class probabilities
    """
    try:
        # Load model
        model = load_production_model()

        # Feature engineering
        X = np.array([[
            features.dec_latitude,
            features.dec_longitude,
            1 if features.wx_cond_basic == "IMC" else 0,
            features.pilot_tot_time or 0,
            features.wx_temp or 15
            # ... more features
        ]])

        # Predict
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]

        severity_map = {0: "Fatal", 1: "Serious", 2: "Minor", 3: "None"}

        return PredictionResponse(
            predicted_severity=severity_map[prediction],
            confidence=float(probas[prediction]),
            probabilities={
                severity_map[i]: float(prob)
                for i, prob in enumerate(probas)
            },
            model_version="v2.3"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """Get current production model metadata"""
    client = mlflow.tracking.MlflowClient()
    model = client.get_latest_versions("AccidentSeverityClassifier", stages=["Production"])[0]

    return {
        "name": model.name,
        "version": model.version,
        "stage": model.current_stage,
        "created_at": model.creation_timestamp,
        "metrics": client.get_run(model.run_id).data.metrics
    }
```

### Model Monitoring

**Data Drift Detection** with Evidently AI:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def detect_data_drift(reference_data, current_data):
    """
    Detect data drift between training data and production data.

    Args:
        reference_data: Training data features
        current_data: Recent production features (last 7 days)

    Returns:
        Report with drift metrics
    """
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=reference_data, current_data=current_data)

    # Save report
    report.save_html("reports/data_drift.html")

    # Check if drift detected
    drift_metrics = report.as_dict()
    dataset_drift = drift_metrics['metrics'][0]['result']['dataset_drift']

    if dataset_drift:
        # Send alert
        send_alert(
            title="Data Drift Detected",
            message=f"Drift detected in {drift_metrics['metrics'][0]['result']['number_of_drifted_columns']} features",
            severity="warning"
        )

    return drift_metrics

# Schedule in Airflow DAG (weekly)
```

### Feature Store Design

**Centralized feature storage** for consistency:

```python
import feast
from feast import FeatureStore, Entity, Feature, FeatureView, FileSource
from datetime import timedelta

# Define entities
accident_entity = Entity(
    name="accident",
    value_type=ValueType.STRING,
    description="Accident event ID"
)

# Define feature view
accident_features = FeatureView(
    name="accident_features",
    entities=["accident"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="dec_latitude", dtype=ValueType.FLOAT),
        Feature(name="dec_longitude", dtype=ValueType.FLOAT),
        Feature(name="pilot_tot_time", dtype=ValueType.INT64),
        Feature(name="is_imc", dtype=ValueType.BOOL),
        Feature(name="power_to_weight_ratio", dtype=ValueType.FLOAT),
    ],
    online=True,
    source=FileSource(
        path="features/accident_features.parquet",
        event_timestamp_column="event_timestamp"
    )
)

# Initialize feature store
store = FeatureStore(repo_path="feature_repo/")

# Get features for online serving
features = store.get_online_features(
    features=[
        "accident_features:dec_latitude",
        "accident_features:dec_longitude",
        "accident_features:pilot_tot_time",
    ],
    entity_rows=[{"accident": "20220101001234"}]
).to_dict()
```

## API Architecture

### REST API Design

**Endpoint Structure**:

```
/api/v1/
├── /events
│   ├── GET /                     # List accidents (paginated)
│   ├── GET /{ev_id}              # Single accident details
│   ├── GET /{ev_id}/aircraft     # Aircraft involved
│   ├── GET /{ev_id}/crew         # Crew details
│   └── GET /{ev_id}/findings     # Investigation findings
├── /search
│   ├── GET /                     # Full-text search
│   └── POST /advanced            # Complex queries
├── /stats
│   ├── GET /summary              # Overall statistics
│   ├── GET /yearly               # Yearly trends
│   ├── GET /by-state             # State-level aggregates
│   └── GET /by-aircraft-type     # Aircraft type statistics
├── /predictions
│   ├── POST /severity            # Predict severity
│   ├── POST /causes              # Predict probable causes
│   └── POST /batch               # Batch predictions
├── /analytics
│   ├── GET /time-series          # Time series data
│   ├── GET /geospatial           # Geospatial clusters
│   └── GET /correlations         # Correlation analysis
└── /health
    ├── GET /                     # Health check
    ├── GET /ready                # Readiness probe
    └── GET /metrics              # Prometheus metrics
```

### Authentication & Authorization

**JWT-based authentication**:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

SECRET_KEY = "your-secret-key"  # Load from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    """Generate JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Usage in endpoint
@app.get("/api/v1/events/{ev_id}")
async def get_event(ev_id: str, token: dict = Depends(verify_token)):
    """Protected endpoint requiring authentication"""
    # token contains user_id and other claims
    pass
```

### Rate Limiting

**Redis-based sliding window rate limiter**:

```python
import redis
from fastapi import Request, HTTPException
from datetime import datetime
import time

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def __call__(self, request: Request):
        """
        Sliding window rate limiter.

        Free tier: 100 requests/minute
        Premium tier: 1000 requests/minute
        """
        # Get user identifier (IP or user_id from token)
        client_id = request.client.host

        # Check if user has API key (premium tier)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Verify API key and get tier
            tier = get_user_tier(api_key)  # Defined elsewhere
            if tier == "premium":
                self.max_requests = 1000

        # Redis key for this client
        key = f"rate_limit:{client_id}"

        # Current timestamp
        now = time.time()

        # Add current request
        pipe = redis_client.pipeline()
        pipe.zadd(key, {str(now): now})

        # Remove requests outside window
        pipe.zremrangebyscore(key, 0, now - self.window_seconds)

        # Count requests in window
        pipe.zcard(key)

        # Set expiry
        pipe.expire(key, self.window_seconds)

        results = pipe.execute()
        request_count = results[2]

        # Check if limit exceeded
        if request_count > self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds} seconds"
            )

# Apply to all routes
app.add_middleware(RateLimiter(max_requests=100, window_seconds=60))
```

### API Versioning

**URL-based versioning** (recommended):

```python
from fastapi import APIRouter

# Version 1 router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/events")
async def list_events_v1():
    """Version 1 API"""
    return {"version": "1.0", "events": [...]}

# Version 2 router (with breaking changes)
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/events")
async def list_events_v2():
    """Version 2 API with enhanced response format"""
    return {
        "version": "2.0",
        "data": {"events": [...]},
        "meta": {"total": 100, "page": 1}
    }

# Include both routers
app.include_router(v1_router)
app.include_router(v2_router)
```

### OpenAPI Documentation

**Auto-generated Swagger UI**:

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="NTSB Aviation Accident Analysis API",
    description="""
    Production API for NTSB aviation accident database analysis.

    ## Features
    - 100K+ accident records (1962-present)
    - Machine learning predictions (90%+ accuracy)
    - Real-time analytics
    - Geospatial queries

    ## Rate Limits
    - **Free**: 100 requests/minute
    - **Premium**: 1000 requests/minute

    ## Authentication
    All endpoints require JWT token via `Authorization: Bearer <token>` header.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "api@ntsb-analysis.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Access docs at /docs (Swagger UI) or /redoc (ReDoc)
```

### SDK Generation

**Auto-generate client SDKs** from OpenAPI spec:

```bash
# Generate Python SDK
openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g python \
    -o sdk/python \
    --package-name ntsb_api_client

# Generate JavaScript SDK
openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g javascript \
    -o sdk/javascript

# Generate R SDK
openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g r \
    -o sdk/r
```

**Usage of generated Python SDK**:

```python
import ntsb_api_client
from ntsb_api_client.api import events_api
from ntsb_api_client.model.accident_features import AccidentFeatures

# Configure API client
configuration = ntsb_api_client.Configuration(
    host="https://api.ntsb-analysis.com",
    access_token="your-jwt-token"
)

# Create API instance
with ntsb_api_client.ApiClient(configuration) as api_client:
    api_instance = events_api.EventsApi(api_client)

    # Get event details
    ev_id = "20220101001234"
    event = api_instance.get_event(ev_id)
    print(event)

    # Predict severity
    features = AccidentFeatures(
        ev_date="2024-01-15",
        dec_latitude=40.7128,
        dec_longitude=-74.0060,
        acft_make="Cessna",
        acft_model="172",
        pilot_tot_time=500,
        wx_cond_basic="VMC"
    )
    prediction = api_instance.predict_severity(features)
    print(f"Predicted severity: {prediction.predicted_severity}")
```

## Dashboard & UI Architecture

### Technology Comparison: Plotly Dash vs Streamlit

| Feature | Plotly Dash | Streamlit | Recommendation |
|---------|-------------|-----------|----------------|
| **Performance** | Good (React-based) | Excellent (pure Python) | **Streamlit** for rapid development |
| **Customization** | High (full React control) | Medium (limited HTML/CSS) | **Dash** for custom branding |
| **Learning Curve** | Steep (requires HTML/CSS) | Gentle (pure Python) | **Streamlit** for data scientists |
| **Interactivity** | Excellent (callbacks) | Good (widgets) | **Dash** for complex interactions |
| **Deployment** | Complex | Simple | **Streamlit** for quick deploy |
| **Scalability** | High (production-ready) | Medium (add gunicorn) | **Dash** for >10K users |
| **Cost** | Open source | Open source | Both free |

**Decision**: Use **Streamlit** for Phase 1-3 (rapid prototyping), migrate to **Dash** for Phase 5 (production scale)

### Streamlit Dashboard Architecture

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="NTSB Aviation Accident Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar filters
st.sidebar.title("Filters")

year_range = st.sidebar.slider(
    "Year Range",
    min_value=1962,
    max_value=datetime.now().year,
    value=(2010, datetime.now().year)
)

severity_filter = st.sidebar.multiselect(
    "Severity",
    options=["Fatal", "Serious", "Minor", "None"],
    default=["Fatal", "Serious"]
)

state_filter = st.sidebar.multiselect(
    "State",
    options=["CA", "TX", "FL", "NY", "AK"],  # Top 5
    default=[]
)

# Main dashboard
st.title("NTSB Aviation Accident Analysis Dashboard")
st.markdown("Real-time analytics on 100K+ aviation accidents (1962-present)")

# KPI metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Accidents",
        value="103,457",
        delta="+23 this month"
    )

with col2:
    st.metric(
        label="Fatal Accidents",
        value="12,843",
        delta="-5% vs last year",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Avg Response Time",
        value="78ms",
        delta="-12ms"
    )

with col4:
    st.metric(
        label="Model Accuracy",
        value="91.2%",
        delta="+1.3%"
    )

# Charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Geographic", "✈️ Aircraft Types", "🤖 ML Predictions"])

with tab1:
    st.subheader("Accident Trends Over Time")

    # Fetch data from API
    @st.cache_data(ttl=3600)
    def fetch_yearly_stats(start_year, end_year):
        response = requests.get(
            f"http://api:8000/api/v1/stats/yearly?start_year={start_year}&end_year={end_year}",
            headers={"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}
        )
        return pd.DataFrame(response.json())

    df = fetch_yearly_stats(year_range[0], year_range[1])

    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['ev_year'],
        y=df['total_accidents'],
        mode='lines+markers',
        name='Total Accidents',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['ev_year'],
        y=df['fatal_accidents'],
        mode='lines+markers',
        name='Fatal Accidents',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title='Yearly Accident Trends',
        xaxis_title='Year',
        yaxis_title='Number of Accidents',
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Geospatial Distribution")

    @st.cache_data(ttl=3600)
    def fetch_geospatial_data():
        response = requests.get(
            "http://api:8000/api/v1/analytics/geospatial",
            headers={"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}
        )
        return pd.DataFrame(response.json())

    geo_df = fetch_geospatial_data()

    # Scatter map
    fig = px.scatter_mapbox(
        geo_df,
        lat="dec_latitude",
        lon="dec_longitude",
        color="severity",
        size="fatalities",
        hover_data=["ev_id", "ev_date", "acft_make"],
        title="Accident Locations",
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 39.8283, "lon": -98.5795}  # US center
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Aircraft Type Analysis")

    @st.cache_data(ttl=3600)
    def fetch_aircraft_stats():
        response = requests.get(
            "http://api:8000/api/v1/stats/by-aircraft-type",
            headers={"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}
        )
        return pd.DataFrame(response.json())

    aircraft_df = fetch_aircraft_stats()

    # Bar chart
    fig = px.bar(
        aircraft_df.head(20),
        x='acft_make',
        y='accident_count',
        color='fatal_rate',
        title='Top 20 Aircraft Makes by Accident Count',
        labels={'acft_make': 'Aircraft Make', 'accident_count': 'Accidents', 'fatal_rate': 'Fatal Rate (%)'},
        color_continuous_scale='Reds'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("ML Model Predictions")

    # Input form for prediction
    st.markdown("### Predict Accident Severity")

    col1, col2 = st.columns(2)

    with col1:
        ev_date = st.date_input("Accident Date", value=datetime.now())
        dec_latitude = st.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0)
        dec_longitude = st.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0)

    with col2:
        acft_make = st.selectbox("Aircraft Make", options=["Cessna", "Piper", "Beech", "Cirrus", "Boeing"])
        acft_model = st.text_input("Aircraft Model", value="172")
        pilot_tot_time = st.number_input("Pilot Total Hours", value=500, min_value=0)

    wx_cond_basic = st.radio("Weather Conditions", options=["VMC", "IMC"])

    if st.button("Predict Severity"):
        # Call prediction API
        payload = {
            "ev_date": str(ev_date),
            "dec_latitude": dec_latitude,
            "dec_longitude": dec_longitude,
            "acft_make": acft_make,
            "acft_model": acft_model,
            "pilot_tot_time": pilot_tot_time,
            "wx_cond_basic": wx_cond_basic
        }

        response = requests.post(
            "http://api:8000/api/v1/predictions/severity",
            json=payload,
            headers={"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}
        )

        if response.status_code == 200:
            result = response.json()

            # Display prediction
            st.success(f"**Predicted Severity**: {result['predicted_severity']}")
            st.info(f"**Confidence**: {result['confidence']:.2%}")

            # Probability bar chart
            prob_df = pd.DataFrame.from_dict(result['probabilities'], orient='index', columns=['Probability'])
            fig = px.bar(prob_df, x=prob_df.index, y='Probability', title='Class Probabilities')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.text}")

# Footer
st.markdown("---")
st.markdown("Data source: NTSB Aviation Accident Database | Updated monthly | API v1.0")
```

### Real-Time Data Updates

**WebSocket integration** for live dashboard updates:

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Sends notifications when:
    - New accident added
    - Prediction completed
    - Data quality alert
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Echo for heartbeat
            await websocket.send_text(f"Heartbeat: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Trigger broadcast when new data arrives (in Airflow DAG)
def notify_dashboard_update(event_type: str, data: dict):
    """Called by Airflow DAG after data load"""
    import asyncio

    message = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    # Broadcast to all connected clients
    asyncio.run(manager.broadcast(message))
```

**Streamlit WebSocket client**:

```python
import streamlit as st
import asyncio
import websockets
import json

async def listen_to_updates():
    """Listen for real-time updates"""
    uri = "ws://api:8000/ws/live-updates"

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            # Update dashboard state
            st.session_state['last_update'] = data
            st.rerun()  # Trigger dashboard refresh

# Run WebSocket listener in background
if 'websocket_started' not in st.session_state:
    asyncio.run(listen_to_updates())
    st.session_state['websocket_started'] = True
```

### Mobile-Responsive Design

**CSS customization** for mobile views:

```python
# Inject custom CSS
st.markdown("""
<style>
    /* Mobile-responsive layout */
    @media (max-width: 768px) {
        .stColumn {
            flex-direction: column !important;
        }

        .stMetric {
            font-size: 0.9rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
        }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
    }
</style>
""", unsafe_allow_html=True)
```

### User Authentication & Authorization

**Role-Based Access Control (RBAC)**:

```python
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"           # Full access
    ANALYST = "analyst"       # Read + ML predictions
    VIEWER = "viewer"         # Read only
    API_USER = "api_user"     # API access only

def check_permission(required_role: UserRole):
    """Decorator to check user permissions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')

            if not user:
                raise HTTPException(status_code=401, detail="Not authenticated")

            # Check role hierarchy
            role_hierarchy = {
                UserRole.ADMIN: 4,
                UserRole.ANALYST: 3,
                UserRole.API_USER: 2,
                UserRole.VIEWER: 1
            }

            if role_hierarchy[user.role] < role_hierarchy[required_role]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@app.post("/api/v1/predictions/severity")
@check_permission(UserRole.ANALYST)
async def predict_severity(features: AccidentFeatures, current_user: User = Depends(get_current_user)):
    """Only analysts and admins can access predictions"""
    pass
```

## Scalability Considerations

### Horizontal Scaling with Kubernetes

**Auto-scaling based on CPU/memory** (Horizontal Pod Autoscaler):

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale when CPU > 70%
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # Scale when memory > 80%
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50  # Add 50% more pods
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1  # Remove 1 pod at a time
        periodSeconds: 120
```

### Database Sharding

**Shard by year range** for large datasets:

```mermaid
graph TB
    APP[Application Layer]
    SHARD_ROUTER[Shard Router<br/>Year-based]

    SHARD1[(PostgreSQL Shard 1<br/>1962-1989)]
    SHARD2[(PostgreSQL Shard 2<br/>1990-2009)]
    SHARD3[(PostgreSQL Shard 3<br/>2010-present)]

    APP --> SHARD_ROUTER
    SHARD_ROUTER --> SHARD1
    SHARD_ROUTER --> SHARD2
    SHARD_ROUTER --> SHARD3
```

**Shard router logic**:

```python
class ShardRouter:
    def __init__(self):
        self.shards = {
            "shard1": {"host": "shard1.db", "year_range": (1962, 1989)},
            "shard2": {"host": "shard2.db", "year_range": (1990, 2009)},
            "shard3": {"host": "shard3.db", "year_range": (2010, 9999)}
        }

    def get_shard(self, year: int) -> str:
        """Determine which shard contains data for given year"""
        for shard_name, config in self.shards.items():
            start, end = config["year_range"]
            if start <= year <= end:
                return config["host"]
        raise ValueError(f"No shard found for year {year}")

    def query(self, sql: str, year: int):
        """Route query to appropriate shard"""
        shard_host = self.get_shard(year)
        conn = psycopg2.connect(f"postgresql://user:pass@{shard_host}/ntsb")
        return pd.read_sql(sql, conn)

# Usage
router = ShardRouter()
df = router.query("SELECT * FROM events WHERE ev_year = 2022", year=2022)
```

### Read Replicas

**Primary-replica topology** for read-heavy workloads:

```mermaid
graph TB
    APP[Application Layer]
    PRIMARY[(PostgreSQL Primary<br/>Read + Write)]
    REPLICA1[(Read Replica 1)]
    REPLICA2[(Read Replica 2)]
    REPLICA3[(Read Replica 3)]

    APP -->|Writes| PRIMARY
    APP -->|Reads| REPLICA1
    APP -->|Reads| REPLICA2
    APP -->|Reads| REPLICA3

    PRIMARY -->|Streaming Replication| REPLICA1
    PRIMARY -->|Streaming Replication| REPLICA2
    PRIMARY -->|Streaming Replication| REPLICA3
```

**Connection pooling with read-write split**:

```python
import psycopg2
from psycopg2 import pool

class DatabasePool:
    def __init__(self):
        # Primary pool (for writes)
        self.primary_pool = pool.SimpleConnectionPool(
            minconn=5,
            maxconn=20,
            host="primary.db",
            database="ntsb",
            user="app",
            password="***"
        )

        # Replica pool (for reads)
        self.replica_hosts = ["replica1.db", "replica2.db", "replica3.db"]
        self.replica_pools = [
            pool.SimpleConnectionPool(
                minconn=10,
                maxconn=50,
                host=host,
                database="ntsb",
                user="app_readonly",
                password="***"
            )
            for host in self.replica_hosts
        ]
        self.replica_index = 0

    def get_write_connection(self):
        """Get connection to primary for writes"""
        return self.primary_pool.getconn()

    def get_read_connection(self):
        """Get connection to replica for reads (round-robin)"""
        pool = self.replica_pools[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replica_pools)
        return pool.getconn()

# Usage
db = DatabasePool()

# Write query
conn = db.get_write_connection()
cursor = conn.cursor()
cursor.execute("INSERT INTO events (...) VALUES (...)")
conn.commit()

# Read query (routed to replica)
conn = db.get_read_connection()
df = pd.read_sql("SELECT * FROM events WHERE ev_year = 2022", conn)
```

### Performance Targets

| Metric | Target | Measurement | Scaling Strategy |
|--------|--------|-------------|------------------|
| API Latency (p50) | <50ms | Prometheus | Increase replicas, add caching |
| API Latency (p95) | <100ms | Prometheus | Optimize queries, add indexes |
| API Latency (p99) | <200ms | Prometheus | Database tuning, connection pooling |
| Dashboard Load Time | <2s | RUM | CDN for static assets, lazy loading |
| Database Query (simple) | <10ms | pg_stat_statements | Indexes, query optimization |
| Database Query (complex) | <100ms | pg_stat_statements | Materialized views, DuckDB |
| ML Inference | <50ms | MLflow | Model quantization, batching |
| Throughput | 1000 req/s | Load testing | Horizontal scaling (HPA) |
| Concurrent Users | 10,000 | Load testing | Read replicas, caching |

### CDN for Static Assets

**Use CloudFlare or AWS CloudFront** for dashboard static files:

```nginx
# nginx.conf for CDN-backed dashboard
server {
    listen 80;
    server_name dashboard.ntsb-analysis.com;

    # CDN for static assets
    location /static/ {
        proxy_pass https://cdn.example.com/static/;
        proxy_cache_valid 200 1d;
        add_header Cache-Control "public, max-age=86400";
    }

    # Application
    location / {
        proxy_pass http://streamlit:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Cloud Deployment Comparison

### Cost Estimates (Monthly)

**Scenario**: 100K events, 1M API requests/month, 1000 concurrent users

| Component | AWS | GCP | Azure | Self-Hosted |
|-----------|-----|-----|-------|-------------|
| **Database** (PostgreSQL) | RDS t3.large: $120 | Cloud SQL db-standard-2: $90 | Azure Database Standard: $100 | Bare metal: $50 |
| **Object Storage** (100GB) | S3: $2.30 | GCS: $2.30 | Blob Storage: $2.20 | MinIO (self-hosted): $0 |
| **Kubernetes** (3 nodes) | EKS + EC2 t3.medium: $200 | GKE e2-standard-2: $180 | AKS Standard: $190 | k3s (self-hosted): $0 |
| **Load Balancer** | ALB: $25 | Cloud Load Balancing: $20 | Application Gateway: $30 | Nginx (self-hosted): $0 |
| **Redis Cache** | ElastiCache t3.micro: $15 | Memorystore Basic: $20 | Azure Cache: $18 | Redis (self-hosted): $0 |
| **Monitoring** | CloudWatch: $30 | Cloud Monitoring: $25 | Azure Monitor: $28 | Prometheus/Grafana: $0 |
| **Backups** (1TB) | S3 + Snapshots: $25 | GCS + Snapshots: $23 | Storage + Snapshots: $26 | Backblaze B2: $5 |
| **Data Transfer** (500GB) | $45 | $50 | $48 | $30 |
| **Total** | **$462/month** | **$410/month** | **$442/month** | **$85/month** + hardware |

**Winner**: **GCP** for cloud deployment, **Self-hosted** for cost optimization

### Feature Comparison

| Feature | AWS | GCP | Azure | Self-Hosted |
|---------|-----|-----|-------|-------------|
| Managed PostgreSQL | ✅ RDS (excellent) | ✅ Cloud SQL (excellent) | ✅ Azure Database (good) | ❌ Manual setup |
| Managed Kubernetes | ✅ EKS (mature) | ✅ GKE (best) | ✅ AKS (good) | ✅ k3s, k8s |
| Serverless Functions | ✅ Lambda | ✅ Cloud Functions | ✅ Azure Functions | ❌ N/A |
| ML Services | ✅ SageMaker (excellent) | ✅ Vertex AI (excellent) | ✅ Azure ML (good) | ❌ Manual MLflow |
| Auto-scaling | ✅ Native | ✅ Native | ✅ Native | ✅ Manual HPA |
| Global CDN | ✅ CloudFront | ✅ Cloud CDN | ✅ Azure CDN | ⚠️ CloudFlare |
| Cost Optimization | ⚠️ Spot instances | ✅ Preemptible VMs | ⚠️ Spot VMs | ✅ Full control |
| Vendor Lock-in | ⚠️ High | ⚠️ High | ⚠️ High | ✅ None |

**Recommendation**:
- **Development**: Self-hosted on bare metal or local k3s cluster
- **Staging**: GCP (best GKE, cheapest)
- **Production**: GCP or hybrid (k8s on bare metal + GCP for managed services)

### Hybrid Cloud Architecture

**Best of both worlds**: Kubernetes on bare metal + cloud-managed services

```mermaid
graph TB
    subgraph "Self-Hosted (Bare Metal)"
        K8S[Kubernetes Cluster<br/>3 nodes @ $150/month]
        PG[(PostgreSQL<br/>Dedicated server)]
        REDIS[Redis Cache]
    end

    subgraph "Cloud (GCP)"
        GCS[Cloud Storage<br/>$2.30/month]
        VERTEX[Vertex AI<br/>Pay-per-use]
        CDN[Cloud CDN<br/>$20/month]
    end

    K8S --> PG
    K8S --> REDIS
    K8S --> GCS
    K8S --> VERTEX
    CDN --> K8S
```

**Total cost**: ~$220/month (61% savings vs full cloud)

## Containerization Strategy

### Docker Multi-Stage Build

**Optimized Dockerfile** for FastAPI application:

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ./app ./app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Image size**: 180MB (vs 800MB without multi-stage)

### Docker Compose for Local Development

**Complete stack** with all services:

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
      POSTGRES_INITDB_ARGS: "-E UTF8 --locale=C"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://app:dev_password@postgres:5432/mlflow
      MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://mlflow/artifacts
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on:
      - postgres
      - minio
    command: mlflow server --host 0.0.0.0 --port 5000

  airflow-webserver:
    image: apache/airflow:2.7.3-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://app:dev_password@postgres:5432/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
      AIRFLOW__WEBSERVER__SECRET_KEY: 'dev_secret_key'
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres
    command: webserver

  airflow-scheduler:
    image: apache/airflow:2.7.3-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://app:dev_password@postgres:5432/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres
      - airflow-webserver
    command: scheduler

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://app:dev_password@postgres:5432/ntsb
      REDIS_URL: redis://redis:6379/0
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - postgres
      - redis
      - mlflow
    volumes:
      - ./app:/app/app  # Mount for hot reload

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      API_URL: http://api:8000
    depends_on:
      - api

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: 'false'
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

**Start all services**: `docker-compose up -d`

### Kubernetes Manifests

**API Deployment**:

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
  namespace: ntsb
  labels:
    app: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: ghcr.io/your-org/ntsb-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: ntsb
spec:
  selector:
    app: api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  namespace: ntsb
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.ntsb-analysis.com
    secretName: api-tls
  rules:
  - host: api.ntsb-analysis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
```

### Helm Charts

**Package for easy deployment**:

```yaml
# helm/ntsb-platform/Chart.yaml
apiVersion: v2
name: ntsb-platform
description: NTSB Aviation Accident Analysis Platform
type: application
version: 1.0.0
appVersion: "1.0.0"

# helm/ntsb-platform/values.yaml
replicaCount: 3

image:
  repository: ghcr.io/your-org/ntsb-api
  pullPolicy: IfNotPresent
  tag: "v1.0.0"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.ntsb-analysis.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: api-tls
      hosts:
        - api.ntsb-analysis.com

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    username: app
    password: changeme
    database: ntsb
  primary:
    persistence:
      size: 100Gi

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: false

# Deploy with Helm
# helm install ntsb-platform ./helm/ntsb-platform
```

### Container Registry

**GitHub Container Registry** (free for public repos):

```bash
# Build and push
docker build -t ghcr.io/your-org/ntsb-api:v1.0.0 .
docker push ghcr.io/your-org/ntsb-api:v1.0.0

# Tag as latest
docker tag ghcr.io/your-org/ntsb-api:v1.0.0 ghcr.io/your-org/ntsb-api:latest
docker push ghcr.io/your-org/ntsb-api:latest
```

## Monitoring & Observability

### Prometheus Metrics

**Expose custom metrics** from FastAPI:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total ML predictions',
    ['model', 'predicted_class']
)

ml_prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'ML prediction confidence scores',
    ['model']
)

database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['query_type']
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type']
)

# Middleware to record metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Record request metrics
    duration = time.time() - start_time
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    api_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response

# Metrics endpoint for Prometheus scraping
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Example: Record prediction metrics
@app.post("/predict/severity")
async def predict_severity(features: AccidentFeatures):
    # ... prediction logic

    # Record metrics
    ml_predictions_total.labels(
        model="AccidentSeverityClassifier",
        predicted_class=result['predicted_severity']
    ).inc()

    ml_prediction_confidence.labels(
        model="AccidentSeverityClassifier"
    ).observe(result['confidence'])

    return result
```

### Grafana Dashboards

**Pre-built dashboard** for system monitoring:

```json
{
  "dashboard": {
    "title": "NTSB Platform Monitoring",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "API Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "ML Predictions/sec",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[1m])",
            "legendFormat": "{{predicted_class}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Database Query Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m]))",
            "legendFormat": "{{query_type}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(api_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

**Access**: http://localhost:3000 (default credentials: admin/admin)

### Logging with Loki

**Structured JSON logging**:

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)

# Usage
logger.info("API request received", extra={
    'user_id': 'user123',
    'request_id': 'req-abc-123',
    'endpoint': '/api/v1/events'
})
```

**Loki configuration** (`loki-config.yaml`):

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

### Alerting with AlertManager

**Alert rules** (`alerts.yml`):

```yaml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate detected"
          description: "API error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "API p95 latency is {{ $value | humanizeDuration }} (threshold: 200ms)"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL instance {{ $labels.instance }} is unreachable"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{pod=~"api-.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in API pod"
          description: "Pod {{ $labels.pod }} is using {{ $value | humanizePercentage }} of memory limit"

  - name: ml_alerts
    interval: 1m
    rules:
      - alert: LowPredictionConfidence
        expr: avg(ml_prediction_confidence) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "ML model confidence degraded"
          description: "Average prediction confidence is {{ $value | humanizePercentage }} (threshold: 70%)"

      - alert: DataDriftDetected
        expr: data_drift_detected == 1
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Significant drift detected in production data distribution"
```

**AlertManager integration** with Slack/PagerDuty:

```yaml
# alertmanager.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: "{{ .GroupLabels.alertname }}"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"

  - name: 'slack'
    slack_configs:
      - channel: '#alerts-warning'
        title: "Warning: {{ .GroupLabels.alertname }}"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
        color: 'warning'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: "{{ .GroupLabels.alertname }}"
```

### Distributed Tracing with Jaeger

**OpenTelemetry integration**:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument SQLAlchemy
SQLAlchemyInstrumentor().instrument()

# Manual tracing example
@app.post("/predict/severity")
async def predict_severity(features: AccidentFeatures):
    with tracer.start_as_current_span("predict_severity") as span:
        span.set_attribute("model", "AccidentSeverityClassifier")
        span.set_attribute("input.acft_make", features.acft_make)

        with tracer.start_as_current_span("feature_engineering"):
            X = feature_engineering(features)

        with tracer.start_as_current_span("model_inference"):
            prediction = model.predict(X)
            span.set_attribute("output.severity", prediction[0])

        return {"predicted_severity": prediction[0]}
```

**View traces**: http://localhost:16686 (Jaeger UI)

### SLOs & SLIs

**Service Level Objectives**:

| SLO | Target | Measurement |
|-----|--------|-------------|
| **Availability** | 99.9% (43 min downtime/month) | `sum(up{job="api"}) / count(up{job="api"})` |
| **Latency** | 95% of requests <100ms | `histogram_quantile(0.95, api_request_duration_seconds)` |
| **Error Rate** | <1% of requests | `rate(api_requests_total{status=~"5.."}[5m])` |
| **Throughput** | >1000 requests/sec | `rate(api_requests_total[1m])` |

**SLI Dashboard** (calculated monthly):

```promql
# Availability SLI (percentage of time service is up)
avg_over_time(up{job="api"}[30d]) * 100

# Latency SLI (percentage of requests under 100ms)
(
  sum(rate(api_request_duration_seconds_bucket{le="0.1"}[30d]))
  /
  sum(rate(api_request_duration_seconds_count[30d]))
) * 100

# Error Rate SLI (percentage of non-error requests)
(
  1 - (
    sum(rate(api_requests_total{status=~"5.."}[30d]))
    /
    sum(rate(api_requests_total[30d]))
  )
) * 100
```

## Disaster Recovery & Backup

### PostgreSQL Backup Strategy

**Daily full backups + hourly incremental**:

```bash
#!/bin/bash
# backup.sh - PostgreSQL backup script

# Configuration
DB_NAME="ntsb"
DB_USER="app"
BACKUP_DIR="/backups/postgresql"
RETENTION_DAYS=30
S3_BUCKET="s3://ntsb-backups/postgres"

DATE=$(date +%Y%m%d_%H%M%S)

# Full backup (daily at 2 AM)
pg_dump -U $DB_USER -F c -b -v -f "$BACKUP_DIR/full_$DATE.backup" $DB_NAME

# Compress backup
gzip "$BACKUP_DIR/full_$DATE.backup"

# Upload to S3
aws s3 cp "$BACKUP_DIR/full_$DATE.backup.gz" "$S3_BUCKET/full/"

# Remove old backups
find "$BACKUP_DIR" -name "full_*.backup.gz" -mtime +$RETENTION_DAYS -delete

# WAL archiving for point-in-time recovery (continuous)
# Configured in postgresql.conf:
# archive_mode = on
# archive_command = 'aws s3 cp %p s3://ntsb-backups/wal/%f'
```

**Cron schedule**:

```cron
# Full backup daily at 2 AM
0 2 * * * /scripts/backup.sh

# WAL archiving is continuous (handled by PostgreSQL)
```

### Point-in-Time Recovery (PITR)

**Restore to specific timestamp**:

```bash
#!/bin/bash
# restore.sh - Restore PostgreSQL to specific point in time

TARGET_TIME="2025-01-15 14:30:00"
LATEST_BACKUP=$(aws s3 ls s3://ntsb-backups/postgres/full/ | sort | tail -n 1 | awk '{print $4}')

# Stop PostgreSQL
systemctl stop postgresql

# Download latest full backup
aws s3 cp "s3://ntsb-backups/postgres/full/$LATEST_BACKUP" /tmp/

# Restore backup
pg_restore -U app -d ntsb -c /tmp/$LATEST_BACKUP

# Create recovery.conf
cat > /var/lib/postgresql/15/main/recovery.conf <<EOF
restore_command = 'aws s3 cp s3://ntsb-backups/wal/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL (recovery mode)
systemctl start postgresql

# Monitor recovery
tail -f /var/log/postgresql/postgresql-15-main.log
```

**Recovery Time Objective (RTO)**: <1 hour
**Recovery Point Objective (RPO)**: <15 minutes (WAL archiving interval)

### Multi-Region Replication

**Streaming replication** to disaster recovery site:

```mermaid
graph LR
    PRIMARY[Primary DB<br/>US-East]
    REPLICA1[Streaming Replica<br/>US-West]
    REPLICA2[Streaming Replica<br/>EU-Central]
    BACKUP[S3 Backups<br/>Multi-region]

    PRIMARY -->|Async Replication| REPLICA1
    PRIMARY -->|Async Replication| REPLICA2
    PRIMARY -->|WAL Archiving| BACKUP
```

**Failover procedure**:

1. **Detection**: Monitor primary with `pg_isready` (alerts on failure)
2. **Promotion**: Promote replica to primary (`pg_ctl promote`)
3. **DNS Update**: Update DNS to point to new primary
4. **Application Restart**: Restart API with new database URL

**Automated failover** with Patroni:

```yaml
# patroni.yml
scope: ntsb-cluster
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: node1:8008

etcd:
  hosts: etcd1:2379,etcd2:2379,etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        wal_level: replica
        hot_standby: "on"
        max_wal_senders: 5
        max_replication_slots: 5

postgresql:
  listen: 0.0.0.0:5432
  connect_address: node1:5432
  data_dir: /var/lib/postgresql/15/main
  authentication:
    replication:
      username: replicator
      password: changeme
    superuser:
      username: postgres
      password: changeme
```

### Backup Testing

**Monthly backup restore drill**:

```bash
#!/bin/bash
# backup_test.sh - Test backup restore procedure

# 1. Restore to staging environment
echo "Testing backup restore..."
./restore.sh --target staging --time latest

# 2. Verify data integrity
psql -U app -d ntsb_staging -c "SELECT COUNT(*) FROM events;"
psql -U app -d ntsb_staging -c "SELECT MAX(ev_date) FROM events;"

# 3. Run data quality checks
python tests/data_quality_tests.py --database ntsb_staging

# 4. Report results
if [ $? -eq 0 ]; then
    echo "Backup test PASSED"
    # Send success notification
else
    echo "Backup test FAILED"
    # Send alert to ops team
fi
```

## Security Architecture

### Network Segmentation

**3-tier architecture** with isolated subnets:

```mermaid
graph TB
    INTERNET[Internet]

    subgraph "Public Subnet (DMZ)"
        LB[Load Balancer<br/>HTTPS only]
        WAF[Web Application Firewall<br/>Rate limiting, DDoS protection]
    end

    subgraph "Private Subnet (Application)"
        API[API Servers<br/>No internet access]
        DASH[Dashboard<br/>No internet access]
    end

    subgraph "Data Subnet (Database)"
        PG[(PostgreSQL<br/>Isolated)]
        REDIS[Redis Cache<br/>Isolated]
    end

    INTERNET --> WAF
    WAF --> LB
    LB --> API
    LB --> DASH
    API --> PG
    API --> REDIS
    DASH --> API
```

**Firewall rules** (iptables):

```bash
# Public subnet (DMZ)
# Allow HTTPS only
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT  # Redirect to 443

# Private subnet (Application)
# Allow only from load balancer
iptables -A INPUT -s 10.0.1.0/24 -p tcp --dport 8000 -j ACCEPT

# Data subnet (Database)
# Allow only from application subnet
iptables -A INPUT -s 10.0.2.0/24 -p tcp --dport 5432 -j ACCEPT
iptables -A INPUT -s 10.0.2.0/24 -p tcp --dport 6379 -j ACCEPT

# Default deny
iptables -P INPUT DROP
iptables -P FORWARD DROP
```

### Secrets Management

**HashiCorp Vault** for secret storage:

```python
import hvac

# Initialize Vault client
client = hvac.Client(
    url='https://vault.example.com:8200',
    token=os.environ['VAULT_TOKEN']
)

# Read database credentials
secret = client.secrets.kv.v2.read_secret_version(
    path='database/ntsb',
    mount_point='secret'
)

db_password = secret['data']['data']['password']
db_url = f"postgresql://app:{db_password}@postgres:5432/ntsb"

# Read API keys
api_secret = client.secrets.kv.v2.read_secret_version(
    path='api/keys',
    mount_point='secret'
)

jwt_secret = api_secret['data']['data']['jwt_secret']
```

**Kubernetes integration** (Vault Secrets Operator):

```yaml
# vault-secret.yaml
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: database-credentials
  namespace: ntsb
spec:
  vaultAuthRef: default
  mount: secret
  path: database/ntsb
  refreshAfter: 3600s
  destination:
    create: true
    name: db-secret
```

### Encryption

**At Rest**:
- PostgreSQL: Transparent Data Encryption (TDE) or LUKS full-disk encryption
- S3/MinIO: Server-side encryption (SSE-S3 or SSE-KMS)
- Backups: Encrypted with GPG before upload

**In Transit**:
- TLS 1.3 for all external communication
- mTLS for internal service-to-service communication

**TLS configuration** (nginx):

```nginx
server {
    listen 443 ssl http2;
    server_name api.ntsb-analysis.com;

    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_ciphers 'TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
    ssl_prefer_server_ciphers on;

    # Certificates
    ssl_certificate /etc/letsencrypt/live/api.ntsb-analysis.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.ntsb-analysis.com/privkey.pem;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/api.ntsb-analysis.com/chain.pem;

    location / {
        proxy_pass http://api:8000;
    }
}
```

### Web Application Firewall (WAF)

**ModSecurity** with OWASP Core Rule Set:

```nginx
# Load ModSecurity
load_module modules/ngx_http_modsecurity_module.so;

http {
    modsecurity on;
    modsecurity_rules_file /etc/nginx/modsecurity.conf;

    server {
        listen 443 ssl;

        location / {
            # Enable OWASP CRS
            modsecurity_rules '
                SecRuleEngine On
                Include /etc/modsecurity/crs-setup.conf
                Include /etc/modsecurity/rules/*.conf
            ';

            proxy_pass http://api:8000;
        }
    }
}
```

**Custom rules** for API protection:

```
# Rate limiting
SecRule REQUEST_URI "@streq /api/v1/predictions/severity" \
    "id:1001,phase:1,pass,setvar:ip.requests=+1,expirevar:ip.requests=60"

SecRule IP:REQUESTS "@gt 100" \
    "id:1002,phase:1,deny,status:429,msg:'Rate limit exceeded'"

# Block SQL injection attempts
SecRule ARGS "@detectSQLi" \
    "id:1003,phase:2,block,msg:'SQL Injection Attack Detected'"

# Block XSS attempts
SecRule ARGS "@detectXSS" \
    "id:1004,phase:2,block,msg:'XSS Attack Detected'"
```

### DDoS Protection

**CloudFlare** (recommended) or AWS Shield:

```yaml
# cloudflare-config.yaml
security_level: "medium"
challenge_ttl: 1800

rate_limiting:
  - threshold: 100
    period: 60
    action: "challenge"
    description: "Rate limit API requests"

firewall_rules:
  - expression: "(http.request.uri.path contains \"/api/v1/predictions\" and ip.geoip.country ne \"US\")"
    action: "challenge"
    description: "Challenge non-US prediction requests"

ddos_protection:
  enabled: true
  sensitivity: "high"
```

## Future Expansion Paths

### Real-Time Streaming

**Apache Kafka** for real-time accident data ingestion:

```mermaid
graph LR
    NTSB_API[NTSB API<br/>Real-time feed]
    KAFKA[Kafka Cluster<br/>3 brokers]
    FLINK[Apache Flink<br/>Stream processing]
    API[FastAPI<br/>WebSocket]
    DASHBOARD[Real-time Dashboard]

    NTSB_API --> KAFKA
    KAFKA --> FLINK
    FLINK --> API
    API --> DASHBOARD
```

**Use cases**:
- Real-time accident alerts (push notifications)
- Live accident map updates
- Streaming ML predictions

### Mobile Applications

**React Native** for iOS/Android:

- Real-time accident notifications
- Offline access to historical data
- Geofenced alerts for pilots (accidents near current location)
- Voice-activated query interface

### Federated Learning

**Privacy-preserving ML** for airline-specific data:

- Train models on airline proprietary data without sharing raw data
- Aggregate model updates from multiple airlines
- Improve accuracy while maintaining data privacy

### Edge Computing

**Real-time alerts** deployed to aircraft systems:

- Onboard ML model for risk assessment
- Alert pilots to similar past accidents in current conditions
- Offline operation (no internet required)

### Blockchain for Audit Trails

**Immutable investigation records**:

- Store investigation findings on blockchain
- Prevent tampering with accident reports
- Public transparency for safety data

---

## Summary

This architecture supports:

- **100K+ accident records** with sub-second query times
- **90%+ ML prediction accuracy** with real-time inference
- **99.9% uptime** with automated failover and monitoring
- **<100ms API latency** (p95) with caching and optimization
- **Horizontal scaling** to 10,000+ concurrent users
- **Cloud-agnostic** deployment (AWS, GCP, Azure, self-hosted)

**Next Steps**: Proceed to `TECHNICAL_IMPLEMENTATION.md` for step-by-step implementation guides.

---

**Version**: 1.0.0 (January 2025)
**Status**: TIER 1 Critical Document - Foundation for Phase 1-5 implementation
**Related Documents**:
- `TECHNICAL_IMPLEMENTATION.md` - Implementation guide
- `NLP_TEXT_MINING.md` - NLP techniques
- `../to-dos/PHASE_1_FOUNDATION.md` - Sprint planning
