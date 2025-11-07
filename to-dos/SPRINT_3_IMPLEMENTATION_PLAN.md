# SPRINT 3 IMPLEMENTATION PLAN: Apache Airflow ETL Pipeline

**Project**: NTSB Aviation Accident Database
**Sprint Duration**: 12 weeks (Weeks 1-12)
**Planning Date**: November 6, 2025
**Version**: 1.0.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Design](#2-architecture-design)
3. [DAG Implementation Specifications](#3-dag-implementation-specifications)
4. [Monitoring & Alerting Design](#4-monitoring--alerting-design)
5. [PRE1982 Integration Plan](#5-pre1982-integration-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Documentation Plan](#7-documentation-plan)
8. [Risk Analysis](#8-risk-analysis)
9. [12-Week Implementation Timeline](#9-12-week-implementation-timeline)
10. [Success Criteria & Metrics](#10-success-criteria--metrics)
11. [Budget & Resource Requirements](#11-budget--resource-requirements)
12. [Decision Log](#12-decision-log)
13. [Appendix: Research Findings](#13-appendix-research-findings)

---

## 1. Executive Summary

### 1.1 Sprint Objectives

Transform the NTSB Aviation Database from **manual ETL** to **fully automated monthly updates** using Apache Airflow orchestration.

**Current State**:
- Manual Python ETL script (`load_with_staging.py`, 597 lines)
- PostgreSQL database (966 MB, 92,771 events, 2000-2025 coverage)
- 6 materialized views, 59 indexes, optimized for analytics
- No automation: Monthly NTSB data requires manual download and load

**Target State**:
- Automated monthly sync from NTSB website
- 5 production Airflow DAGs with 30+ tasks
- Comprehensive monitoring (Slack, email, metrics dashboard)
- PRE1982.MDB integrated (adds 87,000 events, 1962-1981)
- Zero-touch operation after initial setup

### 1.2 Key Deliverables

| Deliverable | Description | Estimated LOC |
|-------------|-------------|---------------|
| **monthly_sync_dag.py** | Automated avall.mdb downloads and loads | 250-300 |
| **data_transformation_dag.py** | Data cleaning and normalization | 200-250 |
| **quality_check_dag.py** | Automated data validation | 150-200 |
| **mv_refresh_dag.py** | Materialized view maintenance | 100-150 |
| **feature_engineering_dag.py** | ML feature preparation | 200-250 |
| **PRE1982 ETL** | Legacy schema integration | 400-500 |
| **Monitoring Infrastructure** | Slack/email alerts, dashboard | 300-400 |
| **Documentation** | Setup, operations, troubleshooting | 8,000 words |

**Total Code**: ~1,600-2,050 lines Python
**Total Documentation**: ~8,000 words

### 1.3 Success Metrics

- **Automation**: 100% hands-off monthly updates (download → validate → load → refresh)
- **Reliability**: 99% DAG success rate over 3 months
- **Performance**: <10 minutes for standard monthly update (30K events)
- **Coverage**: 63 years of data (1962-2025) vs current 26 years
- **Alerting**: <5 minute notification for critical failures
- **Documentation**: GitHub users can deploy in <30 minutes

### 1.4 Strategic Value

**Business Impact**:
- **Operational Efficiency**: Eliminates 2-4 hours/month manual work
- **Data Freshness**: Updates within 24 hours of NTSB release
- **Data Quality**: Automated validation prevents bad data
- **Scalability**: Foundation for ML/AI features (Phase 3)
- **Reproducibility**: All transformations version-controlled

**Technical Impact**:
- **Modern Stack**: Airflow is industry standard for data pipelines
- **Extensibility**: Easy to add new data sources (FAA, weather APIs)
- **Observability**: Full visibility into ETL health and performance
- **Maintainability**: Separation of concerns (5 DAGs vs 1 monolith)

---

## 2. Architecture Design

### 2.1 Deployment Strategy: Docker Compose (Local)

**Decision**: Use **Docker Compose on local hardware** for Airflow deployment.

**Rationale**:

| Factor | Docker Compose (Local) | Kubernetes (K8s) | Cloud Managed (GCP Composer) |
|--------|------------------------|------------------|------------------------------|
| **Cost** | $0 (existing hardware) | $50-100/month (VPS) | $300-500/month |
| **Complexity** | Low (docker-compose up) | High (cluster mgmt) | Low (managed) |
| **Scalability** | Sufficient (30K events/month) | Excellent (auto-scale) | Excellent (auto-scale) |
| **Maintenance** | Moderate (self-managed) | High (K8s expertise) | Low (Google manages) |
| **Control** | Full (local data) | Full | Limited (vendor lock-in) |
| **Setup Time** | 30 minutes | 4-8 hours | 1-2 hours |

**Key Insights from Research**:
- Reddit consensus: "K8s is overkill for small projects, docker-compose works fine on EC2/local" ([source](https://www.reddit.com/r/dataengineering/comments/weykug/))
- Airflow docs: "Docker Compose is NOT production-ready, but works for small deployments" ([source](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/))
- Cost analysis: Cloud Composer = $300-500/month for basic setup ([source](https://www.pythian.com/blog/technical-track/google-cloud-composer-costs-and-performance))

**For This Project**:
- Monthly data: 30K events (~100 MB compressed MDB file)
- Load time: <10 minutes target (currently 30 seconds manual)
- Concurrency: 1-2 concurrent DAG runs (monthly + adhoc)
- **Verdict**: Docker Compose is sufficient and cost-effective

**Migration Path**:
- Start: Docker Compose local (Weeks 1-8)
- Future: Migrate to K8s if scaling needed (>100K events/month or multiple data sources)
- **No code changes required** (DAGs are portable)

### 2.2 Hardware Requirements (Local Deployment)

**Minimum Requirements**:
- **CPU**: 4 cores (Airflow scheduler, webserver, worker, PostgreSQL)
- **RAM**: 8 GB (4 GB Airflow, 2 GB PostgreSQL, 2 GB OS/buffers)
- **Disk**: 50 GB (20 GB Docker images, 20 GB data, 10 GB logs/temp)
- **Network**: 10 Mbps (download 100 MB MDB files)

**Recommended Specifications**:
- **CPU**: 8 cores (for parallel task execution)
- **RAM**: 16 GB (comfortable headroom)
- **Disk**: 100 GB SSD (fast I/O for database)
- **Network**: 50+ Mbps (redundancy)

**Docker Compose Services**:
```yaml
# 5 containers (estimated resource allocation)
postgres:        # 2 GB RAM, 1 CPU
  - Airflow metadata DB + NTSB data DB (separate instances)

redis:           # 512 MB RAM, 0.5 CPU
  - Celery message broker (if using CeleryExecutor)

airflow-webserver: # 1 GB RAM, 1 CPU
  - Web UI on port 8080

airflow-scheduler: # 2 GB RAM, 2 CPU
  - DAG scheduling and task dispatching

airflow-worker:   # 2 GB RAM, 2 CPU
  - Task execution (LocalExecutor) or Celery workers
```

**Total**: ~8 GB RAM, 6-7 CPUs under load

### 2.3 Airflow Configuration

**Executor Choice**: **LocalExecutor** (not CeleryExecutor)

**Rationale**:
- **LocalExecutor**: Runs tasks as subprocesses, no Redis/RabbitMQ needed
- **CeleryExecutor**: Distributed workers, requires message broker (Redis)
- **KubernetesExecutor**: Each task = K8s pod, requires cluster

For monthly 30K event loads, LocalExecutor parallelism (8-10 tasks) is sufficient.

**airflow.cfg Key Settings**:
```ini
[core]
executor = LocalExecutor
parallelism = 10              # Max tasks across all DAGs
dag_concurrency = 5           # Max tasks per DAG
max_active_runs_per_dag = 1   # Prevent duplicate monthly runs
load_examples = False         # Don't load example DAGs

[scheduler]
dag_dir_list_interval = 300   # Scan dags/ every 5 minutes
catchup_by_default = False    # Don't backfill historical runs

[webserver]
expose_config = False         # Security: hide config in UI
rbac = True                   # Enable role-based access control

[logging]
base_log_folder = /opt/airflow/logs
remote_logging = False        # Future: S3 log storage
logging_level = INFO

[smtp]
smtp_host = smtp.gmail.com
smtp_port = 587
smtp_starttls = True
smtp_ssl = False
smtp_user = {{ env_var('AIRFLOW_SMTP_USER') }}
smtp_password = {{ env_var('AIRFLOW_SMTP_PASSWORD') }}
smtp_mail_from = ntsb-airflow@example.com
```

**Environment Variables (.env)**:
```bash
# Airflow core
AIRFLOW__CORE__FERNET_KEY=<generated-fernet-key>
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres-airflow:5432/airflow
AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres-airflow:5432/airflow

# NTSB database connection
AIRFLOW__NTSB__DB_HOST=postgres-ntsb
AIRFLOW__NTSB__DB_PORT=5432
AIRFLOW__NTSB__DB_NAME=ntsb_aviation
AIRFLOW__NTSB__DB_USER=parobek
AIRFLOW__NTSB__DB_PASSWORD=<secure-password>

# SMTP (email alerts)
AIRFLOW_SMTP_USER=ntsb-alerts@gmail.com
AIRFLOW_SMTP_PASSWORD=<app-password>

# Slack (webhook)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

**Security Best Practices**:
1. **Fernet Key**: Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
2. **Secrets Backend**: Use Airflow Connections (stored encrypted in metadata DB)
3. **Database Passwords**: Never hardcode, use environment variables
4. **Web UI**: Enable RBAC, create read-only users for stakeholders
5. **Network**: Run Airflow behind reverse proxy (Nginx) with HTTPS (future enhancement)

### 2.4 File Storage Strategy

**NTSB Data Files**:

**Option 1: Local Filesystem** (Chosen)
```
/opt/airflow/data/
├── downloads/           # Raw MDB files from NTSB
│   ├── avall_2025-11.mdb
│   ├── avall_2025-12.mdb
│   └── PRE1982.MDB
├── extracted/           # CSV files from mdbtools
│   ├── avall_2025-11/
│   │   ├── events.csv
│   │   ├── aircraft.csv
│   │   └── ...
└── archive/             # Older files (retention: 6 months)
    └── avall_2025-05.mdb
```

**Retention Policy**:
- **Current month**: Keep indefinitely (active)
- **Last 6 months**: Keep for rollback/debugging
- **Older files**: Delete to save disk space

**Option 2: Cloud Storage (S3)** (Future Enhancement)
- **Pros**: Unlimited storage, versioning, cheaper than local disk
- **Cons**: Adds AWS dependency, slower download (network latency)
- **When to migrate**: If local disk fills up (>100 GB data)

**Chosen**: Local filesystem for Sprint 3, migrate to S3 in Sprint 4 if needed.

### 2.5 Database Connection Strategy

**Architecture**: **Two Separate PostgreSQL Instances**

```
┌─────────────────────┐
│  Airflow Components │
│  - Scheduler        │
│  - Webserver        │
│  - Workers          │
└──────┬──────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────┐
│ postgres-    │    │ postgres-    │
│ airflow      │    │ ntsb         │
│              │    │              │
│ DB: airflow  │    │ DB: ntsb_    │
│ (metadata)   │    │ aviation     │
│              │    │ (data)       │
│ Port: 5433   │    │ Port: 5432   │
└──────────────┘    └──────────────┘
```

**Rationale**:
1. **Separation of Concerns**: Airflow metadata vs application data
2. **Independent Scaling**: Scale NTSB DB without affecting Airflow
3. **Backup Strategy**: Different backup schedules (metadata vs data)
4. **Resource Isolation**: NTSB queries don't block Airflow scheduler

**Connection Management**:

**Airflow Connection (via UI or CLI)**:
```bash
# Add NTSB PostgreSQL connection
airflow connections add ntsb_postgres \
  --conn-type postgres \
  --conn-host postgres-ntsb \
  --conn-port 5432 \
  --conn-login parobek \
  --conn-password <secure-password> \
  --conn-schema ntsb_aviation
```

**Using PostgresHook in DAGs**:
```python
from airflow.providers.postgres.hooks.postgres import PostgresHook

def load_to_ntsb_db(**context):
    """Load data using connection pooling."""
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Transactional insert
    with hook.get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO events VALUES (...)")
        conn.commit()  # Explicit commit
```

**Connection Pooling**:
- PostgresHook uses psycopg2 connection pool (default: 5 connections)
- Sufficient for LocalExecutor with 10 parallel tasks
- If connection exhaustion: Increase pool size or use PgBouncer

**Transaction Management**:
- **Staging load**: Single transaction per table (fast COPY)
- **Production merge**: Separate transaction (rollback on failure)
- **Materialized views**: Individual transactions (allow partial refresh)

**Rollback Strategy**:
```python
try:
    # Load data
    hook.run("COPY staging.events FROM '/tmp/events.csv'")
    hook.run("INSERT INTO events SELECT * FROM staging.events WHERE ...")
    hook.run("DELETE FROM staging.events")
except Exception as e:
    # Rollback handled automatically by PostgresHook
    logger.error(f"Load failed: {e}")
    raise  # Fail the task, trigger retries
```

### 2.6 System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                         NTSB WEBSITE                              │
│                  https://data.ntsb.gov/avdata                     │
└────────────────────┬──────────────────────────────────────────────┘
                     │ Monthly: Download avall.zip
                     │ (100 MB compressed)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APACHE AIRFLOW CLUSTER                       │
│                    (Docker Compose - Local)                     │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Scheduler  │  │  Webserver  │  │   Worker    │           │
│  │             │  │             │  │             │           │
│  │  Triggers   │  │  UI: 8080   │  │  Executes   │           │
│  │  DAG runs   │  │             │  │  tasks      │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│         │                │                  │                  │
│         └────────────────┴──────────────────┘                  │
│                          │                                     │
│                          ▼                                     │
│              ┌──────────────────────┐                          │
│              │  Metadata DB         │                          │
│              │  (postgres-airflow)  │                          │
│              │  DAG runs, logs      │                          │
│              └──────────────────────┘                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ ETL Tasks
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  NTSB PostgreSQL Database                       │
│                  (postgres-ntsb:5432)                           │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Production      │  │  Staging         │                   │
│  │  Schema          │  │  Schema          │                   │
│  │  - events        │  │  - staging.events│                   │
│  │  - aircraft      │  │  - staging.aircraft                  │
│  │  - ... (11)      │  │  - ... (11)      │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │  Materialized Views (6)                  │                  │
│  │  - mv_yearly_stats                       │                  │
│  │  - mv_state_stats                        │                  │
│  │  - mv_aircraft_stats                     │                  │
│  └──────────────────────────────────────────┘                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Alerts
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MONITORING & ALERTING                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │    Slack     │  │    Email     │  │  Dashboard   │         │
│  │  #ntsb-etl   │  │  SMTP/Gmail  │  │  Streamlit   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow (Monthly Sync)**:
1. **Trigger**: Airflow scheduler detects 1st of month (cron: `0 2 1 * *`)
2. **Download**: `monthly_sync_dag` → NTSB website → `/opt/airflow/data/downloads/avall.mdb`
3. **Extract**: `mdbtools` → CSV files → `/opt/airflow/data/extracted/`
4. **Stage**: `COPY` → `staging.*` tables (bulk load)
5. **Validate**: `quality_check_dag` → run validation queries
6. **Merge**: Deduplicate → INSERT new events → production tables
7. **Transform**: `data_transformation_dag` → normalize data
8. **Refresh**: `mv_refresh_dag` → update materialized views
9. **Alert**: Success → Slack notification + metrics dashboard update

---

## 3. DAG Implementation Specifications

### 3.1 DAG 1: monthly_sync_dag.py

**Purpose**: Automated download and load of monthly NTSB data updates.

**Schedule**: `0 2 1 * *` (2 AM on 1st of every month)

**Estimated LOC**: 250-300 lines

#### Task Breakdown

```python
"""
monthly_sync_dag.py - Automated NTSB Monthly Data Sync

Schedule: 1st of every month at 2 AM (after NTSB releases)
Expected Duration: 5-10 minutes (30K events)
Retries: 3 with exponential backoff
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests
from pathlib import Path

default_args = {
    'owner': 'ntsb-pipeline',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
}

dag = DAG(
    'monthly_sync_dag',
    default_args=default_args,
    description='Download and load monthly NTSB data',
    schedule_interval='0 2 1 * *',  # 2 AM on 1st
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ntsb', 'etl', 'monthly'],
)

# Task 1: Check if new data available
def check_ntsb_website(**context):
    """
    Check NTSB website for new avall.zip file.
    Compare modified date with last successful load.
    """
    ntsb_url = "https://data.ntsb.gov/avdata/avall.zip"

    # HEAD request to get Last-Modified header
    response = requests.head(ntsb_url, timeout=30)
    last_modified = response.headers.get('Last-Modified')

    # Get last successful load date from load_tracking
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')
    last_load = hook.get_first(
        "SELECT load_completed_at FROM load_tracking WHERE database_name = 'avall.mdb' ORDER BY load_completed_at DESC LIMIT 1"
    )

    # Compare dates
    # If new data: return True (continue pipeline)
    # If no new data: return False (skip pipeline)

    context['ti'].xcom_push(key='new_data_available', value=True)

check_new_data = PythonOperator(
    task_id='check_ntsb_website',
    python_callable=check_ntsb_website,
    dag=dag,
)

# Task 2: Download avall.zip
def download_ntsb_data(**context):
    """Download avall.zip from NTSB website (100 MB)."""
    from datetime import datetime
    import shutil

    ntsb_url = "https://data.ntsb.gov/avdata/avall.zip"
    download_dir = Path("/opt/airflow/data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m")
    zip_path = download_dir / f"avall_{timestamp}.zip"

    # Stream download with progress
    response = requests.get(ntsb_url, stream=True, timeout=300)
    with open(zip_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    # Verify file size (should be ~100 MB)
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 50:
        raise ValueError(f"Downloaded file too small: {file_size_mb:.2f} MB")

    context['ti'].xcom_push(key='zip_path', value=str(zip_path))

download_data = PythonOperator(
    task_id='download_ntsb_data',
    python_callable=download_ntsb_data,
    dag=dag,
)

# Task 3: Validate file integrity
validate_file = BashOperator(
    task_id='validate_file_integrity',
    bash_command="""
    ZIP_PATH={{ ti.xcom_pull(task_ids='download_ntsb_data', key='zip_path') }}

    # Check if zip is valid
    unzip -t "$ZIP_PATH"

    # Extract to temp directory
    EXTRACT_DIR="/opt/airflow/data/extracted/$(basename $ZIP_PATH .zip)"
    mkdir -p "$EXTRACT_DIR"
    unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR"

    # Verify avall.mdb exists
    if [ ! -f "$EXTRACT_DIR/avall.mdb" ]; then
        echo "ERROR: avall.mdb not found in zip"
        exit 1
    fi

    # Check file size (should be ~500 MB)
    SIZE=$(stat -c%s "$EXTRACT_DIR/avall.mdb")
    SIZE_MB=$((SIZE / 1024 / 1024))

    if [ $SIZE_MB -lt 400 ]; then
        echo "ERROR: avall.mdb too small: ${SIZE_MB} MB"
        exit 1
    fi

    echo "Validation passed: avall.mdb ${SIZE_MB} MB"
    echo "$EXTRACT_DIR/avall.mdb" > /tmp/avall_path.txt
    """,
    dag=dag,
)

# Task 4: Extract tables to CSV using mdbtools
extract_tables = BashOperator(
    task_id='extract_tables_to_csv',
    bash_command="""
    MDB_PATH=$(cat /tmp/avall_path.txt)
    CSV_DIR="$(dirname $MDB_PATH)/csv"
    mkdir -p "$CSV_DIR"

    # Extract all tables
    TABLES="events aircraft Flight_Crew injury Findings Occurrences seq_of_events Events_Sequence engines narratives NTSB_Admin"

    for TABLE in $TABLES; do
        echo "Extracting $TABLE..."
        mdb-export "$MDB_PATH" "$TABLE" > "$CSV_DIR/${TABLE}.csv"

        # Verify CSV has data (more than header row)
        LINES=$(wc -l < "$CSV_DIR/${TABLE}.csv")
        if [ $LINES -lt 2 ]; then
            echo "WARNING: $TABLE is empty"
        else
            echo "$TABLE: $LINES rows"
        fi
    done

    echo "$CSV_DIR" > /tmp/csv_dir.txt
    """,
    dag=dag,
)

# Task 5: Load to staging tables
def load_to_staging(**context):
    """Bulk COPY CSV files to staging tables."""
    import csv
    from io import StringIO

    csv_dir = Path(open('/tmp/csv_dir.txt').read().strip())
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Clear staging tables
    hook.run("SELECT clear_all_staging();")

    # Load each table
    table_order = [
        'events', 'aircraft', 'Flight_Crew', 'injury', 'Findings',
        'Occurrences', 'seq_of_events', 'Events_Sequence',
        'engines', 'narratives', 'NTSB_Admin'
    ]

    for table in table_order:
        csv_path = csv_dir / f"{table}.csv"

        # Use COPY for bulk load (10x faster than INSERT)
        with hook.get_conn() as conn:
            with conn.cursor() as cursor:
                with open(csv_path, 'r') as f:
                    cursor.copy_expert(
                        f"COPY staging.{table} FROM STDIN WITH CSV HEADER NULL ''",
                        f
                    )
            conn.commit()

        # Log row count
        count = hook.get_first(f"SELECT COUNT(*) FROM staging.{table}")[0]
        context['ti'].xcom_push(key=f'staging_{table}_rows', value=count)

load_staging = PythonOperator(
    task_id='load_to_staging_tables',
    python_callable=load_to_staging,
    dag=dag,
)

# Task 6: Deduplicate and merge to production
deduplicate_merge = PostgresOperator(
    task_id='deduplicate_and_merge',
    postgres_conn_id='ntsb_postgres',
    sql="""
    -- Identify new events (not in production)
    WITH new_events AS (
        SELECT se.ev_id
        FROM staging.events se
        LEFT JOIN events e ON se.ev_id = e.ev_id
        WHERE e.ev_id IS NULL
    )
    -- Insert only new events
    INSERT INTO events
    SELECT se.*
    FROM staging.events se
    INNER JOIN new_events ne ON se.ev_id = ne.ev_id;

    -- Insert child records for new events only
    INSERT INTO aircraft
    SELECT sa.*
    FROM staging.aircraft sa
    INNER JOIN new_events ne ON sa.ev_id = ne.ev_id;

    -- Repeat for all child tables...
    -- (Full SQL in actual implementation)
    """,
    dag=dag,
)

# Task 7: Update load_tracking
update_tracking = PostgresOperator(
    task_id='update_load_tracking',
    postgres_conn_id='ntsb_postgres',
    sql="""
    INSERT INTO load_tracking (
        database_name,
        load_status,
        events_loaded,
        duplicate_events_found,
        load_started_at,
        load_completed_at
    )
    VALUES (
        'avall.mdb',
        'completed',
        (SELECT COUNT(*) FROM staging.events WHERE ev_id NOT IN (SELECT ev_id FROM events)),
        (SELECT COUNT(*) FROM staging.events WHERE ev_id IN (SELECT ev_id FROM events)),
        CURRENT_TIMESTAMP - INTERVAL '{{ dag_run.duration }} seconds',
        CURRENT_TIMESTAMP
    );
    """,
    dag=dag,
)

# Task 8: Cleanup staging
cleanup_staging = PostgresOperator(
    task_id='cleanup_staging_tables',
    postgres_conn_id='ntsb_postgres',
    sql="SELECT clear_all_staging();",
    dag=dag,
)

# Task 9: Send success notification
def send_success_notification(**context):
    """Send Slack + Email notification on success."""
    from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

    # Get metrics from XCom
    staging_events = context['ti'].xcom_pull(key='staging_events_rows')

    # Slack message
    slack_hook = SlackWebhookHook(
        http_conn_id='slack_webhook',
        message=f"✅ NTSB Monthly Sync Completed\n"
                f"- Events loaded: {staging_events:,}\n"
                f"- Duration: {context['dag_run'].duration}\n"
                f"- Next run: 2 AM on 1st of next month",
        channel='#ntsb-etl',
        username='NTSB Airflow Bot',
    )
    slack_hook.execute()

notify_success = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    dag=dag,
)

# Task Dependencies
check_new_data >> download_data >> validate_file >> extract_tables >> load_staging >> deduplicate_merge >> update_tracking >> cleanup_staging >> notify_success
```

#### Error Handling & Retries

**Retry Strategy**:
```python
# Per-task retry configuration
default_args = {
    'retries': 3,                          # Max 3 retries
    'retry_delay': timedelta(minutes=2),   # Start with 2 min delay
    'retry_exponential_backoff': True,     # 2min, 4min, 8min
    'max_retry_delay': timedelta(minutes=10),
}
```

**Failure Callbacks**:
```python
def on_failure_callback(context):
    """Send alert on task failure after all retries exhausted."""
    from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

    task_id = context['task_instance'].task_id
    error = str(context['exception'])
    log_url = context['task_instance'].log_url

    slack_hook = SlackWebhookHook(
        http_conn_id='slack_webhook',
        message=f"🚨 CRITICAL: Monthly Sync Failed\n"
                f"- Task: {task_id}\n"
                f"- Error: {error}\n"
                f"- Logs: {log_url}",
        channel='#ntsb-alerts',
        username='NTSB Airflow Bot',
    )
    slack_hook.execute()

# Add to default_args
default_args['on_failure_callback'] = on_failure_callback
```

**Idempotency**:
- **Download**: Filename includes timestamp, no overwrites
- **Staging**: `clear_all_staging()` before load ensures clean state
- **Merge**: `WHERE NOT EXISTS` prevents duplicate inserts
- **Safe to re-run**: Any task can be retried without side effects

#### Testing Strategy

**Unit Tests** (pytest):
```python
def test_check_ntsb_website():
    """Test website check with mocked requests."""
    with patch('requests.head') as mock_head:
        mock_head.return_value.headers = {'Last-Modified': '2025-11-01'}
        result = check_ntsb_website()
        assert result is True

def test_download_small_file_fails():
    """Test that small files (corrupted) are rejected."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.content = b'small'  # Only 5 bytes
        with pytest.raises(ValueError, match="too small"):
            download_ntsb_data()
```

**Integration Tests**:
```bash
# Test DAG with sample data
airflow dags test monthly_sync_dag 2025-11-01

# Verify staging tables loaded
psql -d ntsb_aviation -c "SELECT * FROM get_row_counts();"
```

---

### 3.2 DAG 2: data_transformation_dag.py

**Purpose**: Data cleaning, normalization, and derived field computation.

**Schedule**: `0 3 1 * *` (3 AM on 1st, after monthly_sync_dag)

**Estimated LOC**: 200-250 lines

#### Task Breakdown

```python
"""
data_transformation_dag.py - Data Cleaning and Normalization

Runs after monthly_sync_dag completes.
Normalizes inconsistent data from NTSB sources.
"""

# Task 1: Normalize aircraft make/model names
normalize_aircraft = PostgresOperator(
    task_id='normalize_aircraft_names',
    sql="""
    -- Fix common typos and variations
    UPDATE aircraft SET
        acft_make = CASE
            WHEN acft_make ILIKE '%CESSNA%' THEN 'Cessna'
            WHEN acft_make ILIKE '%PIPER%' THEN 'Piper'
            WHEN acft_make ILIKE '%BEECH%' THEN 'Beechcraft'
            -- ... more mappings
            ELSE INITCAP(TRIM(acft_make))
        END
    WHERE ev_id IN (
        SELECT ev_id FROM load_tracking
        WHERE load_completed_at > NOW() - INTERVAL '1 day'
    );

    -- Standardize model names
    UPDATE aircraft SET
        acft_model = UPPER(TRIM(acft_model))
    WHERE acft_model IS NOT NULL;
    """,
    dag=dag,
)

# Task 2: Geocode DMS coordinates to decimal
geocode_coordinates = PythonOperator(
    task_id='geocode_dms_to_decimal',
    python_callable=lambda: ...,  # Convert DMS strings to decimal degrees
    dag=dag,
)

# Task 3: Classify injury severity
classify_injuries = PostgresOperator(
    task_id='classify_injury_severity',
    sql="""
    -- Calculate severity score (0-100)
    UPDATE events SET
        severity_score = (
            (inj_tot_f * 100) +  -- Fatal = 100 points
            (inj_tot_s * 50) +   -- Serious = 50 points
            (inj_tot_m * 25) +   -- Minor = 25 points
            (inj_tot_n * 0)      -- None = 0 points
        ) / NULLIF(inj_tot_f + inj_tot_s + inj_tot_m + inj_tot_n, 0)
    WHERE ev_id IN (SELECT ev_id FROM recent_loads);
    """,
    dag=dag,
)

# Task 4: Extract weather conditions
extract_weather = PythonOperator(
    task_id='extract_weather_conditions',
    python_callable=lambda: ...,  # Parse wx_cond_ntsb into structured fields
    dag=dag,
)

# Dependencies
normalize_aircraft >> geocode_coordinates >> classify_injuries >> extract_weather
```

**Transformations**:
1. **Aircraft Name Normalization**: "CESSNA 172" → "Cessna 172"
2. **Coordinate Conversion**: DMS → Decimal degrees
3. **Severity Scoring**: 0-100 scale based on injury counts
4. **Weather Parsing**: "VMC, WIND 10KTS" → structured fields
5. **Date Standardization**: Handle 2-digit years (PRE1982 data)

---

### 3.3 DAG 3: quality_check_dag.py

**Purpose**: Automated data validation using existing `validate_data.sql` checks.

**Schedule**: `0 4 1 * *` (4 AM on 1st, after transformations)

**Estimated LOC**: 150-200 lines

**Validation Framework**: **Custom SQL Checks** (not Great Expectations)

**Rationale**:
- Existing `validate_data.sql` (384 lines) is comprehensive
- Great Expectations: Steep learning curve, heavyweight (20K+ LOC dependency)
- Pandera: Good for pandas DataFrames, not SQL-first workflows
- Custom SQL: Simple, fast, already written

**Research Finding** ([source](https://aeturrell.com/blog/posts/the-data-validation-landscape-in-2025/)):
> "If you're working in a serious production environment and doing wholesale automation, you might find great expectations better... but for SQL-heavy workflows, custom checks are often simpler."

#### Task Breakdown

```python
# Task 1: Row count validation
check_row_counts = PostgresOperator(
    task_id='validate_row_counts',
    sql="""
    -- Verify no data loss during load
    WITH staging_counts AS (
        SELECT
            'events' AS table_name,
            COUNT(*) AS staging_count
        FROM staging.events
        UNION ALL
        SELECT 'aircraft', COUNT(*) FROM staging.aircraft
        -- ... all tables
    ),
    production_counts AS (
        SELECT
            'events' AS table_name,
            COUNT(*) AS prod_count
        FROM events
        WHERE load_id = (SELECT MAX(id) FROM load_tracking)
        -- ... all tables
    )
    SELECT
        s.table_name,
        s.staging_count,
        p.prod_count,
        s.staging_count - p.prod_count AS diff
    FROM staging_counts s
    JOIN production_counts p ON s.table_name = p.table_name
    WHERE s.staging_count != p.prod_count;

    -- Fail if any differences
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM validation_results WHERE diff != 0) THEN
            RAISE EXCEPTION 'Row count mismatch detected';
        END IF;
    END $$;
    """,
    dag=dag,
)

# Task 2: Primary key integrity
check_primary_keys = PostgresOperator(
    task_id='validate_primary_keys',
    sql="""
    -- Check for NULL or duplicate primary keys
    SELECT
        'events' AS table_name,
        COUNT(*) AS null_pks
    FROM events
    WHERE ev_id IS NULL
    UNION ALL
    SELECT
        'events' AS table_name,
        COUNT(*) - COUNT(DISTINCT ev_id) AS duplicate_pks
    FROM events;

    -- ... repeat for all tables
    """,
    dag=dag,
)

# Task 3: Foreign key integrity
check_foreign_keys = PostgresOperator(
    task_id='validate_foreign_keys',
    sql="""
    -- Orphaned aircraft records
    SELECT COUNT(*)
    FROM aircraft a
    LEFT JOIN events e ON a.ev_id = e.ev_id
    WHERE e.ev_id IS NULL;

    -- ... repeat for all child tables
    """,
    dag=dag,
)

# Task 4: Data range validation
check_data_ranges = PostgresOperator(
    task_id='validate_data_ranges',
    sql="""
    -- Invalid coordinates
    SELECT COUNT(*)
    FROM events
    WHERE dec_latitude NOT BETWEEN -90 AND 90
       OR dec_longitude NOT BETWEEN -180 AND 180;

    -- Invalid dates (future dates)
    SELECT COUNT(*)
    FROM events
    WHERE ev_date > CURRENT_DATE;

    -- Invalid crew ages
    SELECT COUNT(*)
    FROM Flight_Crew
    WHERE crew_age NOT BETWEEN 10 AND 120;
    """,
    dag=dag,
)

# Task 5: Generate validation report
def generate_validation_report(**context):
    """Compile validation results and send report."""
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Run all validation queries from validate_data.sql
    results = {}
    with open('/opt/airflow/dags/sql/validate_data.sql') as f:
        queries = f.read().split(';')
        for query in queries:
            if query.strip():
                result = hook.get_records(query)
                results[query[:50]] = result

    # Check for failures
    failures = [k for k, v in results.items() if v and v[0][0] > 0]

    if failures:
        raise ValueError(f"Validation failed: {len(failures)} checks failed")

    # Send success report
    slack_hook.send(
        text=f"✅ Data Quality Check Passed\n"
             f"- {len(results)} checks run\n"
             f"- 0 failures"
    )

validation_report = PythonOperator(
    task_id='generate_validation_report',
    python_callable=generate_validation_report,
    dag=dag,
)

# Dependencies
check_row_counts >> check_primary_keys >> check_foreign_keys >> check_data_ranges >> validation_report
```

**Alert Levels**:
- **CRITICAL** (fail DAG): Orphaned records, duplicate PKs, row count mismatches
- **WARNING** (log only): Missing optional fields, unusual but valid values
- **INFO** (metrics): Query performance, row counts

---

### 3.4 DAG 4: mv_refresh_dag.py

**Purpose**: Refresh materialized views for fast analytics.

**Schedule**: `0 5 1 * *` (5 AM on 1st, after quality checks pass)

**Estimated LOC**: 100-150 lines

**Refresh Strategy**: **CONCURRENTLY** (non-blocking)

**Research Finding** ([source](https://www.postgresqltutorial.com/postgresql-views/postgresql-materialized-views/)):
> "With CONCURRENTLY option, PostgreSQL creates a temporary updated version, compares two versions, and performs INSERT and UPDATE only the differences. PostgreSQL allows you to retrieve data from a materialized view while it is being updated."

**Trade-off**:
- **CONCURRENTLY**: Slower refresh (2-3x), but view remains available
- **Non-CONCURRENTLY**: Fast refresh, but view locked (blocks queries)

**Decision**: Use CONCURRENTLY (availability > speed for monthly updates)

#### Task Breakdown

```python
# Task 1: Refresh yearly stats
refresh_yearly = PostgresOperator(
    task_id='refresh_mv_yearly_stats',
    sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;",
    dag=dag,
)

# Task 2: Refresh state stats
refresh_state = PostgresOperator(
    task_id='refresh_mv_state_stats',
    sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;",
    dag=dag,
)

# Task 3: Refresh aircraft stats
refresh_aircraft = PostgresOperator(
    task_id='refresh_mv_aircraft_stats',
    sql="REFRESH MATERIALIZED VIEW CONCURRENTLY mv_aircraft_stats;",
    dag=dag,
)

# Task 4-6: Refresh remaining 3 views
# ...

# Task 7: Update refresh metadata
update_metadata = PostgresOperator(
    task_id='update_refresh_metadata',
    sql="""
    CREATE TABLE IF NOT EXISTS mv_refresh_log (
        view_name TEXT,
        refresh_started_at TIMESTAMP,
        refresh_completed_at TIMESTAMP,
        duration INTERVAL,
        rows_updated INTEGER
    );

    INSERT INTO mv_refresh_log VALUES
    ('mv_yearly_stats', {{ execution_date }}, CURRENT_TIMESTAMP, ...),
    ('mv_state_stats', {{ execution_date }}, CURRENT_TIMESTAMP, ...),
    -- ... all 6 views
    """,
    dag=dag,
)

# Dependencies: Parallel refresh (all 6 views concurrently)
[refresh_yearly, refresh_state, refresh_aircraft, ...] >> update_metadata
```

**Performance**:
- **mv_yearly_stats**: ~500ms (47 rows)
- **mv_state_stats**: ~800ms (57 rows)
- **mv_aircraft_stats**: ~5 seconds (971 rows)
- **mv_finding_stats**: ~10 seconds (861 rows, complex aggregation)

**Total refresh time**: ~30 seconds (parallel) vs ~60 seconds (sequential)

---

### 3.5 DAG 5: feature_engineering_dag.py

**Purpose**: Prepare ML/AI features for Phase 3 (predictive analytics).

**Schedule**: `0 6 1 * *` (6 AM on 1st, after MV refresh)

**Estimated LOC**: 200-250 lines

**Features to Engineer**:

1. **Temporal Features**:
   - Days since last accident (same aircraft make/model)
   - 7-day, 30-day, 90-day moving averages (accidents per day)
   - Seasonality flags (summer, winter, holidays)

2. **Spatial Features**:
   - Distance to nearest major airport
   - Airspace complexity score (Class A/B/C/D/E/G)
   - Terrain ruggedness index (mountains, flat, water)

3. **Text Features**:
   - TF-IDF on narratives (top 100 keywords)
   - Sentiment scores (positive/negative/neutral)
   - Named entity extraction (airports, locations, aircraft parts)

4. **Risk Factors**:
   - Pilot experience score (total hours, 90-day recency)
   - Aircraft age (years since manufacture)
   - Weather severity score (VMC=0, MVFR=1, IFR=2, LIFR=3)

#### Task Breakdown

```python
# Task 1: Temporal feature engineering
def compute_temporal_features(**context):
    """Calculate time-based features."""
    import pandas as pd
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Load events
    query = """
        SELECT ev_id, ev_date, acft_make, acft_model
        FROM events e
        JOIN aircraft a ON e.ev_id = a.ev_id
        ORDER BY ev_date
    """
    df = hook.get_pandas_df(query)

    # Days since last accident (same aircraft type)
    df['days_since_last_accident'] = df.groupby(['acft_make', 'acft_model'])['ev_date'].diff().dt.days

    # 30-day rolling accident count
    df = df.set_index('ev_date')
    df['accidents_last_30_days'] = df.groupby(['acft_make', 'acft_model']).rolling('30D').size().reset_index(0, drop=True)

    # Seasonality
    df['is_summer'] = df.index.month.isin([6, 7, 8])
    df['is_winter'] = df.index.month.isin([12, 1, 2])

    # Write back to database
    df.to_sql('ml_temporal_features', hook.get_sqlalchemy_engine(), if_exists='replace', index=False)

temporal_features = PythonOperator(
    task_id='compute_temporal_features',
    python_callable=compute_temporal_features,
    dag=dag,
)

# Task 2: Spatial feature engineering
def compute_spatial_features(**context):
    """Calculate location-based features using PostGIS."""
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Nearest airport distance (PostGIS)
    query = """
    CREATE TABLE ml_spatial_features AS
    SELECT
        e.ev_id,
        ST_Distance(
            e.location_geom::geography,
            (SELECT geom FROM airports ORDER BY e.location_geom <-> geom LIMIT 1)::geography
        ) / 1609.34 AS nearest_airport_miles,
        -- Airspace complexity (mock - requires FAA airspace data)
        CASE
            WHEN ST_Within(e.location_geom, (SELECT geom FROM airspace WHERE class = 'B')) THEN 5
            WHEN ST_Within(e.location_geom, (SELECT geom FROM airspace WHERE class = 'C')) THEN 4
            ELSE 1
        END AS airspace_complexity
    FROM events e
    WHERE e.location_geom IS NOT NULL;
    """
    hook.run(query)

spatial_features = PythonOperator(
    task_id='compute_spatial_features',
    python_callable=compute_spatial_features,
    dag=dag,
)

# Task 3: Text feature engineering (NLP)
def compute_text_features(**context):
    """Extract features from narratives using NLP."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Load narratives
    df = hook.get_pandas_df("SELECT ev_id, narr_cause FROM narratives WHERE narr_cause IS NOT NULL")

    # TF-IDF (top 100 keywords)
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['narr_cause'])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{word}' for word in vectorizer.get_feature_names_out()]
    )
    tfidf_df['ev_id'] = df['ev_id'].values

    # Write to database
    tfidf_df.to_sql('ml_text_features', hook.get_sqlalchemy_engine(), if_exists='replace', index=False)

text_features = PythonOperator(
    task_id='compute_text_features',
    python_callable=compute_text_features,
    dag=dag,
)

# Task 4: Risk score calculation
risk_scores = PostgresOperator(
    task_id='calculate_risk_scores',
    sql="""
    CREATE TABLE ml_risk_scores AS
    SELECT
        e.ev_id,
        -- Pilot experience score (0-100)
        LEAST(100, fc.pilot_tot_time / 100) AS pilot_experience_score,
        -- Aircraft age (years)
        EXTRACT(YEAR FROM e.ev_date) - a.acft_year AS aircraft_age_years,
        -- Weather severity (0-3)
        CASE
            WHEN e.wx_cond_ntsb ILIKE '%VMC%' THEN 0
            WHEN e.wx_cond_ntsb ILIKE '%MVFR%' THEN 1
            WHEN e.wx_cond_ntsb ILIKE '%IFR%' THEN 2
            WHEN e.wx_cond_ntsb ILIKE '%LIFR%' THEN 3
            ELSE 0
        END AS weather_severity
    FROM events e
    LEFT JOIN aircraft a ON e.ev_id = a.ev_id
    LEFT JOIN Flight_Crew fc ON e.ev_id = fc.ev_id
    WHERE fc.crew_no = 1;  -- Primary pilot
    """,
    dag=dag,
)

# Dependencies: All features in parallel, then combine
[temporal_features, spatial_features, text_features] >> risk_scores
```

**Output**: 4 tables in `public` schema:
- `ml_temporal_features` (~92K rows, 5 columns)
- `ml_spatial_features` (~92K rows, 3 columns)
- `ml_text_features` (~27K rows, 101 columns)
- `ml_risk_scores` (~92K rows, 4 columns)

**Future Use** (Phase 3):
- Accident severity prediction (classification)
- Causal factor extraction (NLP)
- Trend forecasting (time series)

---

## 4. Monitoring & Alerting Design

### 4.1 Alert Severity Levels

| Level | Trigger | Notification Channels | Example |
|-------|---------|----------------------|---------|
| **CRITICAL** | DAG failure, data corruption, load tracking inconsistency | Slack + Email | "Monthly sync failed after 3 retries" |
| **WARNING** | Data quality issues, slow queries (>2x expected), missing optional fields | Slack only | "15 events missing narratives" |
| **INFO** | Successful runs, metrics summary, scheduled maintenance | Dashboard only | "Monthly sync: 30,245 events in 8m 32s" |

### 4.2 Notification Templates

#### Slack Notification (Critical)

```python
def send_slack_alert_critical(context):
    """Send critical alert with full context."""
    from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

    dag_id = context['dag'].dag_id
    task_id = context['task_instance'].task_id
    execution_date = context['execution_date']
    exception = str(context.get('exception', 'Unknown error'))
    log_url = context['task_instance'].log_url

    # Slack Block Kit message (rich formatting)
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 CRITICAL: Airflow DAG Failure",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*DAG:*\n{dag_id}"},
                {"type": "mrkdwn", "text": f"*Task:*\n{task_id}"},
                {"type": "mrkdwn", "text": f"*Execution Date:*\n{execution_date}"},
                {"type": "mrkdwn", "text": f"*Duration:*\n{context['dag_run'].duration}"}
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Error:*\n```{exception}```"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Logs"},
                    "url": log_url,
                    "style": "danger"
                }
            ]
        }
    ]

    slack_hook = SlackWebhookHook(
        http_conn_id='slack_webhook',
        message="CRITICAL Alert",
        blocks=blocks,
        channel='#ntsb-alerts',
        username='NTSB Airflow Bot',
        icon_emoji=':rotating_light:',
    )
    slack_hook.execute()
```

**Slack Webhook Setup**:
1. Create Slack App: https://api.slack.com/apps
2. Enable Incoming Webhooks
3. Copy webhook URL: `https://hooks.slack.com/services/T00/B00/XXX`
4. Add to Airflow Connections:
   ```bash
   airflow connections add slack_webhook \
     --conn-type http \
     --conn-host https://hooks.slack.com \
     --conn-password /services/T00/B00/XXX
   ```

#### Email Notification (Critical)

```python
def send_email_alert_critical(context):
    """Send HTML email with failure details."""
    from airflow.utils.email import send_email

    dag_id = context['dag'].dag_id
    task_id = context['task_instance'].task_id
    exception = str(context.get('exception', 'Unknown'))
    log_url = context['task_instance'].log_url

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .alert {{ background-color: #f44336; color: white; padding: 20px; }}
            .details {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; }}
            .button {{ background-color: #008CBA; color: white; padding: 10px 20px; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="alert">
            <h1>🚨 Airflow DAG Failure</h1>
        </div>
        <div class="details">
            <p><strong>DAG:</strong> {dag_id}</p>
            <p><strong>Task:</strong> {task_id}</p>
            <p><strong>Execution Date:</strong> {context['execution_date']}</p>
            <p><strong>Error:</strong></p>
            <pre>{exception}</pre>
        </div>
        <a href="{log_url}" class="button">View Logs</a>
    </body>
    </html>
    """

    send_email(
        to=['admin@example.com', 'data-team@example.com'],
        subject=f'[CRITICAL] Airflow DAG Failure: {dag_id}',
        html_content=html_content,
    )
```

**Gmail SMTP Setup**:
1. Enable 2-Factor Authentication on Google Account
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Add to `.env`:
   ```bash
   AIRFLOW_SMTP_USER=ntsb-alerts@gmail.com
   AIRFLOW_SMTP_PASSWORD=<16-char-app-password>
   ```

#### Slack Notification (Success)

```python
def send_slack_success(context):
    """Send concise success message."""
    dag_id = context['dag'].dag_id
    duration = context['dag_run'].duration

    # Get metrics from XCom
    events_loaded = context['ti'].xcom_pull(key='staging_events_rows')

    message = (
        f"✅ *{dag_id}* completed successfully\n"
        f"- Events loaded: {events_loaded:,}\n"
        f"- Duration: {duration}\n"
        f"- Next run: {context['next_execution_date']}"
    )

    slack_hook = SlackWebhookHook(
        http_conn_id='slack_webhook',
        message=message,
        channel='#ntsb-etl',
        username='NTSB Airflow Bot',
        icon_emoji=':white_check_mark:',
    )
    slack_hook.execute()
```

### 4.3 Metrics Dashboard (Streamlit)

**Purpose**: Real-time visibility into ETL health and performance.

**Metrics to Track**:
1. **DAG Run Metrics**:
   - Success rate (last 30 days): 99%+
   - Average duration: 8 minutes
   - Failure count: 0
   - Retry count: 2 (transient network errors)

2. **Data Metrics**:
   - Events loaded (monthly): 30,000
   - Database size growth: +50 MB/month
   - Row count by table (time series)
   - Duplicate events detected: 0

3. **Performance Metrics**:
   - Task duration percentiles (p50, p95, p99)
   - Slowest tasks (top 10)
   - Query performance (MV refresh times)

4. **Data Quality Metrics**:
   - Validation check pass rate: 100%
   - Orphaned records: 0
   - NULL value rates (trending)

**Streamlit Dashboard Code** (250 lines):
```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from airflow.providers.postgres.hooks.postgres import PostgresHook

st.set_page_config(page_title="NTSB ETL Dashboard", layout="wide")
st.title("🛩️ NTSB Aviation Database - ETL Dashboard")

# Connect to Airflow metadata DB
hook_airflow = PostgresHook(postgres_conn_id='airflow_metadata')
hook_ntsb = PostgresHook(postgres_conn_id='ntsb_postgres')

# DAG Run Metrics
st.header("📊 DAG Run Metrics (Last 30 Days)")
dag_runs_query = """
    SELECT
        dag_id,
        state,
        COUNT(*) AS run_count,
        AVG(EXTRACT(EPOCH FROM (end_date - start_date))) AS avg_duration_sec
    FROM dag_run
    WHERE start_date > NOW() - INTERVAL '30 days'
    GROUP BY dag_id, state
    ORDER BY dag_id, state;
"""
df_dag_runs = hook_airflow.get_pandas_df(dag_runs_query)

# Success rate chart
fig_success = px.bar(
    df_dag_runs,
    x='dag_id',
    y='run_count',
    color='state',
    title="DAG Run Success/Failure Count"
)
st.plotly_chart(fig_success, use_container_width=True)

# Data Quality Metrics
st.header("✅ Data Quality Metrics")
quality_query = """
    SELECT
        'Orphaned Aircraft' AS check_name,
        COUNT(*) AS failure_count
    FROM aircraft a
    LEFT JOIN events e ON a.ev_id = e.ev_id
    WHERE e.ev_id IS NULL
    UNION ALL
    SELECT
        'Duplicate Events' AS check_name,
        COUNT(*) - COUNT(DISTINCT ev_id) AS failure_count
    FROM events;
"""
df_quality = hook_ntsb.get_pandas_df(quality_query)
st.dataframe(df_quality, use_container_width=True)

# Database Size Trend
st.header("💾 Database Size Trend")
size_query = """
    SELECT
        DATE_TRUNC('month', load_completed_at) AS month,
        SUM(events_loaded) AS cumulative_events
    FROM load_tracking
    GROUP BY month
    ORDER BY month;
"""
df_size = hook_ntsb.get_pandas_df(size_query)
fig_size = px.line(df_size, x='month', y='cumulative_events', title="Cumulative Events Loaded")
st.plotly_chart(fig_size, use_container_width=True)

# Task Duration Heatmap
st.header("⏱️ Task Duration Heatmap (Last 7 Days)")
task_duration_query = """
    SELECT
        task_id,
        DATE(start_date) AS date,
        AVG(EXTRACT(EPOCH FROM (end_date - start_date))) AS avg_duration_sec
    FROM task_instance
    WHERE start_date > NOW() - INTERVAL '7 days'
      AND state = 'success'
    GROUP BY task_id, DATE(start_date)
    ORDER BY task_id, date;
"""
df_tasks = hook_airflow.get_pandas_df(task_duration_query)
pivot = df_tasks.pivot(index='task_id', columns='date', values='avg_duration_sec')
fig_heatmap = px.imshow(pivot, title="Task Duration Heatmap (seconds)")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Refresh every 5 minutes
st.button("Refresh Dashboard")
```

**Deployment**:
```bash
# Run Streamlit dashboard on port 8501
streamlit run dashboard.py --server.port 8501
```

**Access**: http://localhost:8501

### 4.4 Alert Configuration Summary

| Alert Type | Trigger | Slack Channel | Email Recipients | Frequency |
|------------|---------|---------------|------------------|-----------|
| **DAG Failure** | Any task fails after 3 retries | #ntsb-alerts | admin@, data-team@ | Immediate |
| **Data Quality** | Validation check fails | #ntsb-alerts | data-team@ | Immediate |
| **Slow Query** | Task duration >2x p95 | #ntsb-etl | None | Daily digest |
| **Success** | DAG completes successfully | #ntsb-etl | None | Per run |
| **Metrics** | Dashboard update | None | None | Real-time |

---

## 5. PRE1982 Integration Plan

### 5.1 Challenge Summary

**Problem**: PRE1982.MDB uses incompatible schema (denormalized, 200+ columns, coded fields).

**From Analysis** (`docs/PRE1982_ANALYSIS.md`):
- **Date Range**: 1962-1981 (20 years)
- **Total Events**: ~87,000
- **Schema**: 5 tables vs 11 in modern database
- **Primary Key**: `RecNum` (integer) vs `ev_id` (VARCHAR)
- **Structure**: Denormalized wide tables vs normalized relations

**Integration Effort**: 8-16 hours (custom ETL required)

### 5.2 ETL Strategy: Schema Mapping + Transformation

**Phase 1: Analysis & Mapping** (2-3 hours)

Create mapping table for coded fields:
```sql
CREATE TABLE pre1982_code_mappings (
    legacy_column TEXT,
    legacy_value TEXT,
    modern_column TEXT,
    modern_value TEXT,
    notes TEXT
);

-- Example mappings
INSERT INTO pre1982_code_mappings VALUES
('CAUSE_FACTOR_1P', 'F210', 'cause_factor', 'Improper use of flight controls', 'Section IB Code'),
('TYPE_CRAFT', '1', 'acft_category', 'Airplane', 'Numeric code → text'),
('MEDICAL_CERT_PILOT1', '2', 'pilot_med_cert', 'Class 2', 'Medical certificate type'),
-- ... 100+ mappings from ref_docs/codman.pdf
```

**Phase 2: ETL Pipeline Development** (4-6 hours)

```python
# dags/pre1982_integration_dag.py

def extract_pre1982(**context):
    """Extract PRE1982.MDB to CSV using mdbtools."""
    subprocess.run([
        'mdb-export',
        'datasets/PRE1982.MDB',
        'tblFirstHalf',
        '>',
        '/tmp/pre1982_first_half.csv'
    ])
    subprocess.run([
        'mdb-export',
        'datasets/PRE1982.MDB',
        'tblSecondHalf',
        '>',
        '/tmp/pre1982_second_half.csv'
    ])

def transform_pre1982(**context):
    """Transform legacy schema to modern schema."""
    import pandas as pd

    # Load both halves
    df_first = pd.read_csv('/tmp/pre1982_first_half.csv')
    df_second = pd.read_csv('/tmp/pre1982_second_half.csv')

    # Merge on RecNum
    df = df_first.merge(df_second, on='RecNum')

    # Transform to events table
    events = pd.DataFrame({
        'ev_id': 'PRE1982_' + df['RecNum'].astype(str),  # Synthetic ev_id
        'ev_date': pd.to_datetime(df['DATE_OCCURRENCE'], format='%m/%d/%y %H:%M:%S').dt.date,
        'ev_year': pd.to_datetime(df['DATE_OCCURRENCE']).dt.year,
        'ev_month': pd.to_datetime(df['DATE_OCCURRENCE']).dt.month,
        'ev_state': df['LOCAT_STATE_TERR'],
        'ev_city': df['LOCATION'],
        'dec_latitude': df['LATITUDE'],  # May need DMS → decimal conversion
        'dec_longitude': df['LONGITUDE'],
        'inj_tot_f': df['PILOT_FATAL'] + df['PASSENGERS_FATAL'],
        'inj_tot_s': df['PILOT_SERIOUS'] + df['PASSENGERS_SERIOUS'],
        # ... map all 40+ events columns
    })

    # Transform to aircraft table
    aircraft = pd.DataFrame({
        'ev_id': 'PRE1982_' + df['RecNum'].astype(str),
        'Aircraft_Key': 'PRE1982_' + df['RecNum'].astype(str) + '_1',
        'acft_make': df['ACFT_MAKE'],
        'acft_model': df['ACFT_MODEL'],
        'regist_no': df['REGIST_NO'],
        'num_eng': df['NO_ENGINES'],
        # ... map aircraft columns
    })

    # Map coded fields using lookup table
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')
    mappings = hook.get_pandas_df("SELECT * FROM pre1982_code_mappings")

    for _, mapping in mappings.iterrows():
        legacy_col = mapping['legacy_column']
        if legacy_col in df.columns:
            events[mapping['modern_column']] = df[legacy_col].map(
                dict(zip(mappings['legacy_value'], mappings['modern_value']))
            )

    # Save transformed tables
    events.to_csv('/tmp/pre1982_events.csv', index=False)
    aircraft.to_csv('/tmp/pre1982_aircraft.csv', index=False)

def load_pre1982(**context):
    """Load transformed data to production."""
    hook = PostgresHook(postgres_conn_id='ntsb_postgres')

    # Load events
    with hook.get_conn() as conn:
        with conn.cursor() as cursor:
            with open('/tmp/pre1982_events.csv', 'r') as f:
                cursor.copy_expert(
                    "COPY events FROM STDIN WITH CSV HEADER",
                    f
                )
        conn.commit()

    # Load aircraft, Flight_Crew, etc.
    # ...

# DAG definition
pre1982_dag = DAG(
    'pre1982_integration_dag',
    schedule_interval=None,  # Manual trigger only
    default_args=default_args,
    tags=['ntsb', 'pre1982', 'legacy'],
)

extract = PythonOperator(task_id='extract_pre1982', python_callable=extract_pre1982, dag=pre1982_dag)
transform = PythonOperator(task_id='transform_pre1982', python_callable=transform_pre1982, dag=pre1982_dag)
load = PythonOperator(task_id='load_pre1982', python_callable=load_pre1982, dag=pre1982_dag)

extract >> transform >> load
```

**Phase 3: Validation & Testing** (2-3 hours)

```sql
-- Verify PRE1982 events loaded
SELECT COUNT(*) FROM events WHERE ev_id LIKE 'PRE1982_%';
-- Expected: ~87,000

-- Check date range
SELECT MIN(ev_date), MAX(ev_date) FROM events WHERE ev_id LIKE 'PRE1982_%';
-- Expected: 1962-01-01 to 1981-12-31

-- Verify no orphaned records
SELECT COUNT(*) FROM aircraft WHERE ev_id LIKE 'PRE1982_%' AND ev_id NOT IN (SELECT ev_id FROM events);
-- Expected: 0
```

**Phase 4: Documentation** (1-2 hours)

Document in `docs/PRE1982_INTEGRATION.md`:
- Schema mapping table
- Coded field translations
- Known limitations (missing fields, data quality issues)
- Load statistics

### 5.3 PRE1982 Integration Timeline

**Week 7**: Schema mapping design (2 days)
**Week 8**: ETL development + unit tests (3 days)
**Week 9**: Full integration test + validation (2 days)
**Week 10**: Performance optimization + documentation (2 days)

**Total Effort**: 9 days (~72 hours) vs estimated 8-16 hours (conservative estimate confirmed)

**Deliverables**:
- `pre1982_integration_dag.py` (400-500 LOC)
- `pre1982_code_mappings` table (100+ rows)
- `docs/PRE1982_INTEGRATION.md` (2,000 words)
- 87,000 additional events (1962-1981)

---

## 6. Testing Strategy

### 6.1 Unit Testing (pytest)

**Framework**: pytest + pytest-airflow + pytest-mock

**Test Coverage**:
- **Python Functions**: All ETL transform functions (100% coverage)
- **SQL Queries**: Syntax validation (no runtime errors)
- **Airflow DAGs**: Import tests (no syntax errors)

**Example Tests**:
```python
# tests/test_monthly_sync_dag.py
import pytest
from unittest.mock import patch, MagicMock
from dags.monthly_sync_dag import check_ntsb_website, download_ntsb_data

def test_check_ntsb_website_new_data_available():
    """Test website check detects new data."""
    with patch('requests.head') as mock_head:
        mock_head.return_value.headers = {'Last-Modified': 'Thu, 01 Nov 2025 12:00:00 GMT'}

        with patch('airflow.providers.postgres.hooks.postgres.PostgresHook.get_first') as mock_db:
            mock_db.return_value = ('2025-10-01 00:00:00',)

            context = {'ti': MagicMock()}
            check_ntsb_website(**context)

            # Assert XCom push called with True
            context['ti'].xcom_push.assert_called_with(key='new_data_available', value=True)

def test_download_fails_on_small_file():
    """Test download rejects corrupted (small) files."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.raw = MagicMock()
        mock_get.return_value.raw.read.return_value = b'corrupted'  # Only 9 bytes

        with pytest.raises(ValueError, match="too small"):
            download_ntsb_data(**{})

def test_dag_import_without_errors():
    """Test all DAGs can be imported without errors."""
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder='dags/', include_examples=False)

    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    assert len(dagbag.dags) == 5, "Expected 5 DAGs"

# Run tests
# pytest tests/ -v --cov=dags --cov-report=html
```

**Test Execution**:
```bash
# Install test dependencies
pip install pytest pytest-airflow pytest-mock pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=dags --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Target Coverage**: 80%+ (focus on critical ETL logic)

### 6.2 Integration Testing

**Purpose**: Test full DAG execution with real database.

**Test Environment**:
- Separate PostgreSQL instance: `ntsb_aviation_test`
- Sample data: 100 events from each time period (avall, Pre2008, PRE1982)
- Isolated Docker Compose stack

**Test Cases**:

**Test 1: End-to-End Monthly Sync**
```bash
# Trigger DAG with sample data
airflow dags test monthly_sync_dag 2025-11-01

# Verify results
psql -d ntsb_aviation_test -c "
    SELECT COUNT(*) FROM events;
    SELECT COUNT(*) FROM aircraft;
    SELECT * FROM load_tracking ORDER BY load_completed_at DESC LIMIT 1;
"
```

**Test 2: Duplicate Event Handling**
```bash
# Load same data twice
airflow dags test monthly_sync_dag 2025-11-01
airflow dags test monthly_sync_dag 2025-11-02

# Verify no duplicates
psql -d ntsb_aviation_test -c "
    SELECT ev_id, COUNT(*) FROM events GROUP BY ev_id HAVING COUNT(*) > 1;
"
# Expected: 0 rows
```

**Test 3: Rollback on Failure**
```python
# Simulate failure in merge step
def test_rollback_on_merge_failure():
    """Test that failed merge doesn't corrupt database."""
    # Insert bad data that violates foreign key
    # Verify rollback leaves database in previous state
    pass
```

**Test 4: Materialized View Refresh**
```bash
# Load data, refresh MVs, verify counts
airflow dags test mv_refresh_dag 2025-11-01

psql -d ntsb_aviation_test -c "
    SELECT COUNT(*) FROM mv_yearly_stats;
    SELECT COUNT(*) FROM mv_aircraft_stats;
"
```

### 6.3 Performance Testing

**Goal**: Verify <10 minute target for monthly update.

**Benchmark Tests**:

**Test 1: Load 30K Events (Target: <10 minutes)**
```bash
# Start timer
START_TIME=$(date +%s)

# Trigger monthly_sync_dag with realistic dataset
airflow dags trigger monthly_sync_dag

# Wait for completion
airflow dags list-runs -d monthly_sync_dag --state success --limit 1

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Load duration: ${DURATION} seconds"

# Assert: DURATION < 600 (10 minutes)
if [ $DURATION -gt 600 ]; then
    echo "FAIL: Load took ${DURATION}s, target is 600s"
    exit 1
else
    echo "PASS: Load completed in ${DURATION}s"
fi
```

**Test 2: Materialized View Refresh (Target: <60 seconds)**
```sql
-- Time MV refresh
\timing
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_aircraft_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_decade_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_crew_stats;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_finding_stats;
\timing
```

**Test 3: Concurrent DAG Runs (Stress Test)**
```bash
# Trigger 3 different DAGs simultaneously
airflow dags trigger monthly_sync_dag &
airflow dags trigger data_transformation_dag &
airflow dags trigger quality_check_dag &

# Monitor system resources
docker stats --no-stream
```

**Performance Targets**:
| Task | Target | Measured | Pass/Fail |
|------|--------|----------|-----------|
| Download 100 MB MDB | <60s | TBD | - |
| Extract to CSV | <120s | TBD | - |
| Load to staging | <180s | TBD | - |
| Dedupe + merge | <240s | TBD | - |
| MV refresh (all 6) | <60s | TBD | - |
| **Total pipeline** | **<600s (10 min)** | **TBD** | **-** |

### 6.4 Data Quality Testing

**Framework**: Custom SQL checks (reuse `validate_data.sql`)

**Test Cases**:

```python
# tests/test_data_quality.py
import pytest
from airflow.providers.postgres.hooks.postgres import PostgresHook

@pytest.fixture
def db_hook():
    return PostgresHook(postgres_conn_id='ntsb_postgres_test')

def test_no_orphaned_aircraft(db_hook):
    """Verify all aircraft have corresponding events."""
    result = db_hook.get_first("""
        SELECT COUNT(*)
        FROM aircraft a
        LEFT JOIN events e ON a.ev_id = e.ev_id
        WHERE e.ev_id IS NULL
    """)
    assert result[0] == 0, f"Found {result[0]} orphaned aircraft records"

def test_no_duplicate_events(db_hook):
    """Verify no duplicate event IDs."""
    result = db_hook.get_first("""
        SELECT COUNT(*) - COUNT(DISTINCT ev_id)
        FROM events
    """)
    assert result[0] == 0, f"Found {result[0]} duplicate events"

def test_valid_coordinate_ranges(db_hook):
    """Verify all coordinates within valid ranges."""
    result = db_hook.get_first("""
        SELECT COUNT(*)
        FROM events
        WHERE dec_latitude NOT BETWEEN -90 AND 90
           OR dec_longitude NOT BETWEEN -180 AND 180
    """)
    assert result[0] == 0, f"Found {result[0]} events with invalid coordinates"

def test_future_dates_not_allowed(db_hook):
    """Verify no events dated in the future."""
    result = db_hook.get_first("""
        SELECT COUNT(*)
        FROM events
        WHERE ev_date > CURRENT_DATE
    """)
    assert result[0] == 0, f"Found {result[0]} events with future dates"
```

**Test Execution**:
```bash
pytest tests/test_data_quality.py -v
```

### 6.5 Testing Timeline

**Week 4**: Unit tests (DAG 1-5) (2 days)
**Week 6**: Integration tests (end-to-end) (2 days)
**Week 8**: Performance benchmarks (1 day)
**Week 10**: Data quality tests (PRE1982 integration) (1 day)
**Week 12**: Final regression testing (1 day)

**Total Testing Effort**: 7 days (~15% of Sprint 3 duration)

---

## 7. Documentation Plan

### 7.1 Developer Documentation

**File**: `docs/AIRFLOW_DEVELOPER_GUIDE.md` (~3,000 words)

**Contents**:
1. **Local Development Setup**
   - Clone repo, install dependencies
   - Configure `.env` file
   - Start Docker Compose stack
   - Access Airflow UI (http://localhost:8080)

2. **DAG Development**
   - File structure (`dags/`, `plugins/`, `config/`)
   - Writing new DAGs (template, best practices)
   - Testing DAGs locally (`airflow dags test`)
   - Debugging (logs, Airflow UI, VS Code debugger)

3. **Database Development**
   - Connecting to NTSB PostgreSQL
   - Running migrations (schema changes)
   - Testing with sample data

4. **Troubleshooting**
   - Common errors (connection refused, import errors)
   - Restarting services (`docker-compose restart`)
   - Clearing DAG runs (`airflow dags delete`)

### 7.2 Operator Documentation

**File**: `docs/AIRFLOW_OPERATIONS_GUIDE.md` (~2,500 words)

**Contents**:
1. **Deployment**
   - Production deployment checklist
   - Environment variables (production vs staging)
   - SSL certificate setup (HTTPS for web UI)
   - Backup strategy (Airflow metadata + NTSB data)

2. **Monitoring**
   - Accessing Slack alerts (#ntsb-etl, #ntsb-alerts)
   - Reading email notifications
   - Using Streamlit dashboard (http://localhost:8501)
   - Airflow web UI tour (DAG view, Graph view, Gantt view, Logs)

3. **Operations**
   - Triggering manual DAG runs
     ```bash
     airflow dags trigger monthly_sync_dag
     ```
   - Pausing/Unpausing DAGs
     ```bash
     airflow dags pause monthly_sync_dag
     airflow dags unpause monthly_sync_dag
     ```
   - Clearing failed tasks (retry)
     ```bash
     airflow tasks clear monthly_sync_dag check_ntsb_website -d 2025-11-01
     ```
   - Backfilling historical runs
     ```bash
     airflow dags backfill monthly_sync_dag -s 2025-01-01 -e 2025-11-01
     ```

4. **Incident Response**
   - What to do when DAG fails
     1. Check Slack/email for error message
     2. View logs in Airflow UI
     3. Identify root cause (network, data corruption, bug)
     4. Clear task and retry (or fix code and redeploy)
   - Rollback procedures
     1. Restore database from backup
     2. Clear load_tracking entry
     3. Re-run DAG from scratch
   - Escalation path (who to contact for help)

### 7.3 Architecture Documentation

**File**: `docs/AIRFLOW_ARCHITECTURE.md` (~2,500 words)

**Contents**:
1. **System Architecture**
   - Diagram (Airflow → PostgreSQL → NTSB website)
   - Component descriptions (scheduler, webserver, worker, databases)
   - Data flow (monthly sync pipeline)

2. **DAG Architecture**
   - 5 DAG overview (purpose, schedule, dependencies)
   - Task breakdown (per DAG)
   - Dependency graph (monthly_sync → transformations → quality → MV refresh → features)

3. **Database Architecture**
   - Two PostgreSQL instances (airflow-metadata vs ntsb-data)
   - Schema diagram (11 production tables, 11 staging tables, 6 MVs)
   - Connection pooling strategy

4. **Security Architecture**
   - Secrets management (Airflow Connections, .env variables)
   - Database access control (user roles, permissions)
   - Network security (Docker network isolation)

### 7.4 Documentation Timeline

**Week 11**: Developer guide (2 days)
**Week 11**: Operator guide (1 day)
**Week 12**: Architecture docs (1 day)
**Week 12**: README updates (0.5 days)

**Total Documentation**: ~8,000 words across 4 files

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **NTSB website changes download URL** | Medium | High | Monitor for 404 errors, alert immediately, update URL in DAG config |
| **NTSB changes MDB schema** | Low | High | Schema validation step in DAG, alert on unexpected columns |
| **Database connection pool exhaustion** | Medium | Medium | Increase pool size (10 → 20), use PgBouncer if needed |
| **Disk space fills up** | Medium | High | Retention policy (delete files >6 months), alert at 80% disk usage |
| **Airflow scheduler crashes mid-DAG** | Low | Medium | Tasks are idempotent, safe to retry from any point |
| **Network failures during download** | High | Low | Retries (3x with backoff), download to temp location first |
| **Docker container out-of-memory** | Low | Medium | Allocate 16 GB RAM (current: 8 GB), monitor with `docker stats` |

**Monitoring Mitigations**:
- **Disk space**: Alert at 80% usage (`df -h /opt/airflow/data`)
- **Memory**: Alert at 90% usage (`docker stats --no-stream`)
- **Website changes**: Validate response headers before download
- **Schema drift**: Compare extracted CSV columns to expected schema

### 8.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Maintenance burden (solo developer)** | High | Medium | Comprehensive documentation, simple architecture (no K8s) |
| **Learning curve (Airflow expertise)** | Medium | Low | 2-week ramp-up period, Astronomer tutorials, active community |
| **Debugging complexity (failed DAG runs)** | Medium | Medium | Verbose logging, Airflow UI log viewer, local testing environment |
| **On-call responsibility (24/7?)** | Medium | High | Monthly DAGs run at 2 AM, failures can wait until morning |
| **Burnout (too many alerts)** | Medium | Medium | Alert fatigue mitigation: CRITICAL only to #ntsb-alerts, INFO to dashboard |

**Operational Best Practices**:
- **Runbooks**: Document common failure scenarios + solutions
- **Blameless Postmortems**: After incidents, document root cause + prevention
- **Vacation Coverage**: Pause non-critical DAGs, document emergency procedures
- **Automation First**: Avoid manual interventions, automate fixes in DAG code

### 8.3 Data Quality Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Bad data loaded to production** | Medium | High | Staging tables + validation before merge, rollback capability |
| **Duplicate events after deduplication** | Low | Medium | Unit tests verify dedup logic, data quality checks post-load |
| **Schema drift (NTSB adds new columns)** | Medium | Low | Alert on unexpected columns, manual review before load |
| **Corrupted MDB file** | Low | High | File size validation, unzip test, mdbtools error handling |
| **Missing data (incomplete monthly update)** | Medium | Medium | Row count comparison (current month vs previous), alert on >10% drop |

**Data Quality Gates**:
1. **Pre-Load**: File size, unzip test, mdbtools extraction
2. **Staging**: Row count validation, schema comparison
3. **Pre-Merge**: Deduplication check, foreign key validation
4. **Post-Load**: Data quality DAG (orphaned records, NULL checks, range validation)

**Rollback Procedure**:
```bash
# 1. Stop monthly_sync_dag
airflow dags pause monthly_sync_dag

# 2. Restore database from last good backup
pg_restore -d ntsb_aviation /backups/ntsb_aviation_2025-10-01.dump

# 3. Delete bad load_tracking entry
psql -d ntsb_aviation -c "DELETE FROM load_tracking WHERE load_completed_at > '2025-11-01';"

# 4. Clear staging tables
psql -d ntsb_aviation -c "SELECT clear_all_staging();"

# 5. Resume DAG (after fix)
airflow dags unpause monthly_sync_dag
```

### 8.4 Risk Matrix

```
         Impact →
         Low    Medium    High
       ┌──────┬─────────┬──────────┐
High   │      │ Network │ Bad Data │
       │      │ Failures│ Loaded   │
L      ├──────┼─────────┼──────────┤
i   Med│ Learn│ Debug   │ Website  │
k      │ Curve│Complexity│ Changes │
e      ├──────┼─────────┼──────────┤
l   Low│      │ OOM     │ Schema   │
h      │      │ Errors  │ Drift    │
o      └──────┴─────────┴──────────┘
o
d
```

**Priority**: High Impact + High Likelihood = **Bad Data Loaded** (mitigate with staging + validation)

---

## 9. 12-Week Implementation Timeline

### Week 1-2: Airflow Infrastructure Setup

**Week 1: Docker Compose + Hello World**
- **Day 1-2**: Docker Compose setup (postgres-airflow, postgres-ntsb, redis, webserver, scheduler, worker)
  - Write `docker-compose.yml` (150 lines)
  - Configure `.env` file (20 variables)
  - Test startup: `docker-compose up -d`
  - Access UI: http://localhost:8080 (admin/admin)
- **Day 3**: Hello World DAG
  - Create `dags/hello_world_dag.py` (50 lines)
  - Test local PostgreSQL connection
  - Verify task execution, logs
- **Day 4-5**: NTSB database connection
  - Add `ntsb_postgres` connection in Airflow UI
  - Test query: `SELECT COUNT(*) FROM events;`
  - Configure secrets (Fernet key, SMTP, Slack)

**Week 2: First ETL DAG (monthly_sync_dag.py)**
- **Day 1-2**: Tasks 1-3 (check website, download, validate)
  - Write `check_ntsb_website()` (50 lines)
  - Write `download_ntsb_data()` (80 lines)
  - Test with sample avall.zip (100 MB)
- **Day 3-4**: Tasks 4-6 (extract, load staging, merge)
  - Write `extract_tables_to_csv` (Bash, 30 lines)
  - Write `load_to_staging()` (100 lines)
  - Write merge SQL (50 lines)
- **Day 5**: Tasks 7-9 (update tracking, cleanup, notify)
  - Write `update_tracking` SQL (20 lines)
  - Write `send_success_notification()` (60 lines)
  - Test end-to-end with 100 sample events

**Deliverables**:
- Docker Compose stack (running)
- `monthly_sync_dag.py` (250-300 lines)
- 1 successful test run

---

### Week 3-4: Remaining 4 DAGs

**Week 3: DAGs 2-3 (transformation, quality)**
- **Day 1-2**: `data_transformation_dag.py` (200-250 lines)
  - Aircraft normalization SQL (50 lines)
  - Geocoding function (80 lines)
  - Severity classification SQL (30 lines)
  - Weather extraction (40 lines)
- **Day 3-4**: `quality_check_dag.py` (150-200 lines)
  - Integrate `validate_data.sql` (existing)
  - Write validation report generator (100 lines)
  - Test with bad data (trigger failures)
- **Day 5**: Integration testing
  - Run monthly_sync → transformation → quality (sequential)
  - Verify data flow, logs, alerts

**Week 4: DAGs 4-5 (MV refresh, features)**
- **Day 1-2**: `mv_refresh_dag.py` (100-150 lines)
  - 6 refresh tasks (CONCURRENTLY)
  - Metadata logging (refresh times, row counts)
  - Test concurrent refresh
- **Day 3-4**: `feature_engineering_dag.py` (200-250 lines)
  - Temporal features (pandas, 80 lines)
  - Spatial features (PostGIS SQL, 50 lines)
  - Text features (TF-IDF, 70 lines)
- **Day 5**: Performance testing
  - Benchmark all 5 DAGs
  - Optimize slow tasks
  - Document timings

**Deliverables**:
- 5 production DAGs (1,000-1,150 lines total)
- All DAGs tested individually
- Integration test (monthly pipeline)

---

### Week 5-6: Monitoring & Alerting

**Week 5: Slack + Email Alerts**
- **Day 1-2**: Slack integration
  - Create Slack App + webhook
  - Write `send_slack_alert_critical()` (80 lines)
  - Write `send_slack_success()` (40 lines)
  - Test with manual DAG failures
- **Day 3-4**: Email integration
  - Configure Gmail SMTP (App Password)
  - Write `send_email_alert_critical()` (60 lines)
  - HTML email templates (100 lines)
  - Test email delivery
- **Day 5**: Alert templates
  - Standardize message formats
  - Add to all 5 DAGs (callbacks)
  - Test CRITICAL/WARNING/INFO levels

**Week 6: Metrics Dashboard (Streamlit)**
- **Day 1-2**: Dashboard development
  - DAG run metrics (success rate, duration)
  - Data quality metrics (validation results)
  - Database size trends
- **Day 3**: Dashboard deployment
  - Containerize Streamlit app
  - Add to `docker-compose.yml`
  - Configure auto-refresh (5 min)
- **Day 4-5**: Dashboard enhancements
  - Task duration heatmap
  - Alert history table
  - Export metrics (CSV, JSON)

**Deliverables**:
- Slack + Email alerting (working)
- Streamlit dashboard (http://localhost:8501)
- 5 DAGs with callbacks (notifications on success/failure)

---

### Week 7-10: PRE1982 Integration

**Week 7: Schema Mapping**
- **Day 1-2**: Analyze PRE1982.MDB structure
  - Extract schema (mdb-schema)
  - Identify mappings (tblFirstHalf → events, aircraft, etc.)
  - Review `ref_docs/codman.pdf` (coding system)
- **Day 3-4**: Create mapping tables
  - `pre1982_code_mappings` (100+ rows)
  - Test lookup logic
  - Document decisions in `docs/PRE1982_SCHEMA_MAPPING.md`
- **Day 5**: Prototype transformation
  - Extract 100 sample events
  - Transform to modern schema (pandas)
  - Validate output

**Week 8: ETL Development**
- **Day 1-2**: Extract + Transform (Python)
  - `extract_pre1982()` (50 lines)
  - `transform_pre1982()` (250 lines)
  - Handle DMS coordinates, 2-digit years, coded fields
- **Day 3-4**: Load to production
  - `load_pre1982()` (100 lines)
  - Deduplication logic (ev_id collision handling)
  - Test with 1,000 events
- **Day 5**: DAG integration
  - Create `pre1982_integration_dag.py` (400-500 lines)
  - Test end-to-end (100 events)

**Week 9: Testing + Validation**
- **Day 1-2**: Full load test
  - Load all ~87,000 PRE1982 events
  - Monitor performance, errors
  - Verify date range (1962-1981)
- **Day 3-4**: Data quality validation
  - Run quality_check_dag
  - Verify no orphaned records
  - Compare row counts (staging vs production)
- **Day 5**: User acceptance testing
  - Query PRE1982 data (sample queries)
  - Verify materialized views updated
  - Document limitations

**Week 10: Performance + Documentation**
- **Day 1-2**: Performance optimization
  - Identify slow transformations
  - Batch processing (10K events at a time)
  - Reduce memory usage
- **Day 3-4**: Documentation
  - `docs/PRE1982_INTEGRATION.md` (2,000 words)
  - Schema mapping table
  - Known issues, limitations
- **Day 5**: Final testing
  - Regression tests (ensure avall.mdb still works)
  - Integration test (all 6 DAGs)

**Deliverables**:
- `pre1982_integration_dag.py` (400-500 lines)
- 87,000 additional events (1962-1981)
- Database coverage: 1962-2025 (63 years)

---

### Week 11-12: Documentation + Testing

**Week 11: Comprehensive Documentation**
- **Day 1-2**: Developer guide
  - `docs/AIRFLOW_DEVELOPER_GUIDE.md` (3,000 words)
  - Local setup, DAG development, testing, debugging
- **Day 3**: Operator guide
  - `docs/AIRFLOW_OPERATIONS_GUIDE.md` (2,500 words)
  - Deployment, monitoring, operations, incident response
- **Day 4**: Architecture docs
  - `docs/AIRFLOW_ARCHITECTURE.md` (2,500 words)
  - System diagram, DAG architecture, database architecture
- **Day 5**: README updates
  - Update main `README.md` (quick start, Airflow section)
  - Update `QUICKSTART_POSTGRESQL.md` (Airflow integration)

**Week 12: Final Testing + Sprint Report**
- **Day 1**: Integration testing
  - Run all 5 DAGs sequentially
  - Run all 5 DAGs in parallel (stress test)
  - Verify no conflicts, errors
- **Day 2**: Performance validation
  - Benchmark monthly_sync_dag (target: <10 min)
  - Benchmark all DAGs (total: <30 min)
  - Document actual vs target times
- **Day 3**: Regression testing
  - Test existing functionality (manual ETL still works)
  - Test database queries (MVs, joins)
  - Test rollback procedures
- **Day 4**: Sprint 3 completion report
  - `SPRINT_3_COMPLETION_REPORT.md` (3,000 words)
  - Deliverables, metrics, lessons learned
  - Next steps (Phase 2, Sprint 4)
- **Day 5**: Handoff + celebration
  - Demo to stakeholders
  - Knowledge transfer (if team)
  - Retrospective (what went well, what to improve)

**Deliverables**:
- 8,000 words documentation (4 files)
- `SPRINT_3_COMPLETION_REPORT.md`
- Production-ready Airflow pipeline

---

## 10. Success Criteria & Metrics

### 10.1 Functional Success Criteria

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| **Automated Monthly Sync** | 100% hands-off | Verify Dec 2025 data loads without manual intervention |
| **5 Production DAGs** | All operational | `airflow dags list` shows 5 DAGs, all enabled |
| **PRE1982 Integration** | 87K events loaded | `SELECT COUNT(*) FROM events WHERE ev_id LIKE 'PRE1982_%'` |
| **Data Coverage** | 1962-2025 (63 years) | `SELECT MIN(ev_year), MAX(ev_year) FROM events` → 1962, 2025 |
| **Zero Duplicates** | 0 duplicate events | `SELECT COUNT(*) - COUNT(DISTINCT ev_id) FROM events` → 0 |
| **Alerting** | Slack + Email working | Trigger failure, verify notifications received |
| **Dashboard** | Real-time metrics | Access http://localhost:8501, verify data refresh |

### 10.2 Performance Metrics

| Metric | Target | Measured (Sprint 3 End) |
|--------|--------|-------------------------|
| **Monthly Sync Duration** | <10 minutes | TBD |
| **Transformation Duration** | <5 minutes | TBD |
| **Quality Check Duration** | <2 minutes | TBD |
| **MV Refresh Duration** | <1 minute | TBD |
| **Feature Engineering Duration** | <10 minutes | TBD |
| **Total Pipeline Duration** | <30 minutes | TBD |
| **DAG Success Rate (30 days)** | >99% | TBD |
| **Alert Latency** | <5 minutes | TBD |

### 10.3 Quality Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| **Code Coverage** | >80% | `pytest --cov=dags --cov-report=term` |
| **Data Quality Checks** | 100% pass rate | `quality_check_dag` 0 failures |
| **Orphaned Records** | 0 | `validate_data.sql` foreign key checks |
| **Documentation Completeness** | 100% | All sections in developer/operator/architecture guides |
| **Zero Manual Interventions** | 3 months | Track manual fixes, aim for 0 by Month 3 |

### 10.4 Success Dashboard (Sprint 3 End)

**Expected Metrics**:
```
📊 NTSB ETL Pipeline - Sprint 3 Success Metrics

Functional:
✅ 5 Production DAGs (monthly_sync, transformation, quality, mv_refresh, features)
✅ PRE1982 Integration: 87,000 events (1962-1981)
✅ Database Coverage: 1962-2025 (63 years)
✅ Zero Duplicate Events
✅ Automated Monthly Sync (December 2025)
✅ Slack + Email Alerting
✅ Streamlit Dashboard

Performance:
- Monthly Sync: 8m 32s (Target: <10min) ✅
- Total Pipeline: 24m 15s (Target: <30min) ✅
- DAG Success Rate: 100% (Target: >99%) ✅

Quality:
- Code Coverage: 84% (Target: >80%) ✅
- Data Quality Checks: 0 failures (Target: 0) ✅
- Orphaned Records: 0 (Target: 0) ✅
- Documentation: 8,200 words (Target: 8,000) ✅

Next Steps: Phase 2 (ML/AI Features) - Sprint 4
```

---

## 11. Budget & Resource Requirements

### 11.1 Hardware Costs

**Local Deployment** (Chosen):
- **Existing Hardware**: $0 (use current machine)
- **Upgrade (if needed)**: $200-500 (16 GB RAM, 500 GB SSD)
- **Total Year 1**: $0-500 (one-time)

**Cloud Deployment** (Alternative):
- **AWS EC2 (t3.xlarge)**: $120/month × 12 = $1,440/year
- **Google Cloud Composer**: $400/month × 12 = $4,800/year
- **Total Year 1**: $1,440-4,800 (recurring)

**Decision**: Local deployment saves $1,440-4,800/year

### 11.2 Software Costs

**All Open Source** (Free):
- Apache Airflow (Apache 2.0 License)
- PostgreSQL (PostgreSQL License)
- Docker (Apache 2.0)
- Python (PSF License)
- Streamlit (Apache 2.0)

**Optional Services**:
- **Slack**: Free tier (10K messages/month) - Sufficient
- **Gmail**: Free (15 GB storage) - Sufficient
- **GitHub**: Free tier (public repos) - Sufficient

**Total Software Costs**: $0/year

### 11.3 Time Investment

**Sprint 3 Development** (12 weeks):
- **Week 1-2**: Infrastructure (80 hours)
- **Week 3-4**: DAG development (80 hours)
- **Week 5-6**: Monitoring (80 hours)
- **Week 7-10**: PRE1982 integration (160 hours)
- **Week 11-12**: Documentation + testing (80 hours)
- **Total**: 480 hours (~12 weeks full-time)

**At $50/hour contractor rate**: $24,000 labor cost
**At solo developer (learning)**: Free (but slower, 16-20 weeks)

### 11.4 Ongoing Maintenance

**Monthly Time** (Post-Sprint 3):
- **Monitoring**: 1 hour/month (check dashboard, review alerts)
- **Updates**: 2 hours/quarter (Airflow upgrades, DAG improvements)
- **Incident Response**: 0-4 hours/month (avg 1 hour/month)
- **Total**: ~2 hours/month (~24 hours/year)

**At $50/hour**: $1,200/year maintenance cost

### 11.5 Total Cost of Ownership (3 Years)

| Component | Year 1 | Year 2 | Year 3 | Total |
|-----------|--------|--------|--------|-------|
| **Hardware** | $500 | $0 | $0 | $500 |
| **Software** | $0 | $0 | $0 | $0 |
| **Development** | $24,000 | $0 | $0 | $24,000 |
| **Maintenance** | $1,200 | $1,200 | $1,200 | $3,600 |
| **Total** | **$25,700** | **$1,200** | **$1,200** | **$28,100** |

**ROI Calculation**:
- **Manual ETL Time Saved**: 3 hours/month × 36 months = 108 hours
- **At $50/hour**: $5,400 saved
- **Net Cost**: $28,100 - $5,400 = **$22,700** (infrastructure investment)

**Strategic Value** (Non-Monetary):
- Foundation for ML/AI (Phase 3)
- Scalability (add new data sources easily)
- Data freshness (24-hour latency vs weeks)
- Reproducibility (version-controlled transformations)

---

## 12. Decision Log

### Decision 1: Docker Compose vs Kubernetes

**Date**: November 6, 2025
**Decision**: Use Docker Compose (local deployment)

**Options Considered**:
1. **Docker Compose** (Chosen)
2. Kubernetes (local minikube)
3. Cloud Managed (GCP Composer, AWS MWAA)

**Rationale**:
- **Cost**: $0 vs $1,440-4,800/year
- **Complexity**: Simple `docker-compose up` vs K8s learning curve
- **Scalability**: Sufficient for 30K events/month
- **Migration Path**: Can migrate to K8s later (DAGs portable)

**Trade-offs**:
- **Pro**: Low cost, simple, fast setup
- **Con**: No auto-scaling, manual infrastructure management

**Research**:
- Reddit: "K8s is overkill for small projects" ([source](https://www.reddit.com/r/dataengineering/comments/weykug/))
- Airflow docs: "Docker Compose not production-ready, but works for small scale" ([source](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/))

---

### Decision 2: LocalExecutor vs CeleryExecutor

**Date**: November 6, 2025
**Decision**: Use LocalExecutor

**Options Considered**:
1. **LocalExecutor** (Chosen)
2. CeleryExecutor
3. KubernetesExecutor

**Rationale**:
- **LocalExecutor**: Tasks run as subprocesses, no Redis needed
- **CeleryExecutor**: Requires Redis/RabbitMQ, distributed workers
- **KubernetesExecutor**: Requires K8s cluster

**Trade-offs**:
- **Pro**: Simpler (1 less container), sufficient parallelism (10 tasks)
- **Con**: No distributed workers (but not needed for monthly 30K loads)

---

### Decision 3: Custom SQL vs Great Expectations for Validation

**Date**: November 6, 2025
**Decision**: Use Custom SQL Checks (reuse `validate_data.sql`)

**Options Considered**:
1. **Custom SQL** (Chosen)
2. Great Expectations
3. Pandera

**Rationale**:
- **Custom SQL**: Already written (384 lines), fast, simple
- **Great Expectations**: 20K+ LOC dependency, steep learning curve
- **Pandera**: Pandas-focused, not SQL-first

**Research Finding** ([source](https://aeturrell.com/blog/posts/the-data-validation-landscape-in-2025/)):
> "For serious production automation, Great Expectations is better, but for SQL-heavy workflows, custom checks are simpler."

**Trade-offs**:
- **Pro**: Zero new dependencies, fast execution, already tested
- **Con**: No pretty HTML reports, no data profiling UI

---

### Decision 4: CONCURRENTLY vs Non-CONCURRENTLY MV Refresh

**Date**: November 6, 2025
**Decision**: Use REFRESH MATERIALIZED VIEW CONCURRENTLY

**Rationale**:
- **CONCURRENTLY**: 2-3x slower, but view remains available (no lock)
- **Non-CONCURRENTLY**: Faster, but view locked (blocks queries)

**For monthly updates**: Availability > Speed

**Trade-off**:
- **Pro**: Zero downtime (users can query during refresh)
- **Con**: ~60 seconds vs ~30 seconds (acceptable for monthly cadence)

---

### Decision 5: PRE1982 Integration Now vs Later

**Date**: November 6, 2025
**Decision**: Integrate PRE1982 in Sprint 3 (Weeks 7-10)

**Options Considered**:
1. **Sprint 3** (Chosen)
2. Defer to Sprint 4 (Phase 2)
3. Never (accept 2000-2025 coverage)

**Rationale**:
- **Historical Completeness**: 63 years (1962-2025) vs 26 years (2000-2025)
- **Effort**: 8-16 hours (manageable in 4-week window)
- **Value**: Fills 20-year gap, enables long-term trend analysis

**Trade-off**:
- **Pro**: Complete dataset, valuable for Phase 3 ML/AI
- **Con**: Complex ETL (denormalized schema), potential data quality issues

---

### Decision 6: Slack vs PagerDuty for Alerting

**Date**: November 6, 2025
**Decision**: Use Slack + Email (no PagerDuty)

**Rationale**:
- **Slack**: Free, already used by team, good for INFO/WARNING alerts
- **Email**: Free, good for CRITICAL alerts
- **PagerDuty**: $19/month/user, overkill for monthly DAGs

**When to Upgrade**: If/when running hourly DAGs or 24/7 operations

**Trade-off**:
- **Pro**: $0 cost vs $228/year
- **Con**: No on-call rotation, no escalation policies (acceptable for monthly cadence)

---

### Decision 7: Streamlit vs Grafana for Dashboard

**Date**: November 6, 2025
**Decision**: Use Streamlit

**Options Considered**:
1. **Streamlit** (Chosen)
2. Grafana
3. Custom Flask/React app

**Rationale**:
- **Streamlit**: Python-native, easy (250 lines), fast development
- **Grafana**: Requires Prometheus/InfluxDB, more complex setup
- **Custom**: 2-3x more code (500+ lines)

**Trade-off**:
- **Pro**: Fast development (1-2 days), Python-friendly
- **Con**: Less polished UI than Grafana, no advanced alerting

---

## 13. Appendix: Research Findings

### 13.1 Airflow Best Practices (2025)

**Source**: [Airflow Summit 2025](https://airflowsummit.org/sessions/2025/apache-airflow-3-0-bad-vs-best-practices-in-production/)

**Key Takeaways**:
1. **DAG Naming**: Use `{project}_{purpose}_dag` pattern (e.g., `ntsb_monthly_sync_dag`)
2. **Folder Structure**: Separate `dags/`, `plugins/`, `config/`, `tests/`
3. **Connection Management**: Use Airflow Connections (encrypted), not hardcoded secrets
4. **Idempotency**: All tasks must be safe to retry (delete-write pattern)
5. **Small DAGs**: 5-10 tasks per DAG (modularity > monolith)

### 13.2 ETL Error Handling Patterns

**Source**: [Prefect Blog](https://www.prefect.io/blog/the-importance-of-idempotent-data-pipelines-for-resilience)

**Key Takeaways**:
1. **Idempotency**: "Operation can be applied multiple times without changing result"
2. **Safe Retries**: Failed operations can retry without data duplication
3. **Checkpoint/Restart**: Save progress periodically (e.g., staging tables)
4. **Delete-Write Pattern**: Delete existing data before writing (prevents duplicates)

**Example**:
```python
# NOT idempotent (re-run duplicates data)
INSERT INTO events VALUES (...)

# Idempotent (re-run safe)
DELETE FROM events WHERE ev_id = 'X';
INSERT INTO events VALUES (ev_id='X', ...)
```

### 13.3 PostgreSQL COPY Performance

**Source**: [CYBERTEC](https://www.cybertec-postgresql.com/en/bulk-load-performance-in-postgresql/)

**Key Takeaways**:
1. **COPY is 10-100x faster than INSERT**
2. **Disable indexes during load**: Drop → COPY → Recreate (2x speedup)
3. **Increase work_mem**: Faster sorting for index creation
4. **COPY FREEZE**: Skips vacuuming (for initial loads only)

**Benchmark** (1M rows):
- Single INSERT: 600 seconds
- Batch INSERT (1000 rows): 60 seconds
- COPY: 6 seconds

**Conclusion**: Use COPY for all bulk loads (existing `load_with_staging.py` already does this)

### 13.4 Materialized View Refresh Strategies

**Source**: [PostgreSQL Docs](https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html)

**CONCURRENTLY vs Non-CONCURRENTLY**:

| Feature | CONCURRENTLY | Non-CONCURRENTLY |
|---------|--------------|------------------|
| **Speed** | 2-3x slower | Fast |
| **Locking** | No lock (available) | Exclusive lock (unavailable) |
| **Requirements** | Unique index | None |
| **Disk Space** | 2x (temp table) | 1x |

**When to Use**:
- **CONCURRENTLY**: Production (high availability, monthly updates)
- **Non-CONCURRENTLY**: Development, nightly batch jobs (downtime acceptable)

### 13.5 Data Validation Libraries Comparison

**Source**: [endjin Blog](https://endjin.com/blog/2023/03/a-look-into-pandera-and-great-expectations-for-data-validation)

| Feature | Pandera | Great Expectations | Custom SQL |
|---------|---------|-------------------|------------|
| **Learning Curve** | Low | High | Very Low |
| **Setup Complexity** | Simple | Complex | None |
| **Pandas Integration** | Excellent | Good | N/A |
| **SQL Integration** | Poor | Good | Excellent |
| **Reporting** | Basic | Rich (HTML) | Custom |
| **Dependencies** | 5 MB | 50 MB | 0 |

**Recommendation**:
- **Pandera**: ML pipelines (pandas DataFrames)
- **Great Expectations**: Complex ETL (multiple sources, HTML reports)
- **Custom SQL**: SQL-first workflows (this project)

### 13.6 Legacy Data Migration Strategies

**Source**: [Datafold Blog](https://www.datafold.com/blog/legacy-data-migration)

**Best Practices**:
1. **Phased Approach**: Pilot (100 events) → Test (1K) → Full (87K)
2. **Schema Mapping**: Create explicit mapping table (source → target)
3. **Data Profiling**: Analyze data quality before migration (NULL rates, duplicates)
4. **Validation**: Compare source row counts to target (detect data loss)
5. **Rollback Plan**: Backup before migration, test restore

**For PRE1982**:
- Create `pre1982_code_mappings` table (100+ rows)
- Test with 100 events first
- Document limitations (missing fields, data quality issues)

### 13.7 Airflow Cost Analysis (Cloud vs Local)

**Source**: [Pythian Blog](https://www.pythian.com/blog/technical-track/google-cloud-composer-costs-and-performance)

**Google Cloud Composer Costs**:
- **Small Environment** (1 scheduler, 3 workers): $300-400/month
- **Medium Environment** (2 schedulers, 6 workers): $600-800/month
- **Storage (GCS)**: $0.02/GB/month
- **Data Transfer**: $0.12/GB

**Local Deployment Costs**:
- **Hardware**: $500 one-time (16 GB RAM, 500 GB SSD)
- **Electricity**: ~$10/month (100W × 24h × 30d × $0.12/kWh)
- **Internet**: $0 (existing connection)

**3-Year TCO**:
- **Cloud Composer**: $300/month × 36 months = $10,800
- **Local**: $500 + ($10/month × 36 months) = $860

**Savings**: $9,940 over 3 years

---

## Conclusion

This implementation plan provides a comprehensive roadmap for Sprint 3 (12 weeks) to build a production-ready Apache Airflow ETL pipeline for the NTSB Aviation Database.

**Key Milestones**:
- **Week 2**: First DAG operational (monthly_sync_dag)
- **Week 4**: All 5 DAGs operational
- **Week 6**: Monitoring & alerting complete
- **Week 10**: PRE1982 integrated (63 years data coverage)
- **Week 12**: Full documentation + production deployment

**Expected Outcomes**:
- 100% automated monthly updates
- 63 years historical data (1962-2025)
- Zero manual intervention
- <30 minute total pipeline duration
- Comprehensive monitoring & alerting
- Foundation for Phase 3 ML/AI features

**Next Steps After Sprint 3**:
- **Phase 2, Sprint 4**: Advanced analytics (predictive modeling, trend forecasting)
- **Phase 2, Sprint 5**: Public API (REST endpoints for data access)
- **Phase 3**: Machine learning (accident prediction, causal inference)

---

**End of Sprint 3 Implementation Plan**

**Version**: 1.0.0
**Date**: November 6, 2025
**Author**: NTSB Data Pipeline Team
**Status**: Ready for Implementation
