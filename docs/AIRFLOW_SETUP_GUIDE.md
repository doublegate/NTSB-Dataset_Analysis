# Apache Airflow Setup Guide

**NTSB Aviation Database ETL Pipeline**

**Version**: 1.0.0
**Last Updated**: 2025-11-06
**Sprint**: Phase 1 Sprint 3 Week 1

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)
8. [Reference](#reference)

---

## Introduction

This guide covers the setup and operation of Apache Airflow for the NTSB Aviation Database ETL pipeline. Airflow orchestrates automated data loading, transformation, quality checks, and materialized view updates.

### Architecture

**Deployment**: Docker Compose (LocalExecutor)
**Components**:
- **PostgreSQL (Airflow Metadata)**: Stores DAG runs, task instances, connections
- **Airflow Webserver**: Web UI on port 8080
- **Airflow Scheduler**: Schedules and monitors DAG runs

**Executor**: LocalExecutor (no Celery/Redis needed for single-machine deployment)

---

## Prerequisites

### System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Linux, macOS, Windows with WSL2 | Tested on Arch Linux |
| **RAM** | 4GB minimum, 8GB+ recommended | Airflow + PostgreSQL + DAG execution |
| **CPU** | 2+ cores recommended | Parallel task execution |
| **Disk** | 10GB+ free space | Airflow logs, database growth |
| **Docker** | 24.0+ | Docker Compose V2 |
| **PostgreSQL** | 15+ | Host database must be accessible from Docker |

### Software Dependencies

```bash
# Check Docker and Docker Compose
docker --version         # Should be 24.0+
docker compose version   # Should be 2.0+

# Check PostgreSQL (NTSB database)
psql --version          # Should be 15+
psql -U parobek -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"
```

### PostgreSQL Network Configuration

**CRITICAL**: Airflow runs in Docker and must connect to the host PostgreSQL database. This requires PostgreSQL to accept connections from Docker containers.

#### Check Current Configuration

```bash
# Check listen_addresses
psql -U parobek -d ntsb_aviation -c "SHOW listen_addresses;"
```

If this shows `localhost`, PostgreSQL is NOT accessible from Docker.

#### Configure PostgreSQL for Docker Access

**Option 1: Listen on All Interfaces** (Recommended for development)

Edit `/etc/postgresql/*/main/postgresql.conf` (or equivalent):

```ini
# Change from:
listen_addresses = 'localhost'

# To:
listen_addresses = '*'
```

Edit `/etc/postgresql/*/main/pg_hba.conf`:

```text
# Add this line (before other entries):
host    ntsb_aviation    parobek    172.17.0.0/16    trust
```

Restart PostgreSQL:

```bash
# Arch Linux / systemd
sudo systemctl restart postgresql

# Debian/Ubuntu
sudo systemctl restart postgresql@15-main

# macOS
brew services restart postgresql@15
```

**Option 2: Listen on Docker Bridge Only** (More secure)

Edit `postgresql.conf`:

```ini
listen_addresses = 'localhost,172.17.0.1'
```

Then add the same `pg_hba.conf` entry and restart.

#### Verify Docker Can Connect

```bash
# Get Docker bridge IP
ip addr show docker0 | grep "inet " | awk '{print $2}' | cut -d/ -f1

# Test connection from Docker
docker run --rm postgres:15 psql -h 172.17.0.1 -U parobek -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"
```

**Expected**: Should return `92771` (or current event count).

---

## Installation

### Step 1: Clone Repository (if not already)

```bash
git clone https://github.com/YOUR_USERNAME/NTSB_Datasets.git
cd NTSB_Datasets
```

### Step 2: Verify Directory Structure

```bash
ls -la airflow/
# Should show:
# - docker-compose.yml
# - dags/
# - logs/
# - plugins/
# - config/
```

### Step 3: Create .env File

The `.env` file is gitignored and contains sensitive credentials.

**If it doesn't exist**, create `airflow/.env`:

```bash
cat > airflow/.env << 'EOF'
# Airflow UID (must match your user ID)
AIRFLOW_UID=50000

# Airflow Web UI Credentials
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# NTSB PostgreSQL Database Connection
NTSB_DB_HOST=172.17.0.1
NTSB_DB_PORT=5432
NTSB_DB_NAME=ntsb_aviation
NTSB_DB_USER=parobek
NTSB_DB_PASSWORD=

# Project Directory
AIRFLOW_PROJ_DIR=.
EOF
```

**Set AIRFLOW_UID** to your user ID:

```bash
# Linux/macOS
id -u  # Should be 1000 or similar
# Update AIRFLOW_UID in .env to match

# Or automatically:
sed -i "s/AIRFLOW_UID=50000/AIRFLOW_UID=$(id -u)/" airflow/.env
```

**Update NTSB_DB_USER** if different from `parobek`:

```bash
sed -i "s/NTSB_DB_USER=parobek/NTSB_DB_USER=$USER/" airflow/.env
```

### Step 4: Initialize Airflow

```bash
cd airflow/

# Initialize Airflow database (first time only)
docker compose up airflow-init

# Expected output:
# - Database tables created
# - Admin user created
# - "airflow-init-1 exited with code 0"
```

**Common Issues**:
- **Permission errors**: Check AIRFLOW_UID matches your user
- **Port conflicts**: Ensure ports 8080 and 5433 are free
- **Memory warnings**: Allocate more RAM to Docker

### Step 5: Start Airflow

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# Expected output:
# postgres-airflow: Up (healthy)
# airflow-webserver: Up (healthy)
# airflow-scheduler: Up
```

### Step 6: Access Web UI

Open http://localhost:8080 in your browser.

**Login**:
- Username: `airflow` (from .env)
- Password: `airflow` (from .env)

**Change password in production!**

---

## Configuration

### Database Connection

Airflow needs a connection to the NTSB PostgreSQL database.

#### Via Web UI (Recommended)

1. Navigate to **Admin** → **Connections**
2. Click **+** (Add a new record)
3. Fill in details:
   - **Connection Id**: `ntsb_aviation_db`
   - **Connection Type**: `Postgres`
   - **Host**: `172.17.0.1` (Docker bridge IP)
   - **Schema**: `ntsb_aviation`
   - **Login**: `parobek` (your username)
   - **Password**: (leave empty if passwordless)
   - **Port**: `5432`
4. Click **Test** (if enabled)
5. Click **Save**

#### Via Airflow CLI

```bash
docker compose exec airflow-scheduler airflow connections add \
    ntsb_aviation_db \
    --conn-type postgres \
    --conn-host 172.17.0.1 \
    --conn-schema ntsb_aviation \
    --conn-login parobek \
    --conn-port 5432
```

#### Via Environment Variables

Add to `docker-compose.yml` under `x-airflow-common.environment`:

```yaml
AIRFLOW_CONN_NTSB_AVIATION_DB: postgres://parobek@172.17.0.1:5432/ntsb_aviation
```

### Email Alerts (Optional)

For production, configure email notifications:

Edit `airflow/.env`:

```bash
# Gmail example (requires app password)
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_STARTTLS=True
AIRFLOW__SMTP__SMTP_SSL=False
AIRFLOW__SMTP__SMTP_USER=your-email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=your-app-password
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_MAIL_FROM=your-email@gmail.com
```

Update `default_args` in DAGs:

```python
default_args = {
    'owner': 'airflow',
    'email': ['your-email@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}
```

### Slack Alerts (Optional)

1. Create Slack incoming webhook: https://api.slack.com/messaging/webhooks
2. Add to `airflow/.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

3. Use SlackWebhookOperator in DAGs:

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

slack_alert = SlackWebhookOperator(
    task_id='slack_alert',
    http_conn_id='slack_webhook',
    message='DAG {{ dag.dag_id }} completed successfully!',
)
```

---

## Usage

### Starting and Stopping Airflow

```bash
cd airflow/

# Start all services
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes (DESTRUCTIVE - loses metadata)
docker compose down -v

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver
```

### Triggering DAGs

#### Via Web UI

1. Navigate to **DAGs** page
2. Find your DAG (e.g., `hello_world`)
3. **Unpause** DAG (toggle switch on left)
4. Click **▶️ Play** button on right
5. Select **Trigger DAG**
6. Monitor in **Graph View** or **Grid View**

#### Via CLI

```bash
# List all DAGs
docker compose exec airflow-scheduler airflow dags list

# Trigger a DAG
docker compose exec airflow-scheduler airflow dags trigger hello_world

# List DAG runs
docker compose exec airflow-scheduler airflow dags list-runs -d hello_world

# Check task status
docker compose exec airflow-scheduler airflow tasks states-for-dag-run \
    hello_world manual__2025-11-07T03:52:24+00:00
```

### Viewing Logs

#### Via Web UI

1. Click on DAG
2. Click on DAG run (Grid View)
3. Click on task instance
4. View **Log** tab

#### Via CLI

```bash
# Task logs are in airflow/logs/
ls -la airflow/logs/dag_id=hello_world/

# View specific task log
cat airflow/logs/dag_id=hello_world/run_id=*/task_id=query_ntsb_events/attempt=1.log
```

### Monitoring

#### Web UI Dashboard

- **DAGs**: Overview of all DAGs, status, schedules
- **Grid View**: Visual task status per DAG run
- **Graph View**: DAG structure and dependencies
- **Calendar View**: Historical run success/failures
- **Task Duration**: Performance over time
- **Gantt Chart**: Task execution timeline

#### CLI Monitoring

```bash
# Service health
docker compose ps

# Scheduler health
docker compose exec airflow-scheduler airflow jobs check

# Database health
docker compose exec postgres-airflow pg_isready
```

---

## Development Workflow

### Creating a New DAG

1. **Create DAG file** in `airflow/dags/`

```python
# airflow/dags/my_new_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='my_new_dag',
    default_args=default_args,
    description='Description of my DAG',
    schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    start_date=datetime(2025, 11, 6),
    catchup=False,
    tags=['ntsb', 'etl'],
) as dag:

    task = BashOperator(
        task_id='my_task',
        bash_command='echo "Hello from my DAG!"',
    )
```

2. **Wait for Airflow to detect** (30 seconds by default)

3. **Verify DAG parses**:

```bash
docker compose exec airflow-scheduler airflow dags list | grep my_new_dag
```

4. **Test DAG**:

```bash
# Test task individually
docker compose exec airflow-scheduler airflow tasks test my_new_dag my_task 2025-11-06

# Test entire DAG
docker compose exec airflow-scheduler airflow dags test my_new_dag 2025-11-06
```

5. **Unpause and trigger** via Web UI

### Best Practices

#### DAG Design

- **Idempotent tasks**: Tasks should produce same result when run multiple times
- **Atomic tasks**: Each task does one thing well
- **Clear dependencies**: Use `>>` or `set_upstream()`/`set_downstream()`
- **Proper retries**: Set `retries` and `retry_delay` in `default_args`
- **Timeouts**: Set `execution_timeout` to prevent hanging
- **Catchup**: Set `catchup=False` unless backfilling needed

#### Code Quality

```python
# Good: Specific task IDs
task_load_data = PythonOperator(task_id='load_data', ...)
task_validate = PythonOperator(task_id='validate', ...)

# Bad: Generic task IDs
task1 = PythonOperator(task_id='task1', ...)
task2 = PythonOperator(task_id='task2', ...)

# Good: Descriptive function names
def load_avall_mdb_to_staging():
    """Load avall.mdb to staging tables."""
    pass

# Bad: Vague function names
def process():
    pass
```

#### Logging

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook

def my_function():
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Starting data load...")

    try:
        # Do work
        logger.info(f"Loaded {row_count} rows")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
```

#### Error Handling

```python
from airflow.exceptions import AirflowException

def my_task_function():
    try:
        # Do work
        if not results:
            raise AirflowException("No data returned from query")
    except Exception as e:
        # Log error
        # Clean up resources
        raise
```

### Testing

#### Unit Testing DAGs

```python
# tests/dags/test_hello_world_dag.py
from airflow.models import DagBag

def test_dag_loads():
    """Test that hello_world DAG loads without errors."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert dagbag.import_errors == {}
    assert 'hello_world' in dagbag.dags

def test_task_count():
    """Test that hello_world has correct number of tasks."""
    dagbag = DagBag(dag_folder='dags/')
    dag = dagbag.get_dag('hello_world')
    assert len(dag.tasks) == 5
```

#### Integration Testing

```bash
# Test DAG without triggering
docker compose exec airflow-scheduler airflow dags test hello_world 2025-11-06

# Test single task
docker compose exec airflow-scheduler airflow tasks test hello_world print_hello 2025-11-06
```

---

## Troubleshooting

### Common Issues

#### 1. PostgreSQL Connection Refused

**Symptom**:
```
psycopg2.OperationalError: connection to server at "172.17.0.1", port 5432 failed: Connection refused
```

**Cause**: PostgreSQL not listening on Docker bridge interface.

**Solution**: Configure PostgreSQL as described in [Prerequisites](#postgresql-network-configuration).

**Quick Test**:
```bash
# From host
psql -U parobek -d ntsb_aviation -c "SELECT 1;"  # Works

# From Docker
docker run --rm postgres:15 psql -h 172.17.0.1 -U parobek -d ntsb_aviation -c "SELECT 1;"  # Fails?
```

#### 2. Port 8080 Already in Use

**Symptom**:
```
Error: bind: address already in use
```

**Solution**: Change port in `docker-compose.yml`:

```yaml
airflow-webserver:
  ports:
    - "8081:8080"  # Changed from 8080:8080
```

Then access at http://localhost:8081.

#### 3. Permission Denied on dags/ Directory

**Symptom**:
```
Permission denied: '/opt/airflow/dags'
```

**Solution**: Fix ownership:

```bash
# Check AIRFLOW_UID
grep AIRFLOW_UID airflow/.env

# Fix ownership
sudo chown -R $(id -u):$(id -g) airflow/dags/
sudo chown -R $(id -u):$(id -g) airflow/logs/
sudo chown -R $(id -u):$(id -g) airflow/plugins/
```

#### 4. Scheduler Unhealthy

**Symptom**: `docker compose ps` shows scheduler as "unhealthy"

**Cause**: Healthcheck timeout (scheduler takes time to start)

**Solution**: Wait 1-2 minutes, or check logs:

```bash
docker compose logs airflow-scheduler | tail -50
```

Scheduler is healthy if you see:
```
INFO - Starting the scheduler
INFO - Processing each file at most -1 times
```

#### 5. DAG Not Appearing in UI

**Causes**:
- Syntax error in DAG file
- DAG folder not mounted
- Scheduler not scanning folder

**Solutions**:

```bash
# Check for syntax errors
python3 airflow/dags/my_dag.py

# Check DAG parsing
docker compose logs airflow-scheduler | grep "my_dag"

# List DAGs
docker compose exec airflow-scheduler airflow dags list

# Check volume mount
docker compose exec airflow-scheduler ls -la /opt/airflow/dags/
```

#### 6. Task Stuck in "running" State

**Causes**:
- Task hanging (no timeout)
- Database lock
- Executor issue

**Solutions**:

```bash
# Check task logs
docker compose logs airflow-scheduler | grep task_id

# Kill task
docker compose exec airflow-scheduler airflow tasks clear -t task_id hello_world

# Restart scheduler
docker compose restart airflow-scheduler
```

### Logs and Diagnostics

```bash
# All logs
docker compose logs

# Specific service
docker compose logs airflow-scheduler -f --tail=100

# Scheduler errors only
docker compose logs airflow-scheduler 2>&1 | grep ERROR

# Task logs
ls -la airflow/logs/dag_id=hello_world/
cat airflow/logs/dag_id=hello_world/run_id=*/task_id=*/attempt=1.log

# Database logs
docker compose logs postgres-airflow
```

### Reset Airflow

**DESTRUCTIVE**: This deletes all DAG runs, task instances, connections, and logs.

```bash
cd airflow/

# Stop services
docker compose down

# Remove volumes (metadata database)
docker volume rm postgres-airflow-data

# Remove logs
rm -rf logs/

# Re-initialize
docker compose up airflow-init

# Start services
docker compose up -d
```

---

## Reference

### Airflow CLI Commands

```bash
# DAGs
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-scheduler airflow dags show <dag_id>
docker compose exec airflow-scheduler airflow dags trigger <dag_id>
docker compose exec airflow-scheduler airflow dags pause <dag_id>
docker compose exec airflow-scheduler airflow dags unpause <dag_id>
docker compose exec airflow-scheduler airflow dags delete <dag_id>

# Tasks
docker compose exec airflow-scheduler airflow tasks list <dag_id>
docker compose exec airflow-scheduler airflow tasks test <dag_id> <task_id> <date>
docker compose exec airflow-scheduler airflow tasks clear <dag_id>

# Connections
docker compose exec airflow-scheduler airflow connections list
docker compose exec airflow-scheduler airflow connections add <conn_id> ...
docker compose exec airflow-scheduler airflow connections delete <conn_id>

# Users
docker compose exec airflow-scheduler airflow users list
docker compose exec airflow-scheduler airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Variables
docker compose exec airflow-scheduler airflow variables set <key> <value>
docker compose exec airflow-scheduler airflow variables get <key>
docker compose exec airflow-scheduler airflow variables list
```

### Web UI Navigation

| Page | Path | Description |
|------|------|-------------|
| **DAGs** | `/home` | List of all DAGs |
| **Grid View** | `/dags/<dag_id>/grid` | Task status per run |
| **Graph View** | `/dags/<dag_id>/graph` | DAG structure |
| **Calendar** | `/dags/<dag_id>/calendar` | Run history |
| **Task Duration** | `/dags/<dag_id>/duration` | Performance metrics |
| **Connections** | `/connection/list/` | Database connections |
| **Variables** | `/variable/list/` | Airflow variables |
| **Admin → Users** | `/users/list/` | User management |

### Docker Compose Services

| Service | Container Name | Port | Description |
|---------|----------------|------|-------------|
| **postgres-airflow** | postgres-airflow | 5433:5432 | Airflow metadata database |
| **airflow-webserver** | airflow-webserver | 8080:8080 | Web UI |
| **airflow-scheduler** | airflow-scheduler | - | DAG scheduler |
| **airflow-init** | airflow-init | - | One-time initialization |

### Default Credentials

| Component | Username | Password | Notes |
|-----------|----------|----------|-------|
| **Web UI** | airflow | airflow | Change in production |
| **PostgreSQL (Airflow)** | airflow | airflow | Internal only |
| **PostgreSQL (NTSB)** | parobek | (none) | Host database |

### Environment Variables

**Key variables in `airflow/.env`**:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRFLOW_UID` | 50000 | User ID for file ownership |
| `_AIRFLOW_WWW_USER_USERNAME` | airflow | Web UI username |
| `_AIRFLOW_WWW_USER_PASSWORD` | airflow | Web UI password |
| `NTSB_DB_HOST` | 172.17.0.1 | NTSB database host |
| `NTSB_DB_PORT` | 5432 | NTSB database port |
| `NTSB_DB_NAME` | ntsb_aviation | NTSB database name |
| `NTSB_DB_USER` | parobek | NTSB database user |

### Useful Links

- **Apache Airflow Documentation**: https://airflow.apache.org/docs/
- **Airflow Docker Documentation**: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
- **Airflow Best Practices**: https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- **Airflow PostgreSQL Provider**: https://airflow.apache.org/docs/apache-airflow-providers-postgres/
- **Docker Compose Reference**: https://docs.docker.com/compose/compose-file/

---

## Summary

This guide has covered:

✅ **Prerequisites**: Docker, PostgreSQL network configuration
✅ **Installation**: Docker Compose setup, initialization
✅ **Configuration**: Database connections, email/Slack alerts
✅ **Usage**: Starting/stopping, triggering DAGs, monitoring
✅ **Development**: Creating DAGs, best practices, testing
✅ **Troubleshooting**: Common issues and solutions
✅ **Reference**: CLI commands, Web UI, environment variables

**Next Steps**:
1. Complete Sprint 3 Week 2: Create first production DAG (`monthly_sync_dag.py`)
2. Set up automated testing for DAGs
3. Configure email/Slack notifications for production
4. Implement monitoring and alerting

**Support**:
- Check logs: `docker compose logs -f`
- Review DAG code: `airflow/dags/`
- Consult documentation: https://airflow.apache.org/docs/

---

**End of Airflow Setup Guide**
