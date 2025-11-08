# SPRINT 3 WEEK 3: Monitoring & Observability

**Project**: NTSB Aviation Accident Database
**Sprint**: Phase 1 Sprint 3 Week 3
**Created**: 2025-11-07
**Status**: 📋 PLANNED (Not Started)
**Estimated Effort**: 4-8 hours
**Priority**: HIGH (Production monitoring before December 1st)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Prerequisites](#2-prerequisites)
3. [Task Breakdown](#3-task-breakdown)
4. [Implementation Details](#4-implementation-details)
5. [Testing & Validation](#5-testing--validation)
6. [Success Criteria](#6-success-criteria)
7. [Deliverables](#7-deliverables)
8. [Risk Assessment](#8-risk-assessment)
9. [Timeline](#9-timeline)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Goal

Implement production-grade monitoring and alerting for the automated monthly NTSB data synchronization pipeline **before the first production run on December 1st, 2025**.

### 1.2 Context

**Current State**:
- ✅ Production DAG operational (baseline run: 1m 50s, 8/8 tasks SUCCESS)
- ✅ Scheduled for 1st of month, 2 AM
- ❌ No automated alerts for failures
- ❌ No real-time performance visibility
- ❌ No data quality anomaly detection
- ❌ Manual log inspection required

**Target State**:
- ✅ Slack/Email alerts for DAG failures (CRITICAL severity)
- ✅ Real-time metrics dashboard (DAG health, performance, data quality)
- ✅ Query performance monitoring (detect regressions)
- ✅ Data quality anomaly detection (missing data, outliers)
- ✅ Monthly run success tracking (historical trends)

### 1.3 Business Value

| Benefit | Impact |
|---------|--------|
| **Rapid Incident Response** | Know within 5 minutes if monthly sync fails (vs hours/days) |
| **Proactive Issue Detection** | Catch data quality issues before they propagate |
| **Performance Visibility** | Identify slow tasks for optimization |
| **Operational Confidence** | Peace of mind for hands-off automation |
| **Audit Trail** | Complete history of runs, metrics, and alerts |

### 1.4 Strategic Importance

This is the **final critical piece** for production readiness:
1. **Week 1** ✅ - Infrastructure (Docker, Airflow, connectivity)
2. **Week 2** ✅ - Production DAG (8 tasks, 7 bug fixes, baseline verified)
3. **Week 3** 📋 - **Monitoring** (alerts, dashboard, anomaly detection)

Without monitoring, the first production failure on December 1st could go unnoticed for days.

---

## 2. Prerequisites

### 2.1 Required Before Starting

✅ **Airflow Infrastructure**: Docker Compose running, Web UI accessible
✅ **Production DAG**: `monthly_sync_ntsb_data` baseline verified
✅ **Database Access**: PostgreSQL connection from Airflow containers
✅ **Git Repository**: All Sprint 3 Week 2 work committed

### 2.2 Optional (Can Be Set Up During Week 3)

🔧 **Slack Workspace**: For alert notifications (or use email only)
🔧 **SMTP Credentials**: For email alerts (Gmail App Password recommended)
🔧 **Grafana Instance**: For advanced metrics (optional, can use Streamlit)

### 2.3 Knowledge Requirements

- Basic Python (notification functions, dashboard code)
- Airflow callbacks (`on_failure_callback`, `on_success_callback`)
- SQL queries (metrics extraction from PostgreSQL)
- Streamlit basics (optional, for dashboard)

---

## 3. Task Breakdown

### 3.1 Phase 1: Alert Configuration (Estimated: 2-3 hours)

#### Task 1.1: Slack Integration ⭐ HIGH PRIORITY
**Goal**: Configure Slack webhook for critical DAG failure alerts

**Subtasks**:
- [ ] Create Slack workspace or use existing (1 min)
- [ ] Create Slack app and enable Incoming Webhooks (5 min)
- [ ] Create `#ntsb-alerts` channel (CRITICAL alerts)
- [ ] Create `#ntsb-etl` channel (INFO, SUCCESS messages)
- [ ] Copy webhook URL and store securely
- [ ] Add Airflow connection via Web UI (Connection ID: `slack_webhook`)
- [ ] Test webhook with curl command
- [ ] Install `apache-airflow-providers-slack` package in Airflow container

**Files to Create**:
- `airflow/plugins/notification_callbacks.py` (150-200 lines)
  - `send_slack_alert_critical(context)` - Critical failure notification
  - `send_slack_alert_warning(context)` - Warning notification
  - `send_slack_success(context)` - Success notification with metrics

**Acceptance Criteria**:
- ✅ Slack webhook responds with 200 OK
- ✅ Test message appears in `#ntsb-alerts` channel
- ✅ Message includes: DAG name, task name, error message, log URL

**Resources**:
- Slack API Documentation: https://api.slack.com/messaging/webhooks
- Airflow Slack Provider: https://airflow.apache.org/docs/apache-airflow-providers-slack/

---

#### Task 1.2: Email Integration ⭐ MEDIUM PRIORITY
**Goal**: Configure email notifications as backup/fallback to Slack

**Subtasks**:
- [ ] Generate Gmail App Password (if using Gmail)
- [ ] Add SMTP configuration to `airflow/.env`:
  ```bash
  AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
  AIRFLOW__SMTP__SMTP_STARTTLS=True
  AIRFLOW__SMTP__SMTP_SSL=False
  AIRFLOW__SMTP__SMTP_PORT=587
  AIRFLOW__SMTP__SMTP_MAIL_FROM=ntsb-alerts@gmail.com
  AIRFLOW__SMTP__SMTP_USER=ntsb-alerts@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD=<16-char-app-password>
  ```
- [ ] Restart Airflow services to pick up SMTP config
- [ ] Add email notification functions to `notification_callbacks.py`
- [ ] Test with Airflow's `airflow tasks test` command

**Files to Modify**:
- `airflow/.env` (add SMTP configuration)
- `airflow/plugins/notification_callbacks.py` (add email functions)

**Acceptance Criteria**:
- ✅ Test email received in inbox
- ✅ HTML formatting renders correctly
- ✅ Links to Airflow UI work (http://localhost:8080)

**Resources**:
- Gmail App Passwords: https://myaccount.google.com/apppasswords
- Airflow Email Config: https://airflow.apache.org/docs/apache-airflow/stable/howto/email-config.html

---

#### Task 1.3: Integrate Callbacks into Production DAG ⭐ HIGH PRIORITY
**Goal**: Add notification callbacks to `monthly_sync_ntsb_data` DAG

**Subtasks**:
- [ ] Import notification functions at top of `monthly_sync_dag.py`
- [ ] Add `on_failure_callback` to DAG default_args
- [ ] Add `on_success_callback` to final task only
- [ ] Add `on_retry_callback` for transient failures
- [ ] Test by intentionally failing a task (modify to `exit 1`)
- [ ] Verify alert received in Slack/Email within 30 seconds
- [ ] Revert test failure, verify production DAG still works

**Files to Modify**:
- `airflow/dags/monthly_sync_dag.py` (~10 lines changed)
  - Lines 50-60: Add callback imports
  - Lines 150-160: Add callbacks to default_args

**Example Code**:
```python
# At top of monthly_sync_dag.py
from notification_callbacks import (
    send_slack_alert_critical,
    send_slack_success,
)

# In default_args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_slack_alert_critical,  # NEW
}

# On final task only
send_success_notification = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_slack_success,  # MODIFIED
    provide_context=True,
)
```

**Acceptance Criteria**:
- ✅ Intentional failure triggers alert within 30 seconds
- ✅ Alert includes correct DAG/task names
- ✅ Log URL link works
- ✅ Success message sent after baseline run completes

---

### 3.2 Phase 2: Metrics Dashboard (Estimated: 2-3 hours)

#### Task 2.1: Database Metrics Views ⭐ MEDIUM PRIORITY
**Goal**: Create SQL views to aggregate metrics for dashboard

**Subtasks**:
- [ ] Create `scripts/create_monitoring_views.sql` (200-250 lines)
- [ ] View 1: `vw_dag_run_history` - Airflow DAG run metrics
  - Query Airflow metadata DB: `dag_run`, `task_instance`
  - Fields: dag_id, execution_date, state, duration, task_counts
- [ ] View 2: `vw_database_metrics` - Database size/row count trends
  - Query: `pg_database_size()`, `pg_stat_user_tables`
  - Fields: timestamp, table_name, row_count, table_size_mb
- [ ] View 3: `vw_data_quality_checks` - Validation results over time
  - Query: `events`, `aircraft`, etc. for NULL rates, outliers
  - Fields: check_name, check_result, timestamp
- [ ] View 4: `vw_query_performance` - Slow query detection
  - Query: `pg_stat_statements` (if enabled)
  - Fields: query_hash, avg_exec_time, calls, rows
- [ ] Test each view with `SELECT * FROM vw_* LIMIT 10;`
- [ ] Document view purposes in SQL comments

**Files to Create**:
- `scripts/create_monitoring_views.sql` (200-250 lines)

**Acceptance Criteria**:
- ✅ All 4 views created successfully
- ✅ Each view returns data (not empty)
- ✅ Views refresh quickly (<1 second)
- ✅ Documentation comments explain each view

**Example SQL**:
```sql
-- View 1: DAG Run History (requires connection to Airflow metadata DB)
CREATE OR REPLACE VIEW vw_dag_run_history AS
SELECT
    dag_id,
    execution_date,
    state,
    EXTRACT(EPOCH FROM (end_date - start_date)) as duration_seconds,
    run_type
FROM dag_run
WHERE dag_id = 'monthly_sync_ntsb_data'
ORDER BY execution_date DESC
LIMIT 100;

-- View 2: Database Growth Trends
CREATE OR REPLACE VIEW vw_database_metrics AS
SELECT
    NOW() as timestamp,
    schemaname,
    tablename,
    n_live_tup as row_count,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
```

---

#### Task 2.2: Streamlit Dashboard (Option A) ⭐ MEDIUM PRIORITY
**Goal**: Create interactive web dashboard for metrics visualization

**Subtasks**:
- [ ] Create `dashboard/` directory in project root
- [ ] Create `dashboard/requirements.txt`:
  ```
  streamlit==1.28.0
  psycopg2-binary==2.9.9
  pandas==2.1.3
  plotly==5.18.0
  ```
- [ ] Create `dashboard/app.py` (300-400 lines)
  - Page 1: DAG Run History (success rate, duration trends)
  - Page 2: Database Metrics (size, row counts, growth)
  - Page 3: Data Quality (validation checks, NULL rates)
  - Page 4: Query Performance (slow queries, execution times)
- [ ] Add connection to PostgreSQL (both ntsb_aviation and Airflow metadata)
- [ ] Add auto-refresh every 60 seconds
- [ ] Style with Streamlit themes
- [ ] Test locally: `streamlit run dashboard/app.py`
- [ ] Document access URL in README

**Files to Create**:
- `dashboard/app.py` (300-400 lines)
- `dashboard/requirements.txt` (4 lines)
- `dashboard/README.md` (setup instructions)

**Acceptance Criteria**:
- ✅ Dashboard loads without errors
- ✅ All 4 pages render correctly
- ✅ Charts display real data (not empty)
- ✅ Auto-refresh works (updates every 60s)
- ✅ Accessible at http://localhost:8501

**Resources**:
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Charts: https://plotly.com/python/

---

#### Task 2.3: Grafana Dashboard (Option B - OPTIONAL) ⚙️ LOW PRIORITY
**Goal**: Alternative to Streamlit, more powerful but requires Docker setup

**Subtasks**:
- [ ] Add Grafana service to `airflow/docker-compose.yml`
- [ ] Configure PostgreSQL data source (both databases)
- [ ] Import pre-built dashboard JSON (or create from scratch)
- [ ] Configure dashboard panels:
  - Panel 1: DAG success rate (gauge)
  - Panel 2: DAG duration (time series)
  - Panel 3: Database size growth (time series)
  - Panel 4: Row count by table (bar chart)
- [ ] Set up auto-refresh (every 5 minutes)
- [ ] Export dashboard JSON to `dashboard/grafana/ntsb_dashboard.json`

**Files to Create/Modify**:
- `airflow/docker-compose.yml` (add grafana service)
- `dashboard/grafana/ntsb_dashboard.json` (dashboard config)

**Acceptance Criteria**:
- ✅ Grafana accessible at http://localhost:3000
- ✅ PostgreSQL data sources connected
- ✅ All 4 panels display data
- ✅ Dashboard auto-refreshes every 5 minutes

**Note**: This is OPTIONAL. Choose either Streamlit (2.2) OR Grafana (2.3), not both.

---

### 3.3 Phase 3: Data Quality Monitoring (Estimated: 1-2 hours)

#### Task 3.1: Anomaly Detection Script ⭐ MEDIUM PRIORITY
**Goal**: Detect unusual patterns in loaded data (missing fields, outliers)

**Subtasks**:
- [ ] Create `scripts/detect_anomalies.py` (200-250 lines)
- [ ] Check 1: Missing critical fields (ev_id, ev_date, dec_latitude, dec_longitude)
- [ ] Check 2: Outlier detection (coordinates outside bounds, invalid dates)
- [ ] Check 3: Statistical anomalies (sudden drop in events loaded)
- [ ] Check 4: Referential integrity (orphaned child records)
- [ ] Check 5: Duplicate detection (same ev_id loaded multiple times)
- [ ] Return JSON report with pass/fail for each check
- [ ] Log results to `anomaly_detection_log` table

**Files to Create**:
- `scripts/detect_anomalies.py` (200-250 lines)
- `scripts/create_anomaly_log_table.sql` (30 lines)

**Acceptance Criteria**:
- ✅ Script runs without errors
- ✅ All 5 checks execute successfully
- ✅ Results logged to database
- ✅ Failures trigger Slack alert (integrate with Task 1.1)

**Example Python**:
```python
# detect_anomalies.py
def check_missing_critical_fields(conn):
    """Check for NULL values in critical fields."""
    query = """
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE ev_id IS NULL) as missing_ev_id,
        COUNT(*) FILTER (WHERE ev_date IS NULL) as missing_ev_date,
        COUNT(*) FILTER (WHERE dec_latitude IS NULL) as missing_lat,
        COUNT(*) FILTER (WHERE dec_longitude IS NULL) as missing_lon
    FROM events
    WHERE ev_date >= CURRENT_DATE - INTERVAL '1 month';
    """
    result = pd.read_sql(query, conn)

    # Anomaly if >1% missing
    total = result['total'].iloc[0]
    missing = result['missing_ev_id'].iloc[0] + result['missing_ev_date'].iloc[0]

    return {
        'check': 'missing_critical_fields',
        'passed': missing < total * 0.01,
        'details': result.to_dict('records')[0]
    }
```

---

#### Task 3.2: Integrate Anomaly Detection into DAG ⭐ HIGH PRIORITY
**Goal**: Run anomaly detection automatically after data load

**Subtasks**:
- [ ] Add new task to `monthly_sync_dag.py`:
  ```python
  detect_anomalies = PythonOperator(
      task_id='detect_anomalies',
      python_callable=run_anomaly_detection,
      provide_context=True,
  )
  ```
- [ ] Position between `validate_data_quality` and `refresh_materialized_views`
- [ ] If anomalies detected, trigger WARNING alert (not CRITICAL)
- [ ] Store results in XCom for dashboard display
- [ ] Test with baseline run

**Files to Modify**:
- `airflow/dags/monthly_sync_dag.py` (~30 lines added)

**Acceptance Criteria**:
- ✅ Task appears in DAG graph
- ✅ Task executes successfully in baseline run
- ✅ Anomaly results visible in dashboard
- ✅ WARNING alert sent if anomalies detected

---

### 3.4 Phase 4: Query Performance Monitoring (Estimated: 1 hour)

#### Task 4.1: Enable pg_stat_statements ⭐ MEDIUM PRIORITY
**Goal**: Track slow queries for performance regression detection

**Subtasks**:
- [ ] Enable `pg_stat_statements` extension in PostgreSQL:
  ```sql
  CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
  ```
- [ ] Add to `postgresql.conf`:
  ```
  shared_preload_libraries = 'pg_stat_statements'
  pg_stat_statements.track = all
  ```
- [ ] Restart PostgreSQL to pick up config changes
- [ ] Verify extension loaded:
  ```sql
  SELECT * FROM pg_stat_statements LIMIT 10;
  ```

**Files to Modify**:
- PostgreSQL config (depends on installation method)
- `scripts/schema.sql` (add extension creation)

**Acceptance Criteria**:
- ✅ Extension created successfully
- ✅ Query statistics being collected
- ✅ View shows recent queries with timing

**Resources**:
- pg_stat_statements Documentation: https://www.postgresql.org/docs/current/pgstatstatements.html

---

#### Task 4.2: Query Performance Dashboard ⭐ LOW PRIORITY
**Goal**: Add query performance panel to Streamlit dashboard

**Subtasks**:
- [ ] Query `pg_stat_statements` for slow queries (>1 second)
- [ ] Display in dashboard table with columns:
  - Query (truncated to 100 chars)
  - Calls (execution count)
  - Avg Time (milliseconds)
  - Total Time (milliseconds)
  - Rows (rows returned)
- [ ] Highlight queries that have regressed (>2x slower than baseline)
- [ ] Add export to CSV button

**Files to Modify**:
- `dashboard/app.py` (~50 lines added)

**Acceptance Criteria**:
- ✅ Page shows top 20 slowest queries
- ✅ Regressions highlighted in red
- ✅ CSV export works

---

### 3.5 Phase 5: Documentation & Testing (Estimated: 1 hour)

#### Task 5.1: Monitoring Setup Guide ⭐ HIGH PRIORITY
**Goal**: Document how to set up monitoring for GitHub users

**Subtasks**:
- [ ] Create `docs/MONITORING_SETUP_GUIDE.md` (400-500 lines)
- [ ] Section 1: Overview (what's monitored, why it matters)
- [ ] Section 2: Slack Setup (step-by-step with screenshots)
- [ ] Section 3: Email Setup (Gmail App Password guide)
- [ ] Section 4: Dashboard Access (Streamlit or Grafana)
- [ ] Section 5: Anomaly Detection (how to interpret alerts)
- [ ] Section 6: Troubleshooting (common issues and solutions)
- [ ] Section 7: Customization (how to modify alerts, add checks)

**Files to Create**:
- `docs/MONITORING_SETUP_GUIDE.md` (400-500 lines)

**Acceptance Criteria**:
- ✅ Clear step-by-step instructions
- ✅ Screenshots where helpful
- ✅ Troubleshooting section covers common issues
- ✅ Links to external resources (Slack API, Gmail, etc.)

---

#### Task 5.2: Sprint 3 Week 3 Completion Report ⭐ MEDIUM PRIORITY
**Goal**: Document what was accomplished, metrics, and lessons learned

**Subtasks**:
- [ ] Create `docs/SPRINT_3_WEEK_3_COMPLETION_REPORT.md` (600-800 lines)
- [ ] Section 1: Executive Summary (goals, achievements)
- [ ] Section 2: Deliverables (files created, LOC metrics)
- [ ] Section 3: Testing Results (alert tests, dashboard tests)
- [ ] Section 4: Performance Metrics (alert latency, dashboard load time)
- [ ] Section 5: Lessons Learned (what worked, what didn't)
- [ ] Section 6: Next Steps (Sprint 4 recommendations)

**Files to Create**:
- `docs/SPRINT_3_WEEK_3_COMPLETION_REPORT.md` (600-800 lines)

**Acceptance Criteria**:
- ✅ Comprehensive documentation of all work
- ✅ Metrics included (LOC, test results, performance)
- ✅ Honest assessment of lessons learned
- ✅ Clear recommendations for next steps

---

#### Task 5.3: Update Project Documentation ⭐ MEDIUM PRIORITY
**Goal**: Update README, CHANGELOG, CLAUDE.local.md with Week 3 changes

**Subtasks**:
- [ ] Update `README.md` (add Monitoring section, dashboard screenshots)
- [ ] Update `CHANGELOG.md` (add Sprint 3 Week 3 entry)
- [ ] Update `CLAUDE.local.md` (add monitoring file inventory, metrics)
- [ ] Update `airflow/docker-compose.yml` comments (document monitoring services)

**Files to Modify**:
- `README.md` (~50 lines added)
- `CHANGELOG.md` (~30 lines added)
- `CLAUDE.local.md` (~100 lines added)

**Acceptance Criteria**:
- ✅ README has Monitoring section with dashboard links
- ✅ CHANGELOG documents all monitoring features
- ✅ CLAUDE.local.md reflects current monitoring state

---

## 4. Implementation Details

### 4.1 Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Alerting** | Slack Webhooks | Free, real-time, mobile notifications |
| **Email** | Gmail SMTP | Free, reliable, fallback to Slack |
| **Dashboard** | Streamlit | Python-native, fast development, interactive |
| **Metrics Storage** | PostgreSQL Views | No new database, leverage existing |
| **Query Monitoring** | pg_stat_statements | Built-in, zero overhead, comprehensive |
| **Anomaly Detection** | Python Script | Flexible, easy to customize, integrates with Airflow |

### 4.2 Alert Severity Matrix

| Severity | Conditions | Slack | Email | Dashboard |
|----------|-----------|-------|-------|-----------|
| **CRITICAL** | DAG failure, data corruption, load tracking error | ✅ Immediate | ✅ Immediate | 🔴 Red banner |
| **WARNING** | Data quality issues, slow queries, missing optional fields | ✅ Batched | ❌ No | 🟡 Yellow banner |
| **INFO** | Successful runs, metrics summary | ❌ No | ❌ No | 🟢 Green status |

### 4.3 Notification Channels

**Slack Channels**:
- `#ntsb-alerts` - CRITICAL and WARNING alerts only
- `#ntsb-etl` - INFO messages, success notifications

**Email Recipients**:
- Primary: Data team lead
- CC: Database admin (for infrastructure issues)

### 4.4 Monitoring Frequency

| Metric | Collection Frequency | Alert Threshold |
|--------|---------------------|-----------------|
| DAG Run Status | Real-time (Airflow callbacks) | Immediate on failure |
| Database Size | Every 5 minutes | >2 GB (80% of expected max) |
| Row Counts | Every 5 minutes | <90K events (unexpected drop) |
| Query Performance | Continuous (pg_stat_statements) | >2x baseline duration |
| Data Quality | After each load | >1% critical field NULLs |

---

## 5. Testing & Validation

### 5.1 Alert Testing

#### Test 1: Critical Failure Alert
**Procedure**:
1. Modify `monthly_sync_dag.py` to intentionally fail:
   ```python
   load_new_data = BashOperator(
       task_id='load_new_data',
       bash_command='exit 1',  # INTENTIONAL FAILURE
   )
   ```
2. Trigger DAG: `docker compose exec webserver airflow dags trigger monthly_sync_ntsb_data`
3. Wait for task to fail
4. Verify Slack alert received within 30 seconds
5. Verify email alert received within 1 minute
6. Revert change and verify production DAG works

**Expected Results**:
- ✅ Slack message in `#ntsb-alerts` with error details
- ✅ Email received with HTML formatting
- ✅ Log URL links work
- ✅ Production DAG still passes after revert

---

#### Test 2: Success Notification
**Procedure**:
1. Run baseline DAG to completion
2. Verify success message in `#ntsb-etl` channel
3. Check message includes metrics (events loaded, duration)

**Expected Results**:
- ✅ Success message received within 1 minute of completion
- ✅ Metrics accurate (compare to Airflow UI)

---

### 5.2 Dashboard Testing

#### Test 3: Dashboard Load and Refresh
**Procedure**:
1. Start dashboard: `streamlit run dashboard/app.py`
2. Navigate to http://localhost:8501
3. Verify all 4 pages load without errors
4. Wait 60 seconds, verify auto-refresh updates data
5. Check browser console for JavaScript errors

**Expected Results**:
- ✅ All pages load in <5 seconds
- ✅ Charts display real data (not empty)
- ✅ Auto-refresh updates timestamps
- ✅ No console errors

---

#### Test 4: Metrics Accuracy
**Procedure**:
1. Query database directly for row counts
2. Compare to dashboard displayed values
3. Query Airflow metadata for DAG run duration
4. Compare to dashboard chart

**Expected Results**:
- ✅ Row counts match within 1%
- ✅ DAG durations match exactly
- ✅ Database sizes match within 5 MB

---

### 5.3 Anomaly Detection Testing

#### Test 5: Missing Field Detection
**Procedure**:
1. Manually insert event with NULL ev_id:
   ```sql
   INSERT INTO events (ev_id, ev_date) VALUES (NULL, CURRENT_DATE);
   ```
2. Run anomaly detection: `python scripts/detect_anomalies.py`
3. Verify WARNING alert sent to Slack
4. Delete test record

**Expected Results**:
- ✅ Anomaly detected (missing_critical_fields check fails)
- ✅ Slack alert sent with details
- ✅ Result logged to anomaly_detection_log table

---

#### Test 6: Outlier Detection
**Procedure**:
1. Manually insert event with invalid coordinates:
   ```sql
   INSERT INTO events (ev_id, ev_date, dec_latitude, dec_longitude)
   VALUES ('TEST999', CURRENT_DATE, 999.0, 999.0);
   ```
2. Run anomaly detection
3. Verify outlier caught
4. Delete test record

**Expected Results**:
- ✅ Outlier detected (coordinates outside bounds)
- ✅ Alert sent with lat/lon values
- ✅ Result logged

---

### 5.4 Query Performance Testing

#### Test 7: Slow Query Detection
**Procedure**:
1. Run intentionally slow query:
   ```sql
   SELECT * FROM events e
   CROSS JOIN aircraft a
   WHERE e.ev_id = a.ev_id AND 1=1;  -- No index use
   ```
2. Check pg_stat_statements for query entry
3. Verify query appears in dashboard "Slow Queries" panel
4. Check if query time is highlighted (should be red if >2x baseline)

**Expected Results**:
- ✅ Query logged in pg_stat_statements
- ✅ Query visible in dashboard
- ✅ Execution time accurate
- ✅ Regression highlighting works

---

## 6. Success Criteria

### 6.1 Functional Requirements

- ✅ Slack alerts working for CRITICAL failures
- ✅ Email alerts working as fallback
- ✅ Dashboard accessible and displaying real-time data
- ✅ Anomaly detection running after every load
- ✅ Query performance monitoring enabled
- ✅ All alerts tested and verified

### 6.2 Performance Requirements

- ✅ Alert latency <30 seconds from failure
- ✅ Dashboard page load <5 seconds
- ✅ Dashboard auto-refresh working (60 seconds)
- ✅ Anomaly detection <10 seconds execution time
- ✅ Query monitoring <1% CPU overhead

### 6.3 Documentation Requirements

- ✅ Monitoring setup guide complete (400+ lines)
- ✅ Sprint 3 Week 3 completion report (600+ lines)
- ✅ README updated with monitoring section
- ✅ CHANGELOG updated with Week 3 entry
- ✅ CLAUDE.local.md updated with monitoring state

### 6.4 Quality Requirements

- ✅ All Python code passes ruff check and format
- ✅ All SQL validated with `psql -f script.sql`
- ✅ No credentials committed to git (.env gitignored)
- ✅ All tests passed (7 test procedures)

---

## 7. Deliverables

### 7.1 Code Files

| File | Lines | Description |
|------|-------|-------------|
| `airflow/plugins/notification_callbacks.py` | 150-200 | Slack/email notification functions |
| `scripts/create_monitoring_views.sql` | 200-250 | Database metrics views |
| `scripts/detect_anomalies.py` | 200-250 | Data quality anomaly detection |
| `scripts/create_anomaly_log_table.sql` | 30 | Anomaly log table schema |
| `dashboard/app.py` | 300-400 | Streamlit metrics dashboard |
| `dashboard/requirements.txt` | 4 | Dashboard Python dependencies |

**Total Code**: ~900-1,300 lines

### 7.2 Configuration Files

| File | Changes | Description |
|------|---------|-------------|
| `airflow/.env` | +10 lines | SMTP configuration |
| `airflow/dags/monthly_sync_dag.py` | +40 lines | Callback integration, anomaly detection task |
| `airflow/docker-compose.yml` | +0 lines | No changes (unless Grafana added) |

### 7.3 Documentation Files

| File | Lines | Description |
|------|-------|-------------|
| `docs/MONITORING_SETUP_GUIDE.md` | 400-500 | Complete monitoring setup guide |
| `docs/SPRINT_3_WEEK_3_COMPLETION_REPORT.md` | 600-800 | Sprint completion report |
| `README.md` | +50 | Monitoring section, dashboard links |
| `CHANGELOG.md` | +30 | Sprint 3 Week 3 entry |
| `CLAUDE.local.md` | +100 | Monitoring state, file inventory |
| `dashboard/README.md` | 50 | Dashboard setup instructions |

**Total Documentation**: ~1,200-1,500 lines

### 7.4 Database Objects

- `vw_dag_run_history` - Airflow DAG run metrics view
- `vw_database_metrics` - Database size/row count view
- `vw_data_quality_checks` - Validation results view
- `vw_query_performance` - Slow query view
- `anomaly_detection_log` - Table for logging anomaly checks

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Slack webhook fails** | LOW | MEDIUM | Implement email fallback, test both channels |
| **SMTP credentials exposed** | MEDIUM | HIGH | Use .env, add to .gitignore, document App Passwords |
| **Dashboard crashes** | LOW | LOW | Add error handling, graceful degradation |
| **pg_stat_statements overhead** | LOW | MEDIUM | Monitor CPU, can disable if needed |
| **False positive anomalies** | MEDIUM | LOW | Tune thresholds after initial runs |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Alert fatigue** | MEDIUM | MEDIUM | Use severity levels, batch WARNINGs |
| **Dashboard not monitored** | HIGH | LOW | Send daily summary to Slack |
| **Alerts missed** | LOW | HIGH | Test alerts monthly, use multiple channels |
| **Configuration drift** | MEDIUM | MEDIUM | Document all config in .env, version control |

### 8.3 Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Scope creep** | MEDIUM | MEDIUM | Stick to core tasks, defer Grafana if time-constrained |
| **Testing takes longer** | MEDIUM | LOW | Prioritize critical tests (alert functionality) |
| **Documentation incomplete** | LOW | MEDIUM | Start documentation early, write as you code |

---

## 9. Timeline

### 9.1 Recommended Schedule (4-8 hours)

#### Session 1: Alert Setup (2-3 hours)
- Task 1.1: Slack integration (1 hour)
- Task 1.2: Email integration (45 minutes)
- Task 1.3: Integrate into DAG (30 minutes)
- Task 1.3: Test alerts (30 minutes)

#### Session 2: Dashboard & Monitoring (2-3 hours)
- Task 2.1: Create monitoring views (1 hour)
- Task 2.2: Build Streamlit dashboard (1.5 hours)
- Task 3.1: Anomaly detection script (45 minutes)
- Task 3.2: Integrate into DAG (30 minutes)

#### Session 3: Query Monitoring & Documentation (1-2 hours)
- Task 4.1: Enable pg_stat_statements (15 minutes)
- Task 4.2: Query performance dashboard (45 minutes)
- Task 5.1: Monitoring setup guide (1 hour)
- Task 5.2: Completion report (1 hour)
- Task 5.3: Update project docs (30 minutes)

### 9.2 Critical Path

1. ⚠️ **BLOCKER**: Slack/Email alerts must work before December 1st
2. ⚠️ **BLOCKER**: Anomaly detection integrated into DAG
3. 🔧 **NICE-TO-HAVE**: Dashboard (can launch later)
4. 🔧 **NICE-TO-HAVE**: Query performance monitoring (can enable later)

### 9.3 Minimum Viable Monitoring (if time-constrained)

If only 2-3 hours available, prioritize:
1. Task 1.1 + 1.3: Slack alerts for failures ⭐ CRITICAL
2. Task 3.1 + 3.2: Anomaly detection ⭐ CRITICAL
3. Task 5.1: Basic setup guide ⭐ IMPORTANT

Defer to later:
- Email alerts (Task 1.2) - Slack sufficient for now
- Dashboard (Task 2.2) - Airflow UI provides basic visibility
- Query monitoring (Task 4.x) - Not urgent, can add later

---

## 10. References

### 10.1 Internal Documentation

- [Sprint 3 Implementation Plan](SPRINT_3_IMPLEMENTATION_PLAN.md) - Section 4: Monitoring & Alerting Design
- [Airflow Setup Guide](../docs/AIRFLOW_SETUP_GUIDE.md) - Docker Compose configuration
- [Sprint 3 Week 2 Report](../docs/SPRINT_3_WEEK_2_COMPLETION_REPORT.md) - Current DAG state

### 10.2 External Resources

**Airflow**:
- [Callbacks Documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/callbacks.html)
- [Slack Provider](https://airflow.apache.org/docs/apache-airflow-providers-slack/stable/index.html)
- [Email Configuration](https://airflow.apache.org/docs/apache-airflow/stable/howto/email-config.html)

**Slack**:
- [Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Block Kit Builder](https://app.slack.com/block-kit-builder) - Design rich messages

**Gmail**:
- [App Passwords](https://myaccount.google.com/apppasswords) - Generate SMTP credentials

**PostgreSQL**:
- [pg_stat_statements](https://www.postgresql.org/docs/current/pgstatstatements.html) - Query monitoring
- [System Catalogs](https://www.postgresql.org/docs/current/catalogs.html) - Metadata queries

**Streamlit**:
- [Documentation](https://docs.streamlit.io/)
- [Gallery](https://streamlit.io/gallery) - Example dashboards

**Grafana** (Optional):
- [Getting Started](https://grafana.com/docs/grafana/latest/getting-started/)
- [PostgreSQL Data Source](https://grafana.com/docs/grafana/latest/datasources/postgres/)

---

## Completion Checklist

Before marking Sprint 3 Week 3 complete, verify:

### Functional Completeness
- [ ] Slack alerts tested and working for failures
- [ ] Email alerts configured (or documented as deferred)
- [ ] Dashboard accessible and displaying data
- [ ] Anomaly detection integrated into DAG
- [ ] Query monitoring enabled (pg_stat_statements)
- [ ] All 7 test procedures executed and passed

### Code Quality
- [ ] All Python code passes ruff check and format
- [ ] All SQL validated with syntax check
- [ ] No credentials committed (.env gitignored)
- [ ] Code commented and self-documenting

### Documentation
- [ ] Monitoring setup guide complete (400+ lines)
- [ ] Sprint 3 Week 3 completion report (600+ lines)
- [ ] README updated with monitoring section
- [ ] CHANGELOG updated
- [ ] CLAUDE.local.md updated

### Git Repository
- [ ] All files committed with meaningful messages
- [ ] No merge conflicts
- [ ] .gitignore updated for monitoring artifacts
- [ ] Ready for user review and push

### Operational Readiness
- [ ] Production DAG still works after monitoring integration
- [ ] Baseline run completes successfully
- [ ] Alerts received within 30 seconds of failure
- [ ] Dashboard loads in <5 seconds

---

## Notes & Decisions

### Decision 1: Streamlit vs Grafana

**Choice**: Streamlit
**Rationale**:
- Python-native (no new technology)
- Faster development (<2 hours)
- No additional Docker services
- Easy to customize and extend
- Grafana can be added later if needed

### Decision 2: Slack vs PagerDuty

**Choice**: Slack (+ Email fallback)
**Rationale**:
- Free tier sufficient for monthly updates
- Already used by most teams
- Mobile notifications built-in
- PagerDuty overkill for non-critical pipeline

### Decision 3: Alert Severity Levels

**Choice**: 3 levels (CRITICAL, WARNING, INFO)
**Rationale**:
- CRITICAL: immediate action required (DAG failure)
- WARNING: investigate but not urgent (data quality)
- INFO: for audit trail only (success notifications)

### Decision 4: Monitoring Frequency

**Choice**: Real-time for alerts, 5-minute batch for metrics
**Rationale**:
- Alerts need immediate response
- Metrics can be delayed (not time-critical)
- Reduces database query load

---

**Sprint 3 Week 3 TODO - Version 1.0**
**Created**: 2025-11-07
**Last Updated**: 2025-11-07
**Status**: 📋 PLANNED (Ready to Begin)
