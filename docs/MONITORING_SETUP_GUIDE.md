# NTSB Aviation Database - Monitoring Setup Guide

**Created**: 2025-11-07
**Sprint**: Phase 1 Sprint 3 Week 3
**Status**: Production-Ready
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Slack Integration](#3-slack-integration)
4. [Email Alerts](#4-email-alerts)
5. [Monitoring Views](#5-monitoring-views)
6. [Anomaly Detection](#6-anomaly-detection)
7. [Dashboard Access](#7-dashboard-access)
8. [Troubleshooting](#8-troubleshooting)
9. [Customization](#9-customization)

---

## 1. Overview

### 1.1 What's Monitored

The NTSB Aviation Database monitoring system provides:

- **Real-time Alerts**: Slack and email notifications for DAG failures
- **Data Quality Checks**: Automated anomaly detection after each data load
- **Performance Metrics**: Database size, row counts, and query performance
- **Health Dashboard**: SQL views for monitoring system health

### 1.2 Alert Severity Levels

| Severity | Description | Example | Slack | Email |
|----------|-------------|---------|-------|-------|
| **CRITICAL** | DAG failure, data corruption | Task fails, duplicate events found | ✅ Immediate | ✅ Immediate |
| **WARNING** | Data quality issues | Missing coordinates, orphaned records | ✅ Batched | ❌ Optional |
| **INFO** | Normal operations | Successful DAG completion | ✅ Optional | ❌ No |

### 1.3 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Apache Airflow DAG                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │Download    │→ │Load Data   │→ │Validate    │            │
│  │NTSB Data   │  │            │  │Quality     │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         │                │                │                  │
│         ↓                ↓                ↓                  │
│  ┌──────────────────────────────────────────────┐          │
│  │    Notification Callbacks (Slack + Email)     │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────┐
        │  PostgreSQL Monitoring Views     │
        │  • vw_database_metrics           │
        │  • vw_data_quality_checks        │
        │  • vw_monthly_event_trends       │
        │  • vw_database_health            │
        └─────────────────────────────────┘
```

---

## 2. Quick Start

### 2.1 Prerequisites

- ✅ Airflow infrastructure running (Docker Compose)
- ✅ PostgreSQL database accessible
- ✅ Production DAG (`monthly_sync_ntsb_data`) deployed

### 2.2 Minimum Viable Monitoring (5 minutes)

For fastest setup with essential monitoring:

1. **Configure Slack webhook** (see Section 3)
2. **Update `.env` file** with webhook URL
3. **Restart Airflow** to pick up changes
4. **Test with intentional failure** (see Section 8)

Skip email and dashboard for now - add later if needed.

---

## 3. Slack Integration

### 3.1 Create Slack App

1. **Go to**: https://api.slack.com/apps
2. **Click**: "Create New App" → "From scratch"
3. **App Name**: `NTSB Alerts` (or your preference)
4. **Workspace**: Select your workspace
5. **Click**: "Create App"

### 3.2 Enable Incoming Webhooks

1. **Navigate to**: "Incoming Webhooks" in left sidebar
2. **Toggle**: "Activate Incoming Webhooks" to ON
3. **Click**: "Add New Webhook to Workspace"
4. **Select Channel**: Create or select `#ntsb-alerts` for critical alerts
5. **Copy**: Webhook URL (starts with `https://hooks.slack.com/services/...`)
6. **Repeat**: Create second webhook for `#ntsb-etl` (info messages)

### 3.3 Configure Environment Variables

Edit `airflow/.env`:

```bash
# Slack Webhook URLs
SLACK_WEBHOOK_CRITICAL=https://hooks.slack.com/services/YOUR/CRITICAL/WEBHOOK
SLACK_WEBHOOK_INFO=https://hooks.slack.com/services/YOUR/INFO/WEBHOOK

# Airflow Base URL (for log links)
AIRFLOW_BASE_URL=http://localhost:8080
```

### 3.4 Restart Airflow

```bash
cd airflow
docker compose restart
```

### 3.5 Test Slack Webhook

```bash
# Test from command line
docker compose exec airflow-webserver python -c "
import os
import requests
import json

webhook = os.getenv('SLACK_WEBHOOK_CRITICAL')
payload = {
    'text': '✅ Slack webhook test from NTSB ETL system!'
}
response = requests.post(webhook, json=payload)
print(f'Status: {response.status_code}')
"
```

**Expected Result**: Message appears in `#ntsb-alerts` channel within 5 seconds.

---

## 4. Email Alerts

### 4.1 Gmail Setup (Recommended)

#### Step 1: Generate App Password

1. **Go to**: https://myaccount.google.com/apppasswords
2. **Sign in**: to Google account
3. **App name**: "NTSB Airflow Alerts"
4. **Click**: "Generate"
5. **Copy**: 16-character password (e.g., `abcd efgh ijkl mnop`)

#### Step 2: Configure SMTP in `.env`

```bash
# Email Recipients (comma-separated)
EMAIL_RECIPIENTS_CRITICAL=your-email@gmail.com,team-lead@company.com
EMAIL_RECIPIENTS_INFO=your-email@gmail.com

# SMTP Configuration (Gmail)
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_STARTTLS=True
AIRFLOW__SMTP__SMTP_SSL=False
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_MAIL_FROM=ntsb-alerts@gmail.com
AIRFLOW__SMTP__SMTP_USER=ntsb-alerts@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=abcd efgh ijkl mnop
```

#### Step 3: Restart Airflow

```bash
cd airflow
docker compose restart
```

### 4.2 Other SMTP Providers

#### SendGrid

```bash
AIRFLOW__SMTP__SMTP_HOST=smtp.sendgrid.net
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_USER=apikey
AIRFLOW__SMTP__SMTP_PASSWORD=your-sendgrid-api-key
```

#### AWS SES

```bash
AIRFLOW__SMTP__SMTP_HOST=email-smtp.us-east-1.amazonaws.com
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_USER=your-ses-smtp-username
AIRFLOW__SMTP__SMTP_PASSWORD=your-ses-smtp-password
```

### 4.3 Test Email Alerts

```bash
# Test email from Airflow container
docker compose -f airflow/docker-compose.yml exec airflow-webserver \
  airflow tasks test monthly_sync_ntsb_data check_latest_data 2025-11-07
```

Check your inbox for test email with HTML formatting.

---

## 5. Monitoring Views

### 5.1 Available Views

The monitoring system provides 4 PostgreSQL views for real-time metrics:

| View | Purpose | Refresh Rate | Rows |
|------|---------|--------------|------|
| `vw_database_metrics` | Table sizes and row counts | Real-time | 11 |
| `vw_data_quality_checks` | Data quality issues | Real-time | 9 |
| `vw_monthly_event_trends` | Monthly event statistics | Real-time | 24 |
| `vw_database_health` | Overall system health | Real-time | 1 |

### 5.2 Query Examples

#### Database Metrics

```sql
-- View table sizes and row counts
SELECT
    tablename,
    row_count,
    table_size_pretty,
    last_analyze
FROM vw_database_metrics
ORDER BY row_count DESC
LIMIT 10;
```

**Sample Output**:
```
  tablename   | row_count | table_size_pretty |      last_analyze
--------------+-----------+-------------------+------------------------
 findings     |    101275 | 24 MB             | 2025-11-07 03:44:38
 events       |     92771 | 47 MB             | 2025-11-07 03:42:07
 injury       |     91365 | 10 MB             | 2025-11-07 03:44:38
 narratives   |     88485 | 391 MB            | 2025-11-07 03:44:39
```

#### Data Quality Checks

```sql
-- Check for data quality issues
SELECT
    metric_name,
    metric_value,
    description,
    severity
FROM vw_data_quality_checks
WHERE severity != 'OK'
ORDER BY severity DESC, metric_value DESC;
```

**Sample Output** (healthy system):
```
 metric_name | metric_value | description | severity
-------------+--------------+-------------+----------
(0 rows)  -- No issues found
```

#### Monthly Trends

```sql
-- View monthly event trends (last 12 months)
SELECT
    to_char(month, 'YYYY-MM') as month,
    event_count,
    fatal_accidents,
    serious_injuries
FROM vw_monthly_event_trends
ORDER BY month DESC
LIMIT 12;
```

**Sample Output**:
```
  month   | event_count | fatal_accidents | serious_injuries
----------+-------------+-----------------+------------------
 2025-10  |          54 |               3 |                5
 2025-09  |          67 |               4 |                8
 2025-08  |          71 |               5 |                9
```

#### Database Health

```sql
-- Overall database health summary
SELECT
    database_size,
    total_events,
    events_last_30_days,
    earliest_event_date,
    latest_event_date
FROM vw_database_health;
```

**Sample Output**:
```
 database_size | total_events | events_last_30_days | earliest_event_date | latest_event_date
---------------+--------------+---------------------+---------------------+-------------------
 512 MB        |        92771 |                  54 | 1977-06-19          | 2025-10-30
```

### 5.3 Scheduling Monitoring Queries

Create a cron job or Airflow DAG to periodically query views and log results:

```bash
# Example: Daily health check
0 6 * * * psql -d ntsb_aviation -c "SELECT * FROM vw_database_health;" >> /var/log/ntsb_health.log 2>&1
```

---

## 6. Anomaly Detection

### 6.1 What's Checked

The `detect_anomalies.py` script performs 5 automated checks:

| Check # | Name | Description | Severity |
|---------|------|-------------|----------|
| 1 | Missing Critical Fields | NULL ev_id, ev_date, coordinates | WARNING |
| 2 | Coordinate Outliers | Lat/lon outside valid bounds | WARNING |
| 3 | Event Count Drop | Monthly events <50% of average | WARNING |
| 4 | Referential Integrity | Orphaned child records | WARNING |
| 5 | Duplicate Detection | Same ev_id appears >1 time | CRITICAL |

### 6.2 Run Manually

```bash
# Run anomaly detection on recent data (last 35 days)
python scripts/detect_anomalies.py

# Run with custom lookback period
python scripts/detect_anomalies.py --lookback-days 60

# Run and save results to JSON
python scripts/detect_anomalies.py --output /tmp/anomalies.json
```

### 6.3 Interpret Results

**Successful Run** (no anomalies):
```
================================================================================
NTSB Aviation Data Quality Anomaly Detection
================================================================================
Timestamp: 2025-11-07 19:00:00
Lookback Period: 35 days

🔍 Check 1: Missing Critical Fields
✅ PASS - Total events checked: 92771

🔍 Check 2: Coordinate Outliers
✅ PASS - Total events with coordinates: 77887

🔍 Check 3: Statistical Anomalies (Event Count Drop)
✅ PASS - Latest month: 54 events, 12-month avg: 67

🔍 Check 4: Referential Integrity (Orphaned Records)
✅ PASS
  - No orphaned records found

🔍 Check 5: Duplicate Detection
✅ PASS - Duplicates found: 0

================================================================================
SUMMARY
================================================================================
✅ Passed: 5/5
⚠️  Anomalies Found: 0

✅ SUCCESS: All checks passed!
```

**Failed Run** (with anomalies):
```
🔍 Check 5: Duplicate Detection
❌ CRITICAL - Duplicates found: 3
  - Duplicate ev_id 'CHI12345': 2 occurrences
  - Duplicate ev_id 'LAX67890': 2 occurrences
  - Duplicate ev_id 'NYC11111': 2 occurrences

================================================================================
SUMMARY
================================================================================
✅ Passed: 4/5
⚠️  Anomalies Found: 3

❌ CRITICAL: 1 check(s) failed with critical severity
```

**Exit Codes**:
- `0` - All checks passed
- `1` - Warnings found (non-critical)
- `2` - Critical failures found

### 6.4 Integration with DAG

The anomaly detection is integrated into the `monthly_sync_ntsb_data` DAG and runs automatically after data validation. Results are logged to Airflow task logs and trigger alerts if anomalies are found.

---

## 7. Dashboard Access

### 7.1 Airflow Web UI

The built-in Airflow UI provides:

- **DAG Run History**: http://localhost:8080/dags/monthly_sync_ntsb_data/grid
- **Task Duration Charts**: http://localhost:8080/dags/monthly_sync_ntsb_data/duration
- **Gantt Chart**: http://localhost:8080/dags/monthly_sync_ntsb_data/gantt
- **Task Logs**: Click any task instance for detailed logs

### 7.2 PostgreSQL Admin Tools

Use any PostgreSQL client to query monitoring views:

**Option 1: DBeaver** (recommended)
1. Download: https://dbeaver.io/download/
2. Connect to: `localhost:5432/ntsb_aviation`
3. Run monitoring queries from Section 5

**Option 2: psql (command line)**
```bash
# Interactive session
psql -d ntsb_aviation

# Run monitoring queries
\x  -- Expanded display mode
SELECT * FROM vw_database_health;
```

**Option 3: pgAdmin**
1. Download: https://www.pgadmin.org/download/
2. Connect to: `localhost:5432`
3. Navigate to `ntsb_aviation` database
4. Create custom dashboard with monitoring views

### 7.3 Custom Streamlit Dashboard (Future Enhancement)

A Streamlit dashboard can be added for real-time visualization:

```bash
# Create dashboard directory
mkdir dashboard
cd dashboard

# Create requirements.txt
cat > requirements.txt << EOF
streamlit==1.28.0
psycopg2-binary==2.9.9
pandas==2.1.3
plotly==5.18.0
EOF

# Install dependencies
pip install -r requirements.txt

# Create basic dashboard (app.py)
# See: https://docs.streamlit.io/ for examples

# Run dashboard
streamlit run app.py
```

Dashboard will be accessible at http://localhost:8501

---

## 8. Troubleshooting

### 8.1 Slack Alerts Not Sending

**Problem**: No Slack messages received after DAG failure

**Diagnostics**:
```bash
# Check environment variables loaded in Airflow
docker compose -f airflow/docker-compose.yml exec airflow-webserver env | grep SLACK

# Test webhook manually
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test from curl"}' \
  $SLACK_WEBHOOK_CRITICAL
```

**Common Fixes**:
1. Verify webhook URL is correct (no typos)
2. Restart Airflow after changing `.env`: `docker compose restart`
3. Check Slack app permissions (must have `incoming-webhook` scope)
4. Verify webhook channel still exists

### 8.2 Email Alerts Not Sending

**Problem**: No emails received

**Diagnostics**:
```bash
# Check SMTP configuration
docker compose -f airflow/docker-compose.yml exec airflow-webserver env | grep SMTP

# Test SMTP connection
docker compose -f airflow/docker-compose.yml exec airflow-webserver python3 << EOF
import smtplib
import os

host = os.getenv('AIRFLOW__SMTP__SMTP_HOST')
port = int(os.getenv('AIRFLOW__SMTP__SMTP_PORT'))
user = os.getenv('AIRFLOW__SMTP__SMTP_USER')
password = os.getenv('AIRFLOW__SMTP__SMTP_PASSWORD')

print(f"Connecting to {host}:{port} as {user}...")
server = smtplib.SMTP(host, port)
server.starttls()
server.login(user, password)
print("✅ SMTP connection successful!")
server.quit()
EOF
```

**Common Fixes**:
1. **Gmail**: Must use App Password, not regular password
2. **2FA**: Enable 2-factor auth before generating App Password
3. **"Less secure apps"**: No longer supported - use App Password
4. Check spam/junk folder

### 8.3 Monitoring Views Empty

**Problem**: `SELECT * FROM vw_database_metrics` returns 0 rows

**Diagnostics**:
```sql
-- Check if views exist
SELECT viewname FROM pg_views WHERE schemaname = 'public';

-- Check underlying tables
SELECT COUNT(*) FROM pg_stat_user_tables WHERE schemaname = 'public';
```

**Common Fixes**:
1. Run `scripts/create_monitoring_views.sql` to create views
2. Verify PostgreSQL permissions (user must have SELECT access)
3. Check database connection (correct database selected)

### 8.4 Anomaly Detection Errors

**Problem**: `python scripts/detect_anomalies.py` fails with connection error

**Diagnostics**:
```bash
# Check environment variables
echo $NTSB_DB_HOST
echo $NTSB_DB_PORT
echo $NTSB_DB_NAME

# Test database connection
psql -h $NTSB_DB_HOST -p $NTSB_DB_PORT -d $NTSB_DB_NAME -c "SELECT 1;"
```

**Common Fixes**:
1. Source environment variables: `source airflow/.env`
2. Run from Airflow container: `docker compose exec airflow-webserver python /opt/airflow/dags/detect_anomalies.py`
3. Check PostgreSQL is running: `systemctl status postgresql`

### 8.5 Testing Alerts

**Intentional Failure Test**:

1. Modify DAG to fail intentionally:
```python
# In airflow/dags/monthly_sync_dag.py
# Add temporary test task
test_failure = BashOperator(
    task_id='test_failure',
    bash_command='exit 1',  # INTENTIONAL FAILURE
)
```

2. Trigger DAG:
```bash
docker compose -f airflow/docker-compose.yml exec airflow-webserver \
  airflow dags trigger monthly_sync_ntsb_data
```

3. Verify alert received in Slack/Email within 30 seconds

4. Revert change and re-test:
```bash
# Remove test task, commit changes
git diff airflow/dags/monthly_sync_dag.py
git checkout airflow/dags/monthly_sync_dag.py
```

---

## 9. Customization

### 9.1 Add Custom Data Quality Check

Edit `scripts/detect_anomalies.py`:

```python
def check_custom_metric(conn, lookback_days: int = 35) -> Dict[str, Any]:
    """
    Check 6: Custom business logic check.
    """
    print("\n🔍 Check 6: Custom Metric")

    with conn.cursor() as cursor:
        query = """
        SELECT COUNT(*) as suspicious_events
        FROM events
        WHERE ev_date >= CURRENT_DATE - INTERVAL '%s days'
          AND your_custom_condition = true;
        """ % lookback_days

        cursor.execute(query)
        result = cursor.fetchone()

    anomalies = []
    if result["suspicious_events"] > 10:
        anomalies.append(f"Found {result['suspicious_events']} suspicious events")

    passed = len(anomalies) == 0
    status = "✅ PASS" if passed else "⚠️  WARNING"

    print(f"{status} - Suspicious events: {result['suspicious_events']}")

    return {
        "check": "custom_metric",
        "passed": passed,
        "severity": "WARNING" if not passed else "INFO",
        "anomalies": anomalies,
        "details": dict(result),
    }

# Add to run_all_checks() function:
results.append(check_custom_metric(conn, lookback_days))
```

### 9.2 Modify Alert Thresholds

Edit `airflow/plugins/notification_callbacks.py`:

```python
# Change threshold for missing coordinates (default: 1%)
threshold = total * 0.05  # Now 5%

# Add custom severity logic
if result["missing_latitude"] > threshold * 2:
    severity = "CRITICAL"  # > 10% is critical
elif result["missing_latitude"] > threshold:
    severity = "WARNING"   # > 5% is warning
```

### 9.3 Add Monitoring View

Create custom view in `scripts/create_monitoring_views.sql`:

```sql
-- Custom View: Aircraft Age Distribution
CREATE OR REPLACE VIEW vw_aircraft_age_distribution AS
SELECT
    CASE
        WHEN EXTRACT(YEAR FROM ev_date) - acft_year < 5 THEN '0-5 years'
        WHEN EXTRACT(YEAR FROM ev_date) - acft_year < 10 THEN '5-10 years'
        WHEN EXTRACT(YEAR FROM ev_date) - acft_year < 20 THEN '10-20 years'
        ELSE '20+ years'
    END as age_bracket,
    COUNT(*) as accident_count
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE acft_year IS NOT NULL
GROUP BY age_bracket
ORDER BY age_bracket;
```

### 9.4 Schedule Monitoring Reports

Add cron job for daily/weekly reports:

```bash
# Daily health check email (6 AM)
0 6 * * * psql -d ntsb_aviation -c "SELECT * FROM vw_database_health;" | \
  mail -s "NTSB Daily Health Check" admin@company.com

# Weekly data quality report (Monday 8 AM)
0 8 * * 1 psql -d ntsb_aviation -c "SELECT * FROM vw_data_quality_checks;" | \
  mail -s "NTSB Weekly Quality Report" team@company.com
```

---

## 10. Production Checklist

Before December 1st production run:

- [ ] Slack webhook configured and tested
- [ ] Email SMTP configured and tested (optional)
- [ ] Monitoring views created in database
- [ ] Anomaly detection script tested with recent data
- [ ] DAG includes notification callbacks
- [ ] Intentional failure test passed (alert received <30s)
- [ ] Baseline run completed successfully with success notification
- [ ] Team trained on interpreting alerts
- [ ] Escalation procedure documented
- [ ] Backup contacts configured in `.env`

---

## 11. Support & Resources

**Internal Documentation**:
- [Airflow Setup Guide](AIRFLOW_SETUP_GUIDE.md)
- [Sprint 3 Week 2 Report](SPRINT_3_WEEK_2_COMPLETION_REPORT.md)
- [Production DAG](../airflow/dags/monthly_sync_dag.py)

**External Resources**:
- [Slack Webhooks](https://api.slack.com/messaging/webhooks)
- [Gmail App Passwords](https://myaccount.google.com/apppasswords)
- [Airflow Callbacks](https://airflow.apache.org/docs/apache-airflow/stable/howto/callbacks.html)
- [PostgreSQL Monitoring](https://www.postgresql.org/docs/current/monitoring.html)

**Troubleshooting**:
- Check Airflow logs: `airflow/logs/`
- Check PostgreSQL logs: `journalctl -u postgresql -f`
- Slack API status: https://status.slack.com/
- Gmail SMTP status: https://www.google.com/appsstatus

---

**End of Monitoring Setup Guide**
