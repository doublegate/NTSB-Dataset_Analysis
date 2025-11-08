# NTSB Aviation Database - Quick Start Guide

This guide will get you started with the NTSB Aviation Accident Database in under 15 minutes. Choose your workflow:
- **Option A: PostgreSQL Database** (Recommended) - Full-featured database with optimized queries, geospatial analysis, and monitoring
- **Option B: CSV/DuckDB Analysis** - Quick exploration without database setup

---

## Prerequisites

### System Requirements
- **Operating System**: Linux (tested on CachyOS/Arch), macOS, or Windows WSL2
- **Disk Space**: 2-5 GB free
- **Memory**: 4 GB+ RAM recommended
- **PostgreSQL**: Version 18.0+ (for Option A)
- **Python**: Version 3.11+ with venv
- **Tools**: Git, mdbtools (for .mdb extraction)

### Verify Prerequisites

```bash
# Check PostgreSQL (Option A)
psql --version
# Expected: PostgreSQL 18.0 or higher

# Check Python
python --version
# Expected: Python 3.11 or higher

# Check mdbtools
mdb-ver --version
# Expected: mdbtools version info

# Check disk space
df -h .
# Ensure 2-5 GB free space
```

---

## Option A: PostgreSQL Database (Recommended)

For optimal query performance, advanced analytics, geospatial queries, and production-ready monitoring infrastructure.

### Step 1: Clone Repository

```bash
git clone https://github.com/doublegate/NTSB-Dataset_Analysis.git
cd NTSB-Dataset_Analysis
```

### Step 2: Database Setup (5 minutes)

#### Automated Setup (Recommended)

```bash
# Run the automated setup script
./scripts/setup_database.sh

# This will:
#   - Start PostgreSQL service
#   - Create 'ntsb_aviation' database
#   - Enable PostGIS, pg_trgm, pgcrypto extensions
#   - Create 11 tables + indexes
#   - Set up staging tables and load tracking
#   - Transfer ownership to current user (NO SUDO after setup)
```

#### Manual Setup (If Automated Fails)

```bash
# 1. Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Optional: auto-start on boot

# 2. Create database
sudo -u postgres createdb ntsb_aviation

# 3. Enable extensions
sudo -u postgres psql -d ntsb_aviation -c "CREATE EXTENSION IF NOT EXISTS postgis;"
sudo -u postgres psql -d ntsb_aviation -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
sudo -u postgres psql -d ntsb_aviation -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

# 4. Apply schema
sudo -u postgres psql -d ntsb_aviation -f scripts/schema.sql

# 5. Grant permissions to your user
sudo -u postgres psql -d ntsb_aviation -f scripts/transfer_ownership.sql

# 6. Verify setup
psql -d ntsb_aviation -c "\dt"  # Should show 11 tables
psql -d ntsb_aviation -c "\di"  # Should show 59+ indexes
```

### Step 3: Load Data (10-15 minutes)

```bash
# 1. Activate Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load current data (2008-present, ~29,773 events)
python scripts/load_with_staging.py --source datasets/avall.mdb

# Expected output:
#   - Loading 11 tables in dependency order
#   - Progress updates for each table
#   - Total: ~478,631 rows loaded
#   - Duration: 5-10 minutes
#   - Success confirmation

# 4. Load historical data (optional, 1982-2007)
python scripts/load_with_staging.py --source datasets/Pre2008.mdb
```

### Step 4: Query Optimization (5 minutes)

```bash
# Create materialized views and additional indexes
psql -d ntsb_aviation -f scripts/optimize_queries.sql

# This creates:
#   - 6 materialized views (yearly, state, aircraft, decade, crew, findings)
#   - 9 additional performance indexes
#   - Analyzes all tables for query planner
#   - Result: 30-114x speedup for analytical queries
```

### Step 5: Validation (3 minutes)

```bash
# Run comprehensive data quality checks
psql -d ntsb_aviation -f scripts/validate_data.sql > validation_report.txt

# Review results
less validation_report.txt

# Key checks:
#   ✓ Row counts match expectations
#   ✓ Primary keys are unique
#   ✓ Foreign keys are valid
#   ✓ Coordinates are within valid ranges
#   ✓ Dates are within expected range
#   ✓ Partitions are populated
#   ✓ Indexes are created

# Quick manual verification
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"
# Expected: ~92,771 events (if both avall.mdb and Pre2008.mdb loaded)

psql -d ntsb_aviation -c "SELECT * FROM vw_database_health;"
# Shows overall database health snapshot
```

### Step 6: Monitoring Setup (Production Deployments)

For production deployments with automated data updates, set up monitoring infrastructure:

#### Option 6A: Basic Monitoring (No Airflow)

```bash
# 1. Create monitoring views
psql -d ntsb_aviation -f scripts/create_monitoring_views.sql

# 2. Run anomaly detection
python scripts/detect_anomalies.py --lookback-days 30 --output json

# 3. Check data quality
psql -d ntsb_aviation -c "SELECT * FROM vw_data_quality_checks;"

# 4. View monthly trends
psql -d ntsb_aviation -c "SELECT * FROM vw_monthly_event_trends ORDER BY year DESC, month DESC LIMIT 12;"
```

#### Option 6B: Full Monitoring with Airflow (Automated Monthly Updates)

```bash
# 1. Configure PostgreSQL for Docker access (ONE-TIME SETUP)
# Edit postgresql.conf: set listen_addresses = '*'
# Edit pg_hba.conf: add entry for Docker bridge (172.17.0.0/16)
# Restart PostgreSQL: sudo systemctl restart postgresql

# 2. Start Airflow services
cd airflow/
docker compose up -d

# 3. Access Web UI
open http://localhost:8080  # Login: airflow/airflow

# 4. Configure notifications (optional)
# Edit airflow/.env:
#   - Add Slack webhook URL
#   - Add email SMTP settings
#   - See docs/MONITORING_SETUP_GUIDE.md for details

# 5. Trigger production DAG
docker compose exec airflow-scheduler airflow dags trigger monthly_sync_ntsb_data

# 6. Stop services
docker compose down
```

See **[AIRFLOW_SETUP_GUIDE.md](docs/AIRFLOW_SETUP_GUIDE.md)** and **[MONITORING_SETUP_GUIDE.md](docs/MONITORING_SETUP_GUIDE.md)** for detailed documentation.

### Step 7: Verification

Run a test query to confirm everything is working:

```bash
psql -d ntsb_aviation -c "
SELECT
    ev_year,
    COUNT(*) as accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal,
    SUM(inj_tot_f) as fatalities
FROM events
WHERE ev_year >= 2020
GROUP BY ev_year
ORDER BY ev_year DESC;
"
```

**Expected**: Yearly statistics for 2020-2025 with sub-second response time.

✅ **Congratulations!** Your PostgreSQL database is ready for analysis.

---

## Option B: CSV/DuckDB Analysis

For quick exploration without database setup. Ideal for initial data exploration, one-off analyses, or users who prefer working with CSV files.

### Step 1: Clone Repository

```bash
git clone https://github.com/doublegate/NTSB-Dataset_Analysis.git
cd NTSB-Dataset_Analysis
```

### Step 2: Extract Data from MDB Files

```fish
# Setup tools (Fish shell required)
./setup.fish  # Install tools: mdbtools, Python, Rust tools

# Extract all tables from databases
./scripts/extract_all_tables.fish datasets/avall.mdb
./scripts/extract_all_tables.fish datasets/Pre2008.mdb
./scripts/extract_all_tables.fish datasets/PRE1982.MDB

# Or extract single table
mdb-export datasets/avall.mdb events > data/events.csv
```

### Step 3: Quick CSV Analysis

```bash
# View first 10 rows
head -n 10 data/avall-events.csv

# Count rows
wc -l data/avall-events.csv

# View column names
head -n 1 data/avall-events.csv | tr ',' '\n'

# Quick statistics (requires csvkit)
csvstat data/avall-events.csv

# Filter by state
csvgrep -c ev_state -m "CA" data/avall-events.csv > data/ca_events.csv
```

### Step 4: DuckDB Queries

```bash
# Launch interactive DuckDB
duckdb

# Or one-liner queries
duckdb -c "SELECT COUNT(*) FROM 'data/avall-events.csv'"

duckdb -c "
SELECT ev_state, COUNT(*) as count
FROM 'data/avall-events.csv'
GROUP BY ev_state
ORDER BY count DESC
LIMIT 10
"

# Join multiple tables
duckdb -c "
SELECT e.ev_id, e.ev_date, e.ev_state, a.acft_make, a.acft_model
FROM 'data/avall-events.csv' e
LEFT JOIN 'data/avall-aircraft.csv' a ON e.ev_id = a.ev_id
WHERE e.ev_year >= 2020
LIMIT 100
"
```

### Step 5: Python Analysis

```bash
# Activate Python environment
source .venv/bin/activate

# Start Jupyter
jupyter lab

# Or run example scripts
python examples/quick_analysis.py
python examples/advanced_analysis.py
python examples/geospatial_analysis.py
```

#### Quick Python Examples

```python
import pandas as pd
import duckdb

# Load data with DuckDB (fast)
df = duckdb.query("SELECT * FROM 'data/avall-events.csv' WHERE ev_year >= 2020").to_df()

# Or with pandas
df = pd.read_csv('data/avall-events.csv')

# Quick exploration
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())

# Filter and analyze
recent = df[df['ev_year'] >= 2020]
print(f"Accidents since 2020: {len(recent)}")
print(f"Total fatalities: {recent['inj_tot_f'].sum()}")

# Group by state
by_state = df.groupby('ev_state').size().sort_values(ascending=False)
print(by_state.head(10))
```

### Step 6: Rust Tools (Fast CSV Processing)

```bash
# Count rows
xsv count data/avall-events.csv

# Select specific columns
xsv select ev_id,ev_date,ev_state,ev_city data/avall-events.csv | head

# Search
xsv search -s ev_state "CA" data/avall-events.csv

# Statistics
xsv stats data/avall-events.csv

# Frequency count
xsv frequency -s ev_state data/avall-events.csv
```

---

## Common SQL Queries

### PostgreSQL Examples

```sql
-- Recent fatal accidents with aircraft details
SELECT e.ev_id, e.ev_date, e.ev_type, e.ev_state, e.inj_tot_f,
       a.acft_make, a.acft_model
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.inj_tot_f > 0 AND e.ev_year >= 2020
ORDER BY e.ev_date DESC
LIMIT 100;

-- Yearly accident statistics (from materialized view)
SELECT * FROM mv_yearly_stats
WHERE year >= 2020
ORDER BY year DESC;

-- Top 10 aircraft types by accident count
SELECT * FROM mv_aircraft_stats
ORDER BY total_accidents DESC
LIMIT 10;

-- Geospatial query: accidents near a location (requires PostGIS)
SELECT ev_id, ev_date, ev_state,
       ST_Distance(location_geom, ST_MakePoint(-122.4194, 37.7749)) as distance_meters
FROM events
WHERE location_geom IS NOT NULL
  AND ST_DWithin(location_geom, ST_MakePoint(-122.4194, 37.7749), 50000)
ORDER BY distance_meters;
```

### DuckDB Examples (CSV files)

```sql
-- Recent fatal accidents
SELECT ev_id, ev_date, ev_type, ev_state, inj_tot_f
FROM 'data/avall-events.csv'
WHERE inj_tot_f > 0 AND ev_year >= 2020
ORDER BY ev_date DESC;

-- Most common occurrence types
SELECT occurrence_code, COUNT(*) as count
FROM 'data/avall-Occurrences.csv'
GROUP BY occurrence_code
ORDER BY count DESC
LIMIT 10;

-- Events with aircraft details
SELECT e.ev_id, e.ev_date, e.ev_state, a.acft_make, a.acft_model
FROM 'data/avall-events.csv' e
LEFT JOIN 'data/avall-aircraft.csv' a ON e.ev_id = a.ev_id
WHERE e.ev_year >= 2020
LIMIT 100;
```

---

## Troubleshooting

### PostgreSQL Issues

#### Issue: PostgreSQL Service Not Running

```bash
# Start service
sudo systemctl start postgresql

# Enable auto-start on boot
sudo systemctl enable postgresql

# Check status
systemctl status postgresql
```

#### Issue: Database Connection Refused

```bash
# Check if port 5432 is listening
sudo netstat -tlnp | grep 5432

# Check PostgreSQL logs
sudo journalctl -u postgresql -n 50
```

#### Issue: Permission Denied for Table

```bash
# Re-grant permissions to your user
sudo -u postgres psql -d ntsb_aviation -f scripts/transfer_ownership.sql

# Or manually
sudo -u postgres psql -d ntsb_aviation -c "
    GRANT ALL PRIVILEGES ON DATABASE ntsb_aviation TO $USER;
    GRANT ALL PRIVILEGES ON SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $USER;
"
```

#### Issue: Slow Query Performance

```bash
# Run ANALYZE to update statistics
psql -d ntsb_aviation -c "ANALYZE;"

# Check if indexes are being used
psql -d ntsb_aviation -c "
    EXPLAIN ANALYZE
    SELECT * FROM events WHERE ev_state = 'CA' LIMIT 10;
"
# Look for "Index Scan" in output (not "Seq Scan")

# Check buffer cache hit ratio (should be >95%)
psql -d ntsb_aviation -c "
SELECT
    ROUND(100.0 * sum(blks_hit) / NULLIF(sum(blks_hit) + sum(blks_read), 0), 2) AS cache_hit_ratio
FROM pg_stat_database
WHERE datname = 'ntsb_aviation';
"
```

#### Issue: Out of Memory During Load

```bash
# Load tables one at a time
python scripts/load_with_staging.py --source datasets/avall.mdb

# Or reduce chunk size
python scripts/load_with_staging.py --source datasets/avall.mdb --chunk-size 500
```

### CSV/DuckDB Issues

#### Issue: mdbtools Not Found

```bash
# mdbtools is in AUR, install with:
paru -S mdbtools

# Or fix PKGBUILD if autoconf errors:
./fix_mdbtools_pkgbuild.fish
```

#### Issue: Python Module Not Found

```bash
source .venv/bin/activate
pip install <module_name>
```

#### Issue: Out of Memory Errors

```python
# Use Polars instead of pandas (10x faster, lower memory)
import polars as pl
df = pl.read_csv('data/avall-events.csv')

# Or use DuckDB for querying
import duckdb
result = duckdb.query("SELECT * FROM 'data/avall-events.csv' WHERE ev_year >= 2020").to_df()

# Or read in chunks
for chunk in pd.read_csv('data/avall-events.csv', chunksize=10000):
    process(chunk)
```

#### Issue: Encoding Issues

```python
# Try different encodings
df = pd.read_csv('file.csv', encoding='latin1')
df = pd.read_csv('file.csv', encoding='utf-8')
```

---

## Quick Reference Commands

### Database Operations

```bash
# Check database size
psql -d ntsb_aviation -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));"

# Count events
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"

# Refresh materialized views
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"

# Run anomaly detection
python scripts/detect_anomalies.py --lookback-days 30

# Check data quality
psql -d ntsb_aviation -c "SELECT * FROM vw_data_quality_checks;"

# View database health
psql -d ntsb_aviation -c "SELECT * FROM vw_database_health;"
```

### CSV Operations

```bash
# Extract table from MDB
mdb-export datasets/avall.mdb events > data/events.csv

# Count rows
wc -l data/avall-events.csv

# Filter by column
csvgrep -c ev_state -m "CA" data/avall-events.csv

# Quick statistics
csvstat data/avall-events.csv

# Convert to JSON
csvjson data/avall-events.csv > events.json
```

---

## NTSB Coding System Reference

Quick reference for common NTSB codes (see `ref_docs/codman.pdf` for complete list):

### Occurrence Codes (100-430)
- 100: ABRUPT MANEUVER
- 110: AIRFRAME/COMPONENT/SYSTEM FAILURE/MALFUNCTION
- 200: ENGINE FAILURE
- 250: FIRE
- 300: FUEL EXHAUSTION
- 350: MIDAIR COLLISION
- 400: WEATHER

### Phase Codes (500-610)
- 500: STANDING
- 510: TAXI
- 520: TAKEOFF
- 530: CLIMB
- 540: CRUISE
- 560: DESCENT
- 580: APPROACH
- 590: LANDING
- 600: MANEUVERING

### Aircraft Components (10000-17710)
- 10000-11700: Airframe (wings, fuselage, landing gear, flight controls)
- 12000-13500: Systems (hydraulic, electrical, environmental, fuel)
- 14000-17710: Powerplant (engines, propellers, turbines, exhaust)

---

## Next Steps

### For Data Analysis
1. **Explore Examples**: See `examples/` directory for Jupyter notebooks and Python scripts
2. **Read Documentation**: Check [TOOLS_AND_UTILITIES.md](TOOLS_AND_UTILITIES.md) for comprehensive tool catalog
3. **View Schema**: See `scripts/schema.sql` for complete database structure
4. **Learn Coding System**: Read `ref_docs/codman.pdf` for NTSB coding manual

### For Production Deployment
1. **Set Up Airflow**: See [AIRFLOW_SETUP_GUIDE.md](docs/AIRFLOW_SETUP_GUIDE.md)
2. **Configure Monitoring**: See [MONITORING_SETUP_GUIDE.md](docs/MONITORING_SETUP_GUIDE.md)
3. **Performance Tuning**: See [PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md)
4. **Historical Data**: Load PRE1982.MDB (1962-1981) - see [PRE1982_ANALYSIS.md](docs/PRE1982_ANALYSIS.md)

### For Advanced Analytics
1. **Machine Learning**: See [MACHINE_LEARNING_APPLICATIONS.md](docs/MACHINE_LEARNING_APPLICATIONS.md)
2. **Geospatial Analysis**: See [GEOSPATIAL_ADVANCED.md](docs/GEOSPATIAL_ADVANCED.md)
3. **NLP/Text Mining**: See [NLP_TEXT_MINING.md](docs/NLP_TEXT_MINING.md)
4. **Project Roadmap**: See [to-dos/PHASE_2_ANALYTICS.md](to-dos/PHASE_2_ANALYTICS.md)

---

## Verification Checklist

After completing setup, verify:

### PostgreSQL (Option A)
- [ ] PostgreSQL service is running
- [ ] Database `ntsb_aviation` exists
- [ ] 11 tables created (events, aircraft, Flight_Crew, etc.)
- [ ] 59+ indexes created
- [ ] ~92,771 events loaded (both databases)
- [ ] Foreign key relationships valid (0 orphaned records)
- [ ] Query performance <100ms for indexed queries
- [ ] Spatial queries working (PostGIS enabled)
- [ ] Full-text search working (pg_trgm enabled)
- [ ] Materialized views created (6 views)
- [ ] Monitoring views created (4 views)
- [ ] Data quality: 9/9 checks passed

### CSV/DuckDB (Option B)
- [ ] MDB files extracted to CSV
- [ ] CSV files in `data/` directory
- [ ] DuckDB queries working
- [ ] Python environment activated
- [ ] Example scripts running successfully
- [ ] Jupyter notebooks accessible

---

## Performance Targets

### PostgreSQL Query Performance
- **p50 Latency**: <10ms for simple queries
- **p95 Latency**: <100ms for complex analytical queries
- **p99 Latency**: <500ms for heavy aggregations
- **Buffer Cache Hit Ratio**: >95%
- **Index Usage**: >99% on primary tables

### Data Load Performance
- **avall.mdb**: ~30 seconds for full load (29,773 events)
- **Pre2008.mdb**: ~90 seconds for full load (63,000+ events)
- **Throughput**: 15,000-45,000 rows/sec (varies by table)

---

## Getting Help

- **Complete Documentation**: See [README.md](README.md#comprehensive-documentation)
- **Database Schema**: See `scripts/schema.sql` and `ref_docs/eadmspub.pdf`
- **Coding Manual**: See `ref_docs/codman.pdf` for NTSB coding system
- **Installation Issues**: See [INSTALLATION.md](INSTALLATION.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security**: See [SECURITY.md](SECURITY.md)

---

**Last Updated**: November 7, 2025
**Version**: 2.1.0
**Sprint**: Phase 1 Sprint 3 Week 3 - COMPLETE (Monitoring & Observability)
**Production Ready**: December 1st, 2025
