# CLAUDE.local.md - Current Project State & Development Guidance

**Last Updated**: 2025-11-07
**Sprint**: Phase 1 Sprint 3 Week 3 (Monitoring & Observability)
**Status**: ✅ COMPLETE (Production-ready monitoring infrastructure)

---

## 🔒 CRITICAL DEVELOPMENT PRINCIPLE: NO SUDO OPERATIONS

**IMPORTANT**: This project is designed for GitHub users who clone the repository. Users must be able to set up and operate the database **WITHOUT manual sudo operations** or database administration expertise beyond initial setup.

### Requirements:

1. **Single Setup Script**: `scripts/setup_database.sh` handles ALL initial setup
   - Database creation
   - Extension installation (PostGIS, pg_trgm, pgcrypto, pg_stat_statements)
   - Ownership transfer to current user
   - Schema creation
   - Staging infrastructure
   - Load tracking system

2. **Regular User Operations**: After setup, ALL operations run as regular user:
   - ✅ Data loading (`python scripts/load_with_staging.py`)
   - ✅ Query optimization (`psql -d ntsb_aviation -f scripts/optimize_queries.sql`)
   - ✅ Data validation (`psql -d ntsb_aviation -f scripts/validate_data.sql`)
   - ✅ Index creation, ANALYZE, materialized view management
   - ✅ Performance benchmarks

3. **No Manual SQL Commands**: Users should NEVER need to run manual psql commands as postgres user

4. **Clear Ownership Model**: Current user owns database and ALL objects
   - Database: owned by current user
   - All tables: owned by current user
   - All sequences: owned by current user
   - All functions: owned by current user
   - All materialized views: owned by current user

### When Writing New Scripts:

- ❌ **DO NOT** require sudo or superuser privileges
- ❌ **DO NOT** assume postgres user access
- ❌ **DO NOT** hardcode user names (use `$USER` or script parameters)
- ✅ **DO** assume current user owns the database
- ✅ **DO** handle errors gracefully if permissions missing
- ✅ **DO** document any one-time setup needs in setup_database.sh
- ✅ **DO** test scripts as regular user before committing

### Example Correct Patterns:

```bash
# Good: Regular user operation
psql -d ntsb_aviation -f scripts/optimize_queries.sql

# Good: Python script as regular user
python scripts/load_with_staging.py --source avall.mdb

# Bad: Requires sudo for routine operations
sudo -u postgres psql -d ntsb_aviation -c "ANALYZE events;"  # ❌ WRONG
```

---

## Current Sprint Status: Phase 1 Sprint 3 Week 3

**Objective**: Monitoring & Observability Infrastructure
**Progress**: ✅ 100% COMPLETE

### Completed ✅:

1. **Notification System** (`airflow/plugins/notification_callbacks.py`, 449 lines)
   - Slack webhook integration for real-time alerts (<30s latency)
   - Email SMTP notifications (Gmail App Password support)
   - `notify_failure()` - Combined Slack + Email for DAG failures
   - `notify_success()` - Combined Slack + Email for successful runs
   - Rich message formatting with execution metadata
   - Log URL links to Airflow Web UI
   - Environment variable configuration (no hardcoded credentials)

2. **Anomaly Detection** (`scripts/detect_anomalies.py`, 480 lines)
   - 5 automated data quality checks:
     - Check 1: Missing critical fields (ev_id, ev_date, coordinates)
     - Check 2: Coordinate outliers (lat/lon outside bounds)
     - Check 3: Statistical anomalies (event count drop >50%)
     - Check 4: Referential integrity (orphaned records)
     - Check 5: Duplicate detection (critical)
   - CLI interface with `--lookback-days` and `--output` options
   - Exit codes: 0=pass, 1=warning, 2=critical
   - JSON output for integration with monitoring systems
   - Current status: ✅ All 5 checks passed, 0 anomalies found

3. **Monitoring Views** (`scripts/create_monitoring_views.sql`, 323 lines)
   - `vw_database_metrics`: Real-time table sizes and row counts
   - `vw_data_quality_checks`: 9 quality metrics with severity levels
   - `vw_monthly_event_trends`: Monthly event statistics (24 months)
   - `vw_database_health`: Overall system health snapshot
   - All views query underlying tables in real-time (no materialization)
   - Query performance: <50ms for all views

4. **Monitoring Setup Guide** (`docs/MONITORING_SETUP_GUIDE.md`, 754 lines)
   - Section 1: Overview (architecture, alert severity matrix)
   - Section 2: Quick Start (5-minute minimum viable monitoring)
   - Section 3: Slack Integration (step-by-step webhook setup)
   - Section 4: Email Alerts (Gmail SMTP configuration)
   - Section 5: Monitoring Views (query examples, sample outputs)
   - Section 6: Anomaly Detection (running checks, interpreting results)
   - Section 7: Dashboard Access (Airflow UI, DBeaver, pgAdmin)
   - Section 8: Troubleshooting (5 common issues + diagnostics)
   - Section 9: Customization (adding checks and views)
   - Section 10: Production Checklist
   - Section 11: Support & Resources

5. **Sprint 3 Week 3 Completion Report** (`docs/SPRINT_3_WEEK_3_COMPLETION_REPORT.md`, 640 lines)
   - Executive summary and key achievements
   - Comprehensive deliverables summary (2,006 lines total)
   - Testing results (all 5 tests passed)
   - Performance metrics (queries <50ms, anomaly detection <2s)
   - Lessons learned and technical decisions
   - Production readiness assessment
   - Next steps and recommendations

6. **Environment Configuration** (`airflow/.env`)
   - Added Slack webhook URL placeholders
   - Added email recipient configuration
   - Added SMTP settings for Gmail/SendGrid/AWS SES
   - All credentials in environment variables (gitignored)

### Database Monitoring Status:

| View | Status | Query Time | Rows |
|------|--------|-----------|------|
| vw_database_metrics | ✅ Operational | 12ms | 11 |
| vw_data_quality_checks | ✅ Operational | 45ms | 9 (all OK) |
| vw_monthly_event_trends | ✅ Operational | 8ms | 24 |
| vw_database_health | ✅ Operational | 15ms | 1 |

### Data Quality Summary (2025-11-07):

| Check | Result | Severity |
|-------|--------|----------|
| Missing ev_id | 0 events | ✅ OK |
| Missing ev_date | 0 events | ✅ OK |
| Missing coordinates | 14,884 events | ✅ OK (historical data) |
| Invalid coordinates | 0 events | ✅ OK |
| Orphaned aircraft | 0 records | ✅ OK |
| Orphaned findings | 0 records | ✅ OK |
| Orphaned narratives | 0 records | ✅ OK |
| Duplicate events | 0 duplicates | ✅ OK |

### Production Readiness:

- ✅ Slack alerts configured (placeholders in .env)
- ✅ Email alerts configured (SMTP settings in .env)
- ✅ Monitoring views created and tested
- ✅ Anomaly detection tested (0 anomalies found)
- ✅ Documentation complete (754-line setup guide)
- ✅ No credentials committed to git
- ✅ **READY for December 1st, 2025 first production run**

---

## Previous Sprint Status: Phase 1 Sprint 2

**Objective**: Historical Data Integration + Query Optimization
**Progress**: ✅ 100% COMPLETE

### Completed ✅:

1. **Ownership Model Implemented**
   - All database objects transferred to parobek user
   - Scripts updated to run without sudo (except initial setup)
   - `scripts/transfer_ownership.sql` created for automated ownership transfer

2. **Setup Infrastructure**
   - `scripts/setup_database.sh` v2.0.0 created (8-step automated setup)
   - Minimal sudo requirements (only for database creation and extensions)
   - Tested and verified on existing database

3. **Query Optimization**
   - 6 materialized views created:
     - `mv_yearly_stats` - Accident statistics by year
     - `mv_state_stats` - State-level statistics
     - `mv_aircraft_stats` - Aircraft make/model statistics (971 aircraft types)
     - `mv_decade_stats` - Decade-level trends
     - `mv_crew_stats` - Crew certification statistics
     - `mv_finding_stats` - Investigation finding patterns (861 distinct findings)
   - 59 indexes created (30 base + 29 from materialized views + performance indexes)
   - `refresh_all_materialized_views()` function for easy MV updates
   - All tables analyzed for query planner

4. **Historical Data Integration**
   - Load tracking system operational (`load_tracking` table)
   - Staging table infrastructure created (11 staging tables in `staging` schema)
   - Pre2008.mdb loaded successfully (63,000 duplicate events handled, ~3,000 unique added)
   - Production-grade deduplication logic implemented
   - `scripts/load_with_staging.py` (597 lines) with one-time load guards

5. **PRE1982.MDB Analysis**
   - Comprehensive analysis documented in `docs/PRE1982_ANALYSIS.md`
   - Found: Incompatible schema (denormalized, 200+ columns, coded fields)
   - Decision: Defer to Sprint 3 (requires custom ETL, 8-16 hours)

6. **Performance Benchmarks**
   - 20 benchmark queries executed across 8 categories
   - All queries meet or exceed performance targets
   - p50: ~2ms, p95: ~13ms, p99: ~47ms (all well below targets)
   - 98.81% buffer cache hit ratio, 99.99% index usage
   - Documented in `docs/PERFORMANCE_BENCHMARKS.md` (450+ lines)

7. **Sprint 2 Completion Report**
   - Comprehensive completion report created
   - All deliverables documented with metrics
   - Lessons learned and technical achievements captured
   - Next steps defined for Sprint 3
   - Published as `SPRINT_2_COMPLETION_REPORT.md` (700+ lines)

### Deferred to Sprint 3 ⏸️:

1. **PRE1982 Integration**
   - Custom ETL for legacy schema
   - Estimated 8-16 hours development
   - Load 1962-1981 data (~87,000 events estimated)
   - Requires code mapping tables and denormalized → normalized transformation

---

## Database State (Current)

**Database**: ntsb_aviation
**Owner**: parobek
**Version**: PostgreSQL 18.0 on x86_64-pc-linux-gnu
**Size**: 512 MB (cleaned from 2,759 MB, 81.4% reduction)

### Row Counts (After Cleanup):

| Table | Rows | Notes |
|-------|------|-------|
| events | 92,771 | 1977-2025 (48 years with gaps) |
| aircraft | 94,533 | Multiple aircraft per event |
| flight_crew | 31,003 | Crew records |
| injury | 91,333 | Injury details (cleaned from 1.69M, 94.6% reduction) |
| findings | 101,243 | Investigation findings (cleaned from 698K, 85.5% reduction) |
| narratives | 52,880 | Accident narratives (cleaned from 424K, 80.8% reduction) |
| engines | 27,298 | Engine specifications |
| ntsb_admin | 29,773 | Administrative metadata |
| events_sequence | 29,173 | Event sequencing (cleaned from 638K, 95.4% reduction) |
| seq_of_events | 0 | Empty (not used in current data) |
| occurrences | 0 | Empty (not used in current data) |
| **TOTAL** | **~733K** | Across all tables (cleaned from 3.8M, 80.7% reduction) |

### Load Tracking Status:

| Database | Status | Events Loaded | Duplicates Found | Load Date |
|----------|--------|---------------|------------------|-----------|
| avall.mdb | completed | 29,773 | 0 | 2025-11-05 |
| Pre2008.mdb | completed | 92,771 | 63,000 | 2025-11-06 |
| PRE1982.MDB | pending | 0 | - | Not loaded yet |

### Data Coverage:

- **Current Data**: 2008-2025 (avall.mdb) - 29,773 unique events
- **Historical Data**: 2000-2007 (Pre2008.mdb) - ~3,000 unique events (rest were duplicates)
- **Legacy Data**: 1962-1981 (PRE1982.MDB) - Not yet integrated (incompatible schema)
- **Coverage Gaps**: 1982-1999, some years in 1977-2000

### Data Quality:

- ✅ Zero duplicate events in production tables
- ✅ Zero orphaned records (complete foreign key integrity)
- ✅ All coordinates within valid bounds (-90/90, -180/180)
- ✅ All dates within valid range (1962-present)
- ✅ Crew ages validated (10-120 years, 42 invalid ages converted to NULL)

### Materialized Views:

| View | Rows | Description |
|------|------|-------------|
| mv_yearly_stats | 47 | Accident statistics by year |
| mv_state_stats | 57 | State-level statistics |
| mv_aircraft_stats | 971 | Aircraft make/model statistics (5+ accidents) |
| mv_decade_stats | 6 | Decade-level trends |
| mv_crew_stats | 10 | Crew certification statistics |
| mv_finding_stats | 861 | Investigation findings (10+ occurrences) |

**Refresh Command**: `SELECT * FROM refresh_all_materialized_views();`

### Indexes:

- **Total**: 59 indexes
  - 30 base indexes (from schema.sql)
  - 20 indexes on materialized views
  - 9 additional performance indexes (composite, partial)

---

## Project Files & Scripts

### Core Database Scripts:

1. **`scripts/setup_database.sh`** (285 lines, v2.0.0)
   - Complete automated setup for GitHub users
   - Minimal sudo requirements (only initial setup)
   - 8-step process: check, initialize, create DB, enable extensions, transfer ownership, create schema, staging tables, load tracking
   - **Usage**: `./scripts/setup_database.sh [db_name] [db_user]`

2. **`scripts/transfer_ownership.sql`** (98 lines)
   - Transfers ownership of all database objects to specified user
   - Iterates through tables, sequences, views, materialized views, functions
   - **Usage**: `sudo -u postgres psql -d ntsb_aviation -f scripts/transfer_ownership.sql`

3. **`scripts/schema.sql`** (468 lines)
   - Complete PostgreSQL schema definition
   - 11 core tables with triggers, constraints, indexes
   - Generated columns (ev_year, ev_month, location_geom)
   - **Usage**: `psql -d ntsb_aviation -f scripts/schema.sql`

4. **`scripts/create_staging_tables.sql`** (279 lines)
   - Creates `staging` schema with 11 tables
   - Helper functions: `get_row_counts()`, `get_duplicate_stats()`, `clear_all_staging()`
   - 13 performance indexes for duplicate detection
   - **Usage**: `psql -d ntsb_aviation -f scripts/create_staging_tables.sql`

5. **`scripts/create_load_tracking.sql`** (123 lines)
   - Creates `load_tracking` table to prevent duplicate loads
   - Tracks database_name, load_status, events_loaded, duplicate_events_found
   - Initialized with 3 databases: avall.mdb (completed), Pre2008.mdb (completed), PRE1982.MDB (pending)
   - **Usage**: `psql -d ntsb_aviation -f scripts/create_load_tracking.sql`

6. **`scripts/load_with_staging.py`** (842 lines, updated 2025-11-07)
   - Production-grade ETL loader with staging table pattern
   - Checks load_tracking to prevent re-loads
   - Bulk COPY to staging, identifies duplicates, merges unique events
   - Loads ALL child records (even for duplicate events)
   - **All 7 Bug Fixes Applied (Sprint 3 Week 2)**:
     1. **INTEGER conversion** (22 columns): float64 → Int64, prevents "0.0" errors
     2. **TIME conversion** (ev_time): HHMM → HH:MM:SS format (825 → "08:25:00")
     3. **Generated columns**: Dynamic exclusion from INSERT (location_geom, search_vector)
     4. **Qualified columns**: Table-aliased JOIN references (s.ev_id, e.ev_id)
     5. **--force flag support**: Allow monthly re-loads with duplicate detection
   - **Usage**: `python scripts/load_with_staging.py --source avall.mdb [--force]`

7. **`scripts/optimize_queries.sql`** (358 lines, updated 2025-11-07)
   - Creates 6 materialized views
   - Creates 9 additional performance indexes
   - **Bug Fix #7**: Added UNIQUE indexes to 3 MVs for CONCURRENT refresh
   - Analyzes all tables and materialized views
   - `refresh_all_materialized_views()` function
   - **Usage**: `psql -d ntsb_aviation -f scripts/optimize_queries.sql`

8. **`scripts/cleanup_test_duplicates.sql`** (366 lines, created 2025-11-07)
   - Comprehensive cleanup script for test load duplicates
   - 8 phases: backup, identification, deletion, VACUUM FULL, MV refresh, analysis
   - **Results**: Removed 3,179,771 duplicates, reduced DB from 2,759 MB → 512 MB
   - Safe: Keeps first occurrence of duplicates, preserves all foreign key integrity
   - **Usage**: `psql -d ntsb_aviation -f scripts/cleanup_test_duplicates.sql`

9. **`scripts/validate_data.sql`** (384 lines)
   - Comprehensive data quality validation queries
   - 10 validation categories (row counts, primary keys, NULL values, data integrity, foreign keys, partitions, indexes, generated columns, database size)
   - **Usage**: `psql -d ntsb_aviation -f scripts/validate_data.sql`

### Airflow Files:

1. **`airflow/Dockerfile`** (38 lines, created 2025-11-06)
   - Custom Airflow 2.7.3 image with dependencies
   - Includes mdbtools, requests, psycopg2-binary
   - **Usage**: Built automatically by docker-compose

2. **`airflow/docker-compose.yml`** (196 lines, updated 2025-11-06)
   - 3-service orchestration: postgres-airflow, webserver, scheduler
   - LocalExecutor configuration (no Celery/Redis)
   - Volume mounts for DAGs, logs, PostgreSQL data
   - **Usage**: `cd airflow && docker compose up -d`

3. **`airflow/dags/monthly_sync_dag.py`** (1,467 lines, created 2025-11-07)
   - Production DAG for automated monthly NTSB data sync
   - 8 tasks: check updates, download, extract, backup, load, validate, refresh MVs, notify
   - **All 3 DAG-level bugs fixed**:
     - Bug #1: --force flag for monthly re-loads
     - Bug #6: System catalog compatibility (relname vs tablename)
     - Bug #7: UNIQUE indexes for CONCURRENT MV refresh
   - Smart skip logic (only downloads when file size changes)
   - Scheduled: 1st of month, 2 AM
   - **Baseline Run**: manual__2025-11-07T08:26:41+00:00 (1m 50s, 8/8 SUCCESS)
   - **Usage**: `docker compose exec webserver airflow dags trigger monthly_sync_ntsb_data`

### Documentation Files:

1. **`SPRINT_1_REPORT.md`** (251 lines)
   - Phase 1 Sprint 1 completion report
   - Documents initial PostgreSQL migration (478,631 rows)
   - Defines Sprint 2 next steps

2. **`SPRINT_2_COMPLETION_REPORT.md`** (700+ lines)
   - Phase 1 Sprint 2 completion report (✅ 100% COMPLETE)
   - All deliverables documented with comprehensive metrics
   - Performance benchmarks, lessons learned, technical achievements
   - Next steps defined for Sprint 3

3. **`docs/PRE1982_ANALYSIS.md`** (408 lines)
   - Comprehensive analysis of PRE1982.MDB structure
   - Schema comparison with current database
   - Integration complexity assessment
   - Recommendation: Defer to Sprint 4

4. **`docs/PERFORMANCE_BENCHMARKS.md`** (450+ lines)
   - Comprehensive performance analysis with 20 benchmark queries
   - Query performance across 8 categories (lookups, joins, spatial, etc.)
   - Buffer cache and index usage statistics
   - Before/after optimization comparison (30-114x speedup)
   - Recommendations and best practices

5. **`docs/AIRFLOW_SETUP_GUIDE.md`** (874 lines, created 2025-11-06)
   - Complete Airflow installation and configuration guide
   - Docker Compose setup, database connectivity, DAG development
   - Troubleshooting guide for common issues
   - Integration with existing ETL scripts

6. **`docs/SPRINT_3_WEEK_1_COMPLETION_REPORT.md`** (756 lines, created 2025-11-06)
   - Sprint 3 Week 1 deliverables (Airflow infrastructure)
   - PostgreSQL network configuration resolution
   - Hello-world DAG verification
   - Complete setup documentation

7. **`docs/SPRINT_3_WEEK_2_COMPLETION_REPORT.md`** (896 lines, created 2025-11-07)
   - Sprint 3 Week 2 deliverables (Production DAG + 7 bug fixes)
   - Comprehensive documentation of all 7 bugs discovered
   - Database cleanup results (3.2M duplicates removed)
   - Baseline DAG execution verification
   - Complete file inventory and testing results

8. **`CLAUDE.local.md`** (this file)
   - Current project state and development guidance
   - "NO SUDO" principle documentation
   - Sprint status and database metrics

---

## Current Sprint: Phase 1 Sprint 3

**Objective**: Apache Airflow ETL Pipeline for Automated Monthly Updates
**Progress**: Week 1 Complete (✅ 100%)

### Sprint 3 Week 1: Infrastructure Setup (2025-11-06)

**Status**: ✅ 100% COMPLETE
**Blocker**: ✅ RESOLVED (PostgreSQL network configuration completed)

#### Completed ✅:

1. **Airflow Installation**
   - Docker Compose setup (3 services: postgres-airflow, webserver, scheduler)
   - LocalExecutor configured (no Celery/Redis needed)
   - Web UI accessible at http://localhost:8080
   - All services operational and healthy

2. **Database Connectivity**
   - Connection to ntsb_aviation database configured
   - Connection ID: `ntsb_aviation_db` created
   - ✅ PostgreSQL configured to accept Docker network connections (172.17.0.0/16, 172.19.0.0/16)
   - ✅ All Airflow services can connect to ntsb_aviation database

3. **Hello-World DAG**
   - Tutorial DAG with 5 tasks created (`airflow/dags/hello_world_dag.py`, 173 lines)
   - Demonstrates BashOperator, PythonOperator, PostgresOperator
   - ✅ All 5 tasks executed successfully
   - ✅ Query returned 92,771 events from database
   - ✅ DAG execution verified end-to-end

4. **Documentation**
   - Airflow Setup Guide created (`docs/AIRFLOW_SETUP_GUIDE.md`, 874 lines)
   - Week 1 Completion Report published (`docs/SPRINT_3_WEEK_1_COMPLETION_REPORT.md`)
   - Comprehensive troubleshooting guide included

#### Files Created:
- `airflow/docker-compose.yml` (196 lines)
- `airflow/dags/hello_world_dag.py` (173 lines)
- `airflow/.env` (32 lines, gitignored)
- `docs/AIRFLOW_SETUP_GUIDE.md` (874 lines)
- `docs/SPRINT_3_WEEK_1_COMPLETION_REPORT.md` (comprehensive report)

#### Resolved Issues ✅:

**PostgreSQL Network Configuration** (RESOLVED 2025-11-06):
- ✅ PostgreSQL configured to listen on all interfaces (`listen_addresses = '*'`)
- ✅ Docker bridge network access granted (172.17.0.0/16, 172.19.0.0/16)
- ✅ Airflow containers can connect to ntsb_aviation database
- ✅ All 5 hello_world DAG tasks completed successfully
- ✅ Database query verified: 92,771 events returned

**Known Issues**:
- **LOW PRIORITY - Cosmetic**:
  - Scheduler shows "unhealthy" status (healthcheck timing issue, functionally working)
  - Docker Compose `version` deprecation warning

### Sprint 3 Week 2: First Production DAG (2025-11-07)

**Status**: ✅ 100% COMPLETE

#### Completed ✅:

1. **monthly_sync_dag.py Created**
   - Complete 8-task DAG for automated monthly data sync
   - Integration with existing load_with_staging.py script
   - Smart skip logic (only download when file size changes)
   - Database backup before load
   - Data quality validation
   - Materialized view refresh
   - Success notifications

2. **Load Script Integration Fix** (2025-11-07)
   - **Issue**: load_new_data task failing because avall.mdb was already loaded
   - **Root Cause**: DAG was calling load_with_staging.py WITHOUT --force flag
   - **Fix**: Added --force flag for monthly re-loads (safe due to duplicate detection)
   - **Files Modified**: `airflow/dags/monthly_sync_dag.py` (lines 733-748, 776-778)
   - **Why Safe**:
     - load_with_staging.py has built-in duplicate detection via staging tables
     - Only new events are merged into production (duplicates skipped)
     - avall.mdb is designed for monthly updates (unlike historical databases)
   - **Impact**: load_new_data task should now succeed on re-runs

3. **Data Type Conversion Fix #1 - INTEGER** (2025-11-07) ✅ CRITICAL FIX
   - **Issue**: PostgreSQL COPY failing with "invalid input syntax for type integer: '0.0'"
   - **Root Cause**: Pandas writing float64 as "0.0" to CSV, but PostgreSQL INTEGER rejects decimal points
   - **Affected Columns**: 22 INTEGER columns across 7 tables (wx_temp, crew_age, etc.)
   - **Solution**: Added explicit float-to-integer conversion using pandas Int64 dtype
   - **Files Modified**:
     - `scripts/load_with_staging.py` (lines 106-146, 334-346)
     - Added INTEGER_COLUMNS mapping (41 lines)
     - Added conversion logic in clean_dataframe() (13 lines)
   - **Testing**: Verified with unit tests (test_int64_fix.py)
   - **Documentation**: `/tmp/NTSB_Datasets/DATA_TYPE_FIX_REPORT.md` (comprehensive report)
   - **Impact**:
     - ✅ All 11 tables now load successfully
     - ✅ INTEGER columns accept whole numbers without decimals
     - ✅ DECIMAL columns preserve precision (wx_vis, dec_latitude, etc.)
     - ✅ NULL values handled correctly with pd.NA
     - ✅ Monthly sync DAG can run end-to-end

4. **Data Type Conversion Fix #2 - TIME** (2025-11-07) ✅ CRITICAL FIX
   - **Issue**: PostgreSQL COPY failing with "invalid input syntax for type time: '825.0'"
   - **Root Cause**: NTSB stores times as HHMM integers (825 = 8:25 AM), PostgreSQL TIME requires HH:MM:SS format
   - **Affected Column**: `ev_time` in `events` table (TIME data type)
   - **Solution**: Added HHMM → HH:MM:SS conversion function with validation
   - **Files Modified**:
     - `scripts/load_with_staging.py` (799 lines total, +161 lines from original 638)
     - Lines 77-124: Added `convert_ntsb_time_to_postgres()` function
     - Lines 197-203: Added TIME_COLUMNS mapping
     - Lines 408-426: Added TIME conversion in clean_dataframe()
   - **Conversion Logic**:
     - Extract hours: `825 // 100 = 8`
     - Extract minutes: `825 % 100 = 25`
     - Validate ranges: 0-23 hours, 0-59 minutes
     - Format with leading zeros: `"08:25:00"`
     - Handle NULL/NaN: return None (preserved as PostgreSQL NULL)
     - Invalid times (9999, -100): return None
   - **Testing**: Comprehensive unit tests (22/22 passed)
     - Valid times: 0 → "00:00:00", 825 → "08:25:00", 2359 → "23:59:00"
     - Edge cases: NaN → None, invalid hours/minutes → None
     - CSV export format verified (PostgreSQL COPY compatible)
   - **Documentation**: `/tmp/NTSB_Datasets/TIME_CONVERSION_FIX_REPORT.md` (comprehensive 450+ line report)

5. **Data Type Conversion Fix #3 - Generated Columns** (2025-11-07) ✅ CRITICAL FIX
   - **Issue**: PostgreSQL INSERT failing with "cannot insert a non-DEFAULT value into column 'search_vector'" - Column "search_vector" is a generated column
   - **Root Cause**: Script used `SELECT *` in INSERT operations, which includes generated columns that cannot be explicitly inserted
   - **Affected Tables**:
     - `events` (location_geom - GEOGRAPHY point computed from lat/lon)
     - `narratives` (search_vector - TSVECTOR for full-text search)
   - **Solution**: Dynamic column list query excluding generated columns
   - **Files Modified**:
     - `scripts/load_with_staging.py` (+52 lines total)
     - Lines 651-681: Added `_get_insertable_columns()` helper method
     - Lines 579-596: Modified `merge_unique_events()` to use explicit column list
     - Lines 598-643: Modified `load_child_table()` to use explicit column list
   - **Implementation**:
     - Query information_schema at runtime for columns where is_generated = 'NEVER'
     - Build explicit column list: `INSERT INTO table (col1, col2) SELECT col1, col2 FROM staging`
     - Automatically handles schema changes (new columns, new generated columns)
   - **Generated Columns**:
     - `events.location_geom`: ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)
     - `narratives.search_vector`: to_tsvector('english', narr_accp || ' ' || narr_cause)
   - **Testing**: Comprehensive unit tests and verification
     - events table: 35 insertable columns (excludes location_geom)
     - narratives table: 6 insertable columns (excludes search_vector)
     - All 11 tables verified (2 with generated columns, 9 without)
     - Python syntax check passed
     - Ruff linting passed (format + check)
     - Integration test passed (test_generated_columns.py)
   - **Documentation**: `/tmp/NTSB_Datasets/GENERATED_COLUMN_FIX_REPORT.md` (comprehensive 8,900+ word report)
   - **Impact**:
     - ✅ INSERT operations work for all tables
     - ✅ Generated columns automatically computed by PostgreSQL
     - ✅ Future-proof - handles new generated columns automatically
     - ✅ No performance impact (<0.1% overhead for schema query)
   - **Impact**:
     - ✅ INSERT operations work for all tables
     - ✅ Generated columns automatically computed by PostgreSQL
     - ✅ Future-proof - handles new generated columns automatically

6. **Bug Fix #4 - Qualified Column Names** (2025-11-07) ✅ CRITICAL FIX
   - **Issue**: Ambiguous column reference "ev_id" in JOIN queries
   - **Fix**: Table-aliased column references (s.ev_id, e.ev_id)
   - **Files Modified**: `scripts/load_with_staging.py` (lines 615-640)

7. **Bug Fix #5 - System Catalog Compatibility** (2025-11-07) ✅ CRITICAL FIX
   - **Issue**: pg_stat_user_tables uses 'relname', not 'tablename'
   - **Fix**: Aliased relname as tablename for compatibility
   - **Files Modified**: `airflow/dags/monthly_sync_dag.py` (line 880)

8. **Database Cleanup** (2025-11-07) ✅ COMPLETE
   - **Issue**: 3.2M duplicate records from test loads, database size 2,759 MB
   - **Solution**: Created comprehensive cleanup script with 8 phases
   - **Results**:
     - injury: deleted 1,602,005 duplicates (94.6% reduction)
     - findings: deleted 597,105 duplicates (85.5%)
     - narratives: deleted 371,705 duplicates (80.8%)
     - events_sequence: deleted 608,956 duplicates (95.4%)
     - Database: 2,759 MB → 512 MB (81.4% reduction)
   - **Files Created**: `scripts/cleanup_test_duplicates.sql` (366 lines)

9. **Baseline DAG Run** (2025-11-07) ✅ COMPLETE
   - **Run ID**: manual__2025-11-07T08:26:41+00:00
   - **Duration**: 1m 50s
   - **Tasks**: 8/8 SUCCESS
   - **Validation**: All data quality checks passed, 0 duplicates, 100% foreign key integrity

10. **Commit and Push** (2025-11-07) ✅ COMPLETE
    - **Commit**: 8873f3a "feat(sprint3-week2): complete production Airflow DAG with 7 critical bug fixes"
    - **Files Changed**: 14 files, 5,125 insertions, 177 deletions
    - **Pushed**: origin/main successfully
    - **Documentation**: Comprehensive commit message documenting all 7 bugs, cleanup, baseline run

#### Sprint 3 Week 2 Summary:

**All Deliverables Complete** ✅:
- monthly_sync_dag.py (1,467 lines) - Production-ready with baseline run
- All 7 bugs fixed (INTEGER, TIME, generated columns, qualified columns, --force, system catalog, MV CONCURRENT)
- Database cleaned (3.2M duplicates removed, 81% size reduction)
- Documentation complete (3 reports: setup guide, week 1, week 2)
- Baseline run verified (1m 50s, 8/8 tasks SUCCESS)
- All code committed and pushed to repository

---

## Development Conventions

### Git Workflow:

- **Feature branches**: `feature/sprint-X-Y-description`
- **Commit messages**: Conventional commits (feat:, fix:, docs:, refactor:, test:)
- **Never commit** unless user explicitly requests
- **Run format/lint** before marking task complete

### Code Quality:

- **Python**: ruff format, ruff check, type hints
- **SQL**: Consistent formatting, comments for complex queries
- **Bash**: shellcheck compliance, error handling (set -e)

### Testing:

- **Data validation**: Always run validate_data.sql after major changes
- **Performance testing**: Benchmark queries after optimization changes
- **Staging tests**: Test with small datasets before full loads

### Documentation:

- **README.md**: User-facing, Quick Start, installation
- **CHANGELOG.md**: Track all significant changes
- **Sprint reports**: Document completion, metrics, lessons learned
- **CLAUDE.local.md**: Keep updated with current state

---

## Troubleshooting

### Permission Errors:

**Problem**: `ERROR: permission denied for table events`

**Solution**:
```bash
# Transfer ownership to current user
sudo -u postgres psql -d ntsb_aviation -f scripts/transfer_ownership.sql

# Verify ownership
psql -d ntsb_aviation -c "\dt"
```

### Duplicate Load Prevention:

**Problem**: Accidentally trying to reload PRE1982 or Pre2008

**Solution**: Load tracking system will prompt for confirmation:
```
⚠ Pre2008.mdb already loaded on 2025-11-06 03:15:23
⚠ Historical databases should only be loaded once!
Continue anyway? (yes/no):
```

### Materialized View Refresh:

**Problem**: Materialized views out of date after data load

**Solution**:
```sql
-- Refresh all materialized views concurrently
SELECT * FROM refresh_all_materialized_views();

-- Check refresh times
SELECT
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
FROM pg_matviews
WHERE schemaname = 'public';
```

### Staging Table Cleanup:

**Problem**: Staging tables need to be cleared after failed load

**Solution**:
```sql
-- Clear all staging tables
SELECT clear_all_staging();

-- Verify staging is empty
SELECT * FROM get_row_counts();
```

---

## Performance Targets

### Query Performance:

- **p50 latency**: <10ms for simple queries
- **p95 latency**: <100ms for complex analytical queries
- **p99 latency**: <500ms for heavy aggregations

### Data Load Performance:

- **avall.mdb**: ~30 seconds for full load (29,773 events)
- **Pre2008.mdb**: ~90 seconds for full load (906,176 rows staging)
- **Throughput**: 15,000-45,000 rows/sec (varies by table)

### Database Size:

- **Current**: 966 MB
- **With PRE1982**: Estimated 1.2-1.5 GB
- **Full historical (1962-2025)**: Estimated 1.5-2.0 GB

---

## Quick Reference Commands

```bash
# Setup new database (GitHub users)
./scripts/setup_database.sh

# Load current data
source .venv/bin/activate
python scripts/load_with_staging.py --source avall.mdb

# Load historical data
python scripts/load_with_staging.py --source Pre2008.mdb

# Optimize database
psql -d ntsb_aviation -f scripts/optimize_queries.sql

# Validate data
psql -d ntsb_aviation -f scripts/validate_data.sql

# Check load status
psql -d ntsb_aviation -c "SELECT * FROM load_tracking ORDER BY load_completed_at DESC;"

# Refresh materialized views
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"

# Database size
psql -d ntsb_aviation -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));"

# Table row counts
psql -d ntsb_aviation -c "
  SELECT schemaname, tablename, n_live_tup as rows
  FROM pg_stat_user_tables
  WHERE schemaname = 'public'
  ORDER BY n_live_tup DESC;
"
```

---

## Memory Bank Updates

**When to update this file**:
- After completing each sprint
- After major architectural changes
- After discovering new gotchas or troubleshooting patterns
- When "NO SUDO" principle is violated and needs fixing

**How to update**:
1. Read this file first to understand current state
2. Update relevant sections (Sprint Status, Database State, etc.)
3. Maintain version history in git (don't just overwrite)
4. Keep concise - this is guidance, not a novel

---

**End of CLAUDE.local.md**
