# CLAUDE.local.md - Current Project State & Development Guidance

**Last Updated**: 2025-11-06
**Sprint**: Phase 1 Sprint 2 (Query Optimization & Historical Data Integration)
**Status**: ✅ 100% COMPLETE

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

## Current Sprint Status: Phase 1 Sprint 2

**Objective**: Historical Data Integration + Query Optimization
**Progress**: ~95% complete

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
**Size**: 966 MB

### Row Counts:

| Table | Rows | Notes |
|-------|------|-------|
| events | 92,771 | 1977-2025 (48 years with gaps) |
| aircraft | 94,533 | Multiple aircraft per event |
| flight_crew | 31,003 | Crew records |
| injury | 169,337 | Injury details |
| findings | 69,838 | Investigation findings |
| narratives | 27,485 | Accident narratives |
| engines | 27,298 | Engine specifications |
| ntsb_admin | 29,773 | Administrative metadata |
| events_sequence | 63,852 | Event sequencing |
| seq_of_events | 0 | Empty (not used in current data) |
| occurrences | 0 | Empty (not used in current data) |
| **TOTAL** | **726,969** | Across all tables |

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

6. **`scripts/load_with_staging.py`** (597 lines)
   - Production-grade ETL loader with staging table pattern
   - Checks load_tracking to prevent re-loads
   - Bulk COPY to staging, identifies duplicates, merges unique events
   - Loads ALL child records (even for duplicate events)
   - **Usage**: `python scripts/load_with_staging.py --source avall.mdb`

7. **`scripts/optimize_queries.sql`** (324 lines, updated)
   - Creates 6 materialized views
   - Creates 9 additional performance indexes
   - Analyzes all tables and materialized views
   - `refresh_all_materialized_views()` function
   - **Usage**: `psql -d ntsb_aviation -f scripts/optimize_queries.sql`

8. **`scripts/validate_data.sql`** (384 lines)
   - Comprehensive data quality validation queries
   - 10 validation categories (row counts, primary keys, NULL values, data integrity, foreign keys, partitions, indexes, generated columns, database size)
   - **Usage**: `psql -d ntsb_aviation -f scripts/validate_data.sql`

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
   - Recommendation: Defer to Sprint 3

4. **`docs/PERFORMANCE_BENCHMARKS.md`** (450+ lines)
   - Comprehensive performance analysis with 20 benchmark queries
   - Query performance across 8 categories (lookups, joins, spatial, etc.)
   - Buffer cache and index usage statistics
   - Before/after optimization comparison (30-114x speedup)
   - Recommendations and best practices

5. **`CLAUDE.local.md`** (this file)
   - Current project state and development guidance
   - "NO SUDO" principle documentation
   - Sprint status and database metrics

---

## Next Sprint: Phase 1 Sprint 3

**Objective**: Apache Airflow ETL Pipeline for Automated Monthly Updates

**Planned Deliverables**:

1. **Airflow DAG Infrastructure**
   - 5 production DAGs:
     - `monthly_sync_dag.py` - Automated avall.mdb updates
     - `data_transformation_dag.py` - Data cleaning and normalization
     - `quality_check_dag.py` - Automated validation
     - `mv_refresh_dag.py` - Materialized view updates
     - `feature_engineering_dag.py` - ML feature preparation

2. **Monitoring & Alerting**
   - Email notifications for failures
   - Slack integration for status updates
   - Dashboard for ETL metrics

3. **PRE1982 Integration** (if time permits)
   - Custom ETL for legacy schema
   - Mapping coded fields to modern taxonomy
   - Load 1962-1981 data (~87,000 events)

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
