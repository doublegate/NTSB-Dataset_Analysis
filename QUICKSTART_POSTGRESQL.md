# NTSB Aviation Database - PostgreSQL Migration Quick Start

**Phase 1 Sprint 1: PostgreSQL Migration**

This guide will help you complete the PostgreSQL database migration in ~45 minutes.

---

## Prerequisites Check

```bash
# 1. Verify PostgreSQL is installed
psql --version
# Expected: PostgreSQL 18.0 (or similar)

# 2. Verify Python virtual environment
source .venv/bin/activate
python --version
# Expected: Python 3.x

# 3. Verify CSV data exists
ls -lh data/avall-*.csv | wc -l
# Expected: 11 files
```

---

## Step 1: Database Setup (15 minutes)

### Option A: Automated Setup (Recommended)

```bash
# Run the automated setup script
./scripts/setup_database.sh

# This will:
#   - Start PostgreSQL service
#   - Create 'ntsb_aviation' database
#   - Enable PostGIS, pg_trgm, pgcrypto extensions
#   - Create 11 tables + 7 partitions
#   - Create 29 indexes
#   - Set up user permissions
```

### Option B: Manual Setup

If the automated script has permission issues:

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
sudo -u postgres psql -d ntsb_aviation -c "GRANT ALL PRIVILEGES ON DATABASE ntsb_aviation TO $USER;"
sudo -u postgres psql -d ntsb_aviation -c "GRANT ALL PRIVILEGES ON SCHEMA public TO $USER;"
sudo -u postgres psql -d ntsb_aviation -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $USER;"
sudo -u postgres psql -d ntsb_aviation -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $USER;"

# 6. Verify setup
psql -d ntsb_aviation -c "\dt"  # Should show 11 tables
psql -d ntsb_aviation -c "\di"  # Should show 29+ indexes
```

---

## Step 2: Data Loading (10-15 minutes)

```bash
# 1. Activate Python environment
source .venv/bin/activate

# 2. Load data from CSV files
python scripts/load_to_postgres.py

# Expected output:
#   - Loading 11 tables in dependency order
#   - Progress updates for each table
#   - Total: ~478,631 rows loaded
#   - Duration: 5-10 minutes
#   - Success confirmation

# Optional: Load only specific tables
python scripts/load_to_postgres.py --skip-tables Occurrences seq_of_events
```

**Troubleshooting**:

If you get connection errors:
```bash
# Check PostgreSQL is running
systemctl status postgresql

# Test connection manually
psql -d ntsb_aviation -c "SELECT 1;"
```

If you get permission errors:
```bash
# Re-grant permissions
sudo -u postgres psql -d ntsb_aviation -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $USER;"
```

---

## Step 3: Validation (5 minutes)

### Data Quality Validation

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
```

### Quick Manual Checks

```bash
# Connect to database
psql -d ntsb_aviation

# Check row counts
SELECT 'events' AS table_name, COUNT(*) FROM events;
-- Expected: ~29,773 rows

# Check data range
SELECT MIN(ev_date), MAX(ev_date) FROM events;
-- Expected: 2008-01-01 to 2025-10-30

# Check spatial data
SELECT COUNT(*) FROM events WHERE location_geom IS NOT NULL;
-- Expected: ~26,644 events with coordinates

# Test a simple query
SELECT ev_id, ev_date, ev_state, ev_highest_injury
FROM events
WHERE ev_state = 'CA' AND ev_year = 2023
LIMIT 10;

# Exit
\q
```

---

## Step 4: Performance Testing (10 minutes)

```bash
# Run performance benchmarks
psql -d ntsb_aviation -f scripts/test_performance.sql > performance_report.txt

# Review results
less performance_report.txt

# Expected results:
#   - Simple lookups: <10ms
#   - Indexed queries: <50ms
#   - Join queries: <100ms
#   - Spatial queries: <100ms
#   - Aggregate queries: <100ms
```

### Performance Optimization (if needed)

```bash
# If queries are slow, run ANALYZE
psql -d ntsb_aviation -c "ANALYZE;"

# Check index usage
psql -d ntsb_aviation -c "
SELECT
    schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC
LIMIT 10;
"

# Check cache hit ratio (should be >95%)
psql -d ntsb_aviation -c "
SELECT
    ROUND(100.0 * sum(blks_hit) / NULLIF(sum(blks_hit) + sum(blks_read), 0), 2) AS cache_hit_ratio
FROM pg_stat_database
WHERE datname = 'ntsb_aviation';
"
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] PostgreSQL service is running
- [ ] Database `ntsb_aviation` exists
- [ ] 11 tables created (events, aircraft, Flight_Crew, etc.)
- [ ] 7 partitions for events table (1960s-2020s)
- [ ] 29+ indexes created
- [ ] ~29,773 events loaded
- [ ] ~478,631 total rows loaded
- [ ] Foreign key relationships valid (0 orphaned records)
- [ ] Query performance <100ms for indexed queries
- [ ] Spatial queries working (PostGIS enabled)
- [ ] Full-text search working (pg_trgm enabled)

---

## Common Issues

### Issue 1: PostgreSQL Service Not Running

```bash
# Start service
sudo systemctl start postgresql

# Enable auto-start on boot
sudo systemctl enable postgresql
```

### Issue 2: Database Connection Refused

```bash
# Check service status
systemctl status postgresql

# Check if port 5432 is listening
sudo netstat -tlnp | grep 5432

# Check PostgreSQL logs
sudo journalctl -u postgresql -n 50
```

### Issue 3: Permission Denied

```bash
# Grant all privileges to your user
sudo -u postgres psql -d ntsb_aviation -c "
    GRANT ALL PRIVILEGES ON DATABASE ntsb_aviation TO $USER;
    GRANT ALL PRIVILEGES ON SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $USER;
"
```

### Issue 4: Slow Query Performance

```bash
# Run ANALYZE to update statistics
psql -d ntsb_aviation -c "ANALYZE;"

# Check if indexes are being used
psql -d ntsb_aviation -c "
    EXPLAIN ANALYZE
    SELECT * FROM events WHERE ev_state = 'CA' LIMIT 10;
"
# Look for "Index Scan" in output (not "Seq Scan")
```

### Issue 5: Out of Memory During Load

```bash
# Load tables one at a time
python scripts/load_to_postgres.py --skip-tables aircraft Flight_Crew injury Findings Events_Sequence engines narratives NTSB_Admin
python scripts/load_to_postgres.py --skip-tables events Flight_Crew injury Findings Events_Sequence engines narratives NTSB_Admin
# ... continue for each table

# Or reduce chunk size
python scripts/load_to_postgres.py --chunk-size 500
```

---

## Next Steps

Once validation and performance testing are complete:

1. **Review Reports**:
   - `validation_report.txt` - Data quality metrics
   - `performance_report.txt` - Query performance benchmarks
   - `SPRINT_1_REPORT.md` - Complete implementation summary

2. **Proceed to Sprint 2**: Apache Airflow ETL Pipeline
   - See `to-dos/PHASE_1_FOUNDATION.md` (lines 233-410)
   - Set up Airflow for automated data updates
   - Implement monthly NTSB data ingestion

3. **Optional Enhancements**:
   - Load historical data (Pre2008.mdb, PRE1982.MDB)
   - Refresh materialized views: `REFRESH MATERIALIZED VIEW mv_yearly_stats;`
   - Set up automated backups
   - Configure connection pooling for production

---

## Getting Help

- **Documentation**: See `SPRINT_1_REPORT.md` for detailed implementation guide
- **Schema Reference**: See `scripts/schema.sql` for complete database structure
- **Validation**: See `scripts/validate_data.sql` for data quality checks
- **Performance**: See `scripts/test_performance.sql` for benchmarking queries

---

## Success Confirmation

If all steps completed successfully, you should be able to run:

```bash
# Test query
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

Expected output: Yearly statistics for 2020-2025 with sub-second response time.

**Congratulations!** Your NTSB Aviation Database PostgreSQL migration is complete. 🎉

---

**Last Updated**: 2025-11-05
**Sprint**: Phase 1 Sprint 1 (PostgreSQL Migration)
**Status**: Ready for Execution
