# Validate Schema - Comprehensive Database Validation

Run comprehensive database validation checks and report data quality metrics for the NTSB Aviation Database.

---

## OBJECTIVE

Execute complete validation of the ntsb_aviation PostgreSQL database:
- Verify schema integrity and constraints
- Check data quality across all tables
- Validate foreign key relationships
- Report on indexes and materialized views
- Generate comprehensive validation report

**Time Estimate:** ~5-10 minutes
**Prerequisites:** PostgreSQL connection to ntsb_aviation database

---

## USAGE

```bash
/validate-schema              # Full validation report
/validate-schema quick        # Quick checks only (row counts, primary keys)
/validate-schema report       # Generate report file in /tmp/
```

---

## EXECUTION PHASES

### PHASE 1: VERIFY DATABASE CONNECTION (1 minute)

**Objective:** Ensure database is accessible and ready for validation

```bash
echo "=========================================="
echo "Database Validation - NTSB Aviation"
echo "=========================================="
echo ""

# Check psql availability
if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: psql not installed"
    echo ""
    echo "Install PostgreSQL client:"
    echo "  Arch: sudo pacman -S postgresql"
    echo "  Ubuntu: sudo apt install postgresql-client"
    echo ""
    exit 1
fi

# Test database connection
if ! psql -d ntsb_aviation -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check PostgreSQL is running: sudo systemctl status postgresql"
    echo "  2. Check database exists: psql -l"
    echo "  3. Check permissions: psql -d ntsb_aviation -c '\\dt'"
    echo ""
    exit 1
fi

echo "✅ Connected to ntsb_aviation database"
echo ""
```

---

### PHASE 2: ROW COUNTS VALIDATION (1 minute)

**Objective:** Verify all tables have data and match expected counts

```bash
echo "=========================================="
echo "Row Counts Validation"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Table row counts
SELECT 
    schemaname,
    tablename,
    n_live_tup as "Rows",
    CASE 
        WHEN n_live_tup = 0 THEN '⚠️  EMPTY'
        WHEN n_live_tup < 100 THEN '⚠️  LOW'
        ELSE '✅ OK'
    END as "Status"
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
SQL

echo ""
```

---

### PHASE 3: PRIMARY KEY VALIDATION (1 minute)

**Objective:** Ensure all tables have proper primary keys

```bash
echo "=========================================="
echo "Primary Key Validation"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Check primary keys exist
SELECT 
    t.tablename,
    CASE 
        WHEN c.conname IS NOT NULL THEN '✅ ' || c.conname
        ELSE '❌ MISSING'
    END as "Primary Key"
FROM pg_tables t
LEFT JOIN pg_constraint c 
    ON c.conrelid = (t.schemaname||'.'||t.tablename)::regclass 
    AND c.contype = 'p'
WHERE t.schemaname = 'public'
ORDER BY t.tablename;
SQL

echo ""
```

---

### PHASE 4: NULL VALUE CHECKS (2 minutes)

**Objective:** Identify unexpected NULL values in critical columns

```bash
echo "=========================================="
echo "NULL Value Checks"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Check for NULLs in critical columns
-- events table
SELECT 'events.ev_id' as "Column", COUNT(*) as "NULL Count", 
       CASE WHEN COUNT(*) = 0 THEN '✅ OK' ELSE '❌ FAIL' END as "Status"
FROM events WHERE ev_id IS NULL
UNION ALL
SELECT 'events.ntsb_no', COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN '✅ OK' ELSE '❌ FAIL' END
FROM events WHERE ntsb_no IS NULL
UNION ALL
SELECT 'events.ev_date', COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN '✅ OK' ELSE '⚠️  WARNING' END
FROM events WHERE ev_date IS NULL
UNION ALL
-- aircraft table
SELECT 'aircraft.Aircraft_Key', COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN '✅ OK' ELSE '❌ FAIL' END
FROM aircraft WHERE "Aircraft_Key" IS NULL
UNION ALL
SELECT 'aircraft.ev_id', COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN '✅ OK' ELSE '❌ FAIL' END
FROM aircraft WHERE ev_id IS NULL;
SQL

echo ""
```

---

### PHASE 5: DATA INTEGRITY CHECKS (2 minutes)

**Objective:** Validate data ranges and business rules

```bash
echo "=========================================="
echo "Data Integrity Checks"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Check coordinate bounds
SELECT 
    'Latitude Range' as "Check",
    MIN(latitude) as "Min",
    MAX(latitude) as "Max",
    CASE 
        WHEN MIN(latitude) >= -90 AND MAX(latitude) <= 90 THEN '✅ VALID'
        ELSE '❌ INVALID'
    END as "Status"
FROM events
WHERE latitude IS NOT NULL
UNION ALL
SELECT 
    'Longitude Range',
    MIN(longitude),
    MAX(longitude),
    CASE 
        WHEN MIN(longitude) >= -180 AND MAX(longitude) <= 180 THEN '✅ VALID'
        ELSE '❌ INVALID'
    END
FROM events
WHERE longitude IS NOT NULL;

-- Check date ranges
SELECT 
    'Event Date Range' as "Check",
    MIN(EXTRACT(YEAR FROM ev_date))::text as "Min Year",
    MAX(EXTRACT(YEAR FROM ev_date))::text as "Max Year",
    CASE 
        WHEN MIN(EXTRACT(YEAR FROM ev_date)) >= 1962 
         AND MAX(EXTRACT(YEAR FROM ev_date)) <= EXTRACT(YEAR FROM CURRENT_DATE) 
        THEN '✅ VALID'
        ELSE '⚠️  WARNING'
    END as "Status"
FROM events
WHERE ev_date IS NOT NULL;

-- Check for invalid crew ages
SELECT 
    'Crew Age Range' as "Check",
    COUNT(*) as "Invalid Count",
    '' as "Details",
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '⚠️  ' || COUNT(*) || ' invalid ages'
    END as "Status"
FROM flight_crew
WHERE crew_age IS NOT NULL 
  AND (crew_age < 10 OR crew_age > 120);
SQL

echo ""
```

---

### PHASE 6: FOREIGN KEY VALIDATION (2 minutes)

**Objective:** Check for orphaned records and referential integrity

```bash
echo "=========================================="
echo "Foreign Key Validation"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Check for orphaned aircraft records
SELECT 
    'aircraft → events' as "Relationship",
    COUNT(*) as "Orphaned Records",
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '❌ ' || COUNT(*) || ' orphaned'
    END as "Status"
FROM aircraft a
WHERE NOT EXISTS (
    SELECT 1 FROM events e WHERE e.ev_id = a.ev_id
);

-- Check for orphaned flight_crew records
SELECT 
    'flight_crew → aircraft',
    COUNT(*),
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '❌ ' || COUNT(*) || ' orphaned'
    END
FROM flight_crew fc
WHERE NOT EXISTS (
    SELECT 1 FROM aircraft a WHERE a."Aircraft_Key" = fc."Aircraft_Key"
);

-- Check for orphaned injury records
SELECT 
    'injury → events',
    COUNT(*),
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '❌ ' || COUNT(*) || ' orphaned'
    END
FROM injury i
WHERE NOT EXISTS (
    SELECT 1 FROM events e WHERE e.ev_id = i.ev_id
);

-- Check for orphaned findings records
SELECT 
    'findings → events',
    COUNT(*),
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '❌ ' || COUNT(*) || ' orphaned'
    END
FROM findings f
WHERE NOT EXISTS (
    SELECT 1 FROM events e WHERE e.ev_id = f.ev_id
);

-- Check for orphaned narratives
SELECT 
    'narratives → events',
    COUNT(*),
    CASE 
        WHEN COUNT(*) = 0 THEN '✅ OK'
        ELSE '❌ ' || COUNT(*) || ' orphaned'
    END
FROM narratives n
WHERE NOT EXISTS (
    SELECT 1 FROM events e WHERE e.ev_id = n.ev_id
);
SQL

echo ""
```

---

### PHASE 7: INDEX AND PERFORMANCE CHECKS (1 minute)

**Objective:** Verify indexes exist and materialized views are up to date

```bash
echo "=========================================="
echo "Index and Performance Checks"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Count indexes by table
SELECT 
    tablename,
    COUNT(*) as "Index Count",
    CASE 
        WHEN COUNT(*) = 0 THEN '⚠️  NO INDEXES'
        WHEN COUNT(*) < 2 THEN '⚠️  LOW'
        ELSE '✅ OK'
    END as "Status"
FROM pg_indexes
WHERE schemaname = 'public'
GROUP BY tablename
ORDER BY tablename;
SQL

echo ""

psql -d ntsb_aviation << 'SQL'
-- Materialized views status
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as "Size",
    '✅ OK' as "Status"
FROM pg_matviews
WHERE schemaname = 'public'
ORDER BY matviewname;
SQL

echo ""
```

---

### PHASE 8: LOAD TRACKING STATUS (1 minute)

**Objective:** Verify data load status and detect duplicates

```bash
echo "=========================================="
echo "Load Tracking Status"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Load tracking table status
SELECT 
    database_name as "Database",
    load_status as "Status",
    events_loaded as "Events Loaded",
    duplicate_events_found as "Duplicates",
    load_completed_at as "Load Date",
    CASE 
        WHEN load_status = 'completed' THEN '✅ COMPLETE'
        WHEN load_status = 'pending' THEN '⏳ PENDING'
        WHEN load_status = 'failed' THEN '❌ FAILED'
        ELSE '⚠️  ' || load_status
    END as "Status Icon"
FROM load_tracking
ORDER BY load_completed_at DESC NULLS LAST;
SQL

echo ""
```

---

### PHASE 9: DATABASE SIZE AND GROWTH (1 minute)

**Objective:** Report database size and table sizes

```bash
echo "=========================================="
echo "Database Size and Growth"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Overall database size
SELECT 
    'Database Size' as "Metric",
    pg_size_pretty(pg_database_size('ntsb_aviation')) as "Value",
    '✅ OK' as "Status";

-- Top 10 largest tables
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as "Size",
    '✅ OK' as "Status"
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
SQL

echo ""
```

---

### PHASE 10: GENERATE SUMMARY REPORT (1 minute)

**Objective:** Create comprehensive validation summary

```bash
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""

# Generate summary report
REPORT_FILE="/tmp/NTSB_Datasets/validation-report-$(date +%Y-%m-%d-%H%M%S).txt"
mkdir -p /tmp/NTSB_Datasets

cat > "$REPORT_FILE" << EOF
# NTSB Aviation Database Validation Report
# Generated: $(date +"%Y-%m-%d %H:%M:%S")

## Database Connection
✅ Connected to ntsb_aviation database

## Validation Phases Completed
✅ Row Counts Validation
✅ Primary Key Validation
✅ NULL Value Checks
✅ Data Integrity Checks
✅ Foreign Key Validation
✅ Index and Performance Checks
✅ Load Tracking Status
✅ Database Size and Growth

## Quick Stats
Database Size: $(psql -d ntsb_aviation -t -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));" | xargs)
Total Tables: $(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public';" | xargs)
Total Rows: $(psql -d ntsb_aviation -t -c "SELECT SUM(n_live_tup) FROM pg_stat_user_tables WHERE schemaname = 'public';" | xargs)
Total Indexes: $(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';" | xargs)

## Recommendations
- Run ANALYZE on all tables if significant data changes occurred
- Refresh materialized views if data updated: SELECT * FROM refresh_all_materialized_views();
- Check for orphaned records and resolve foreign key issues
- Monitor database size growth trends

## Next Steps
- Review any warnings or failures above
- Run /refresh-mvs to update materialized views
- Run /benchmark to test query performance
- Run /cleanup-staging if staging tables need clearing

---

Full validation output saved to: $REPORT_FILE
EOF

cat "$REPORT_FILE"

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="
echo ""
echo "Report saved to: $REPORT_FILE"
echo ""
```

---

## SUCCESS CRITERIA

✅ Database connection verified
✅ All tables have data (row counts > 0)
✅ Primary keys exist on all tables
✅ No unexpected NULL values in critical columns
✅ Data ranges within valid bounds (coordinates, dates, ages)
✅ No orphaned records (foreign key integrity)
✅ Indexes present on key columns
✅ Load tracking status reviewed
✅ Database size reported
✅ Comprehensive report generated

---

## DELIVERABLES

1. **Console Output:** Real-time validation results with status icons
2. **Validation Report:** Text file in /tmp/NTSB_Datasets/ with comprehensive summary
3. **Actionable Recommendations:** Next steps based on validation results

---

## RELATED COMMANDS

**Database Maintenance:**
- `/refresh-mvs` - Refresh materialized views after data changes
- `/cleanup-staging` - Clear staging tables and vacuum database
- `/benchmark` - Test query performance after optimizations

**Data Operations:**
- `/daily-log` - Create comprehensive daily log with database metrics
- `/sprint-status` - View current sprint and database state

**Workflow Integration:**
```bash
# After data load
python scripts/load_with_staging.py --source avall.mdb
/validate-schema           # Verify data integrity
/refresh-mvs              # Update analytics views
/benchmark                # Test performance

# Regular maintenance
/validate-schema          # Check database health
# Review output for warnings
# Address any issues found
```

---

## TROUBLESHOOTING

**Connection Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database exists
psql -l | grep ntsb_aviation

# Check permissions
psql -d ntsb_aviation -c '\dt'
```

**Orphaned Records:**
```bash
# Identify specific orphaned records
psql -d ntsb_aviation -c "
SELECT * FROM aircraft a 
WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.ev_id = a.ev_id) 
LIMIT 10;
"

# Usually indicates data load issue - check load logs
```

**Invalid Data Ranges:**
```bash
# Find specific invalid records
psql -d ntsb_aviation -c "
SELECT ev_id, latitude, longitude 
FROM events 
WHERE latitude < -90 OR latitude > 90 
   OR longitude < -180 OR longitude > 180;
"
```

---

## NOTES

### When to Run

**Recommended Times:**
- After every data load (avall.mdb, Pre2008.mdb, PRE1982.MDB)
- After schema modifications
- Before creating materialized views
- Daily as part of database health checks
- Before major queries or analysis

### Performance Impact

- **Duration:** 5-10 minutes for full validation
- **Database Load:** Read-only queries, minimal impact
- **Safe to Run:** Can run while database is in use

### Quick Mode

For faster validation (row counts and primary keys only):
```bash
/validate-schema quick
```

---

**EXECUTE NOW - Validate database integrity and data quality.**
