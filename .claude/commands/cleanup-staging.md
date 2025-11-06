# Cleanup Staging - Database Maintenance

Clear staging tables, vacuum database, and perform comprehensive maintenance operations for the NTSB Aviation Database.

---

## OBJECTIVE

Perform database maintenance operations:
- Clear all staging schema tables
- Vacuum and analyze tables
- Rebuild materialized views
- Remove temporary data
- Report space saved

**Time Estimate:** ~5-10 minutes
**Prerequisites:** PostgreSQL connection to ntsb_aviation database

---

## USAGE

```bash
/cleanup-staging              # Full cleanup (staging + vacuum + analyze)
/cleanup-staging staging      # Staging tables only
/cleanup-staging vacuum       # Vacuum only (no staging clear)
/cleanup-staging full         # VACUUM FULL (aggressive, locks tables)
```

---

## EXECUTION

```bash
echo "=========================================="
echo "Database Cleanup - NTSB Aviation"
echo "=========================================="
echo ""

# Check database connection
if ! psql -d ntsb_aviation -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    exit 1
fi

MODE="${1:-full}"
echo "Cleanup mode: $MODE"
echo ""

# Phase 1: Clear Staging Tables
if [ "$MODE" = "full" ] || [ "$MODE" = "staging" ]; then
    echo "=========================================="
    echo "Clearing Staging Tables"
    echo "=========================================="
    echo ""
    
    # Get row counts before clearing
    echo "Before clearing:"
    psql -d ntsb_aviation -c "SELECT * FROM staging.get_row_counts();"
    echo ""
    
    # Clear all staging tables
    psql -d ntsb_aviation -c "SELECT staging.clear_all_staging();"
    echo ""
    
    echo "After clearing:"
    psql -d ntsb_aviation -c "SELECT * FROM staging.get_row_counts();"
    echo ""
    
    echo "✅ Staging tables cleared"
    echo ""
fi

# Phase 2: Vacuum Database
if [ "$MODE" = "full" ] || [ "$MODE" = "vacuum" ]; then
    echo "=========================================="
    echo "Vacuuming Database"
    echo "=========================================="
    echo ""
    
    if [ "$MODE" = "full" ] && [ "$1" = "full" ]; then
        echo "⚠️  Running VACUUM FULL (this will lock tables)"
        read -p "Continue? (yes/no): " answer
        if [ "$answer" = "yes" ]; then
            psql -d ntsb_aviation -c "VACUUM FULL;"
        else
            echo "Aborted VACUUM FULL"
        fi
    else
        echo "Running VACUUM ANALYZE on all tables..."
        psql -d ntsb_aviation -c "VACUUM ANALYZE;"
    fi
    
    echo ""
    echo "✅ Vacuum complete"
    echo ""
fi

# Phase 3: Report Space Saved
echo "=========================================="
echo "Database Statistics"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
SELECT 
    'Database Size' as "Metric",
    pg_size_pretty(pg_database_size('ntsb_aviation')) as "Value";

SELECT 
    schemaname,
    COUNT(*) as "Tables",
    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))) as "Total Size"
FROM pg_tables
WHERE schemaname IN ('public', 'staging')
GROUP BY schemaname;
SQL

echo ""
echo "✅ Cleanup complete"
