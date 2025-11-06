# Refresh MVs - Materialized View Management

Refresh all materialized views and report on row counts and refresh times for the NTSB Aviation Database.

---

## OBJECTIVE

Refresh database materialized views:
- Refresh all 6 materialized views concurrently
- Report refresh times
- Show updated row counts
- Run ANALYZE on refreshed views

**Time Estimate:** ~1-2 minutes
**Prerequisites:** PostgreSQL connection to ntsb_aviation database

---

## USAGE

```bash
/refresh-mvs                # Refresh all materialized views
/refresh-mvs quick          # Refresh without ANALYZE
/refresh-mvs <view_name>    # Refresh specific view only
```

---

## EXECUTION

```bash
echo "=========================================="
echo "Refresh Materialized Views - NTSB Aviation"
echo "=========================================="
echo ""

# Check database connection
if ! psql -d ntsb_aviation -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    exit 1
fi

VIEW_NAME="$1"

# Get materialized views before refresh
echo "Materialized Views Status (Before):"
psql -d ntsb_aviation << 'SQL'
SELECT 
    matviewname as "View Name",
    pg_size_pretty(pg_total_relation_size('public.' || matviewname)) as "Size"
FROM pg_matviews
WHERE schemaname = 'public'
ORDER BY matviewname;
SQL
echo ""

# Refresh views
if [ -z "$VIEW_NAME" ] || [ "$VIEW_NAME" = "quick" ]; then
    echo "Refreshing all materialized views..."
    START=$(date +%s)
    
    psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"
    
    END=$(date +%s)
    DURATION=$((END - START))
    echo ""
    echo "✅ Refresh complete in ${DURATION}s"
else
    echo "Refreshing specific view: $VIEW_NAME"
    psql -d ntsb_aviation -c "REFRESH MATERIALIZED VIEW CONCURRENTLY $VIEW_NAME;"
    echo "✅ View refreshed: $VIEW_NAME"
fi

echo ""

# Show updated statistics
echo "Materialized Views Status (After):"
psql -d ntsb_aviation << 'SQL'
SELECT 
    matviewname as "View Name",
    pg_size_pretty(pg_total_relation_size('public.' || matviewname)) as "Size",
    '✅ Updated' as "Status"
FROM pg_matviews
WHERE schemaname = 'public'
ORDER BY matviewname;
SQL

echo ""

# Run ANALYZE unless quick mode
if [ "$VIEW_NAME" != "quick" ]; then
    echo "Running ANALYZE on materialized views..."
    psql -d ntsb_aviation -c "ANALYZE mv_yearly_stats, mv_state_stats, mv_aircraft_stats, mv_decade_stats, mv_crew_stats, mv_finding_stats;"
    echo "✅ ANALYZE complete"
    echo ""
fi

echo "Materialized views are up to date and ready for queries."
echo ""
