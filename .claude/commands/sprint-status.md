# Sprint Status - Project Overview

Display current sprint status, database metrics, and pending tasks for the NTSB Aviation Database project.

---

## OBJECTIVE

Show comprehensive project status:
- Current sprint and progress
- Database state and row counts
- Recent changes and commits
- Pending tasks
- Quick reference commands

**Time Estimate:** <1 minute
**Prerequisites:** None (reads from CLAUDE.local.md and database)

---

## USAGE

```bash
/sprint-status              # Full status overview
/sprint-status quick        # Sprint info only
/sprint-status db           # Database metrics only
```

---

## EXECUTION

```bash
echo "=========================================="
echo "NTSB Aviation Database - Sprint Status"
echo "=========================================="
echo ""

# Read current sprint from CLAUDE.local.md
if [ -f CLAUDE.local.md ]; then
    SPRINT=$(grep -m 1 "Sprint:" CLAUDE.local.md | sed 's/.*Sprint: //' | sed 's/ .*//')
    STATUS=$(grep -m 1 "Status:" CLAUDE.local.md | sed 's/.*Status: //')
    
    echo "Current Sprint: $SPRINT"
    echo "Status: $STATUS"
    echo ""
fi

# Database metrics
if command -v psql &> /dev/null && psql -d ntsb_aviation -c "SELECT 1;" > /dev/null 2>&1; then
    echo "Database Metrics:"
    psql -d ntsb_aviation -t -c "SELECT '  Size: ' || pg_size_pretty(pg_database_size('ntsb_aviation'));"
    psql -d ntsb_aviation -t -c "SELECT '  Total Rows: ' || SUM(n_live_tup)::text FROM pg_stat_user_tables WHERE schemaname = 'public';"
    echo ""
    
    echo "Load Tracking:"
    psql -d ntsb_aviation -c "SELECT database_name, load_status, events_loaded FROM load_tracking ORDER BY load_completed_at DESC NULLS LAST;"
    echo ""
fi

# Recent commits
if [ -d .git ]; then
    echo "Recent Commits (last 5):"
    git log --oneline -5
    echo ""
fi

# Pending tasks from CLAUDE.local.md
if [ -f CLAUDE.local.md ]; then
    echo "Pending Tasks:"
    grep -A 10 "Pending" CLAUDE.local.md | head -15 || echo "  No pending tasks found in CLAUDE.local.md"
    echo ""
fi

echo "Quick Reference Commands:"
echo "  /validate-schema   - Validate database integrity"
echo "  /benchmark         - Test query performance"
echo "  /refresh-mvs       - Refresh materialized views"
echo "  /cleanup-staging   - Clear staging and vacuum"
echo "  /daily-log         - Create daily consolidation"
echo ""
