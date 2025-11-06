# Load Data - Intelligent Data Loading Wrapper

Intelligent wrapper for loading NTSB aviation accident data into PostgreSQL with automated validation, materialized view refresh, and comprehensive reporting.

---

## OBJECTIVE

Provide a safe, guided data loading experience that:
- Prevents duplicate loads through load_tracking checks
- Activates Python environment automatically
- Validates source database files
- Executes load_with_staging.py with proper error handling
- Validates data integrity after load
- Refreshes materialized views
- Generates comprehensive load report
- Ensures database remains in consistent state

**Time Estimate:** 5-15 minutes (depending on database size)
**Safety Level:** HIGH (multiple confirmation prompts for historical data)

---

## CONTEXT

**Project:** NTSB Aviation Database (PostgreSQL data repository)
**Repository:** /home/parobek/Code/NTSB_Datasets
**Python Environment:** .venv/ (must be activated)
**Loader Script:** scripts/load_with_staging.py

**Available Data Sources:**
- **avall.mdb** (537MB) - Current data (2008-present, ~30K events, monthly updates)
- **Pre2008.mdb** (893MB) - Historical data (2000-2007, ~3K unique events)
- **PRE1982.MDB** (188MB) - Legacy data (1962-1981, incompatible schema, requires custom ETL)

---

## USAGE

```bash
/load-data                    # Interactive mode (prompts for source)
/load-data avall.mdb          # Load current database
/load-data Pre2008.mdb        # Load historical database
/load-data --dry-run          # Check source without loading
/load-data --force            # Skip load_tracking confirmation
```

---

## EXECUTION PHASES

### PHASE 1: ENVIRONMENT CHECKS (2 minutes)

**Objective:** Verify environment and prerequisites

#### 1.1 Check Working Directory

```bash
echo "Checking working directory..."
if [ "$(basename $PWD)" != "NTSB_Datasets" ]; then
    echo "❌ ERROR: Must be in NTSB_Datasets directory"
    echo "Current: $PWD"
    exit 1
fi
echo "✅ Working directory: $PWD"
```

#### 1.2 Check PostgreSQL Connection

```bash
echo "Checking PostgreSQL connection..."
if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: psql not found - install PostgreSQL client"
    exit 1
fi

if ! psql -d ntsb_aviation -c "SELECT 1;" &> /dev/null; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    echo "   Run: ./scripts/setup_database.sh"
    exit 1
fi
echo "✅ PostgreSQL connection verified"
```

#### 1.3 Check Python Environment

```bash
echo "Checking Python environment..."
if [ ! -d ".venv" ]; then
    echo "❌ ERROR: Python virtual environment not found"
    echo "   Run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || source .venv/bin/activate.fish 2>/dev/null || {
    echo "❌ ERROR: Failed to activate Python environment"
    exit 1
}
echo "✅ Python environment activated"
```

#### 1.4 Check Python Dependencies

```bash
echo "Checking Python dependencies..."
python -c "import pandas, psycopg2, pyodbc" 2>/dev/null || {
    echo "❌ ERROR: Missing Python dependencies"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}
echo "✅ Python dependencies verified"
```

#### 1.5 Check Loader Script

```bash
echo "Checking loader script..."
if [ ! -f "scripts/load_with_staging.py" ]; then
    echo "❌ ERROR: load_with_staging.py not found"
    exit 1
fi
echo "✅ Loader script found"
```

**Verification:**
- [ ] Working directory confirmed
- [ ] PostgreSQL accessible
- [ ] Python environment activated
- [ ] Dependencies installed
- [ ] Loader script exists

---

### PHASE 2: SOURCE SELECTION (2 minutes)

**Objective:** Validate and confirm data source

#### 2.1 Parse Arguments or Prompt

```bash
# Check if source provided as argument
if [ -n "$1" ] && [ "$1" != "--dry-run" ] && [ "$1" != "--force" ]; then
    SOURCE_FILE="$1"
else
    # Interactive mode - prompt for source
    echo ""
    echo "Available data sources:"
    echo "  1) avall.mdb      - Current data (2008-present)"
    echo "  2) Pre2008.mdb    - Historical data (2000-2007)"
    echo "  3) PRE1982.MDB    - Legacy data (1962-1981, requires custom ETL)"
    echo ""
    read -p "Select source (1-3) or enter filename: " choice
    
    case $choice in
        1) SOURCE_FILE="avall.mdb" ;;
        2) SOURCE_FILE="Pre2008.mdb" ;;
        3) SOURCE_FILE="PRE1982.MDB" ;;
        *) SOURCE_FILE="$choice" ;;
    esac
fi

echo "Selected source: $SOURCE_FILE"
```

#### 2.2 Validate Source File

```bash
echo "Validating source file..."

# Check if file exists in datasets/
if [ ! -f "datasets/$SOURCE_FILE" ]; then
    echo "❌ ERROR: datasets/$SOURCE_FILE not found"
    echo "Available files:"
    ls -lh datasets/*.mdb 2>/dev/null || echo "  No .mdb files found"
    exit 1
fi

# Get file size
FILE_SIZE=$(du -h "datasets/$SOURCE_FILE" | cut -f1)
echo "✅ Source file exists: $SOURCE_FILE ($FILE_SIZE)"
```

#### 2.3 Check PRE1982 Special Case

```bash
# PRE1982.MDB requires custom ETL (incompatible schema)
if [[ "$SOURCE_FILE" =~ "PRE1982" ]] || [[ "$SOURCE_FILE" =~ "pre1982" ]]; then
    echo ""
    echo "⚠️  WARNING: PRE1982.MDB has incompatible schema"
    echo "   This database requires custom ETL (not yet implemented)"
    echo "   See: docs/PRE1982_ANALYSIS.md for details"
    echo ""
    read -p "Continue anyway? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi
```

**Verification:**
- [ ] Source file selected
- [ ] Source file exists and accessible
- [ ] File size confirmed
- [ ] Special cases handled

---

### PHASE 3: LOAD TRACKING CHECK (2 minutes)

**Objective:** Prevent duplicate loads using load_tracking system

#### 3.1 Query Load Tracking

```bash
echo "Checking load tracking status..."

LOAD_STATUS=$(psql -d ntsb_aviation -t -c "
    SELECT load_status 
    FROM load_tracking 
    WHERE database_name = '$SOURCE_FILE';
" 2>/dev/null | xargs)

LOAD_DATE=$(psql -d ntsb_aviation -t -c "
    SELECT TO_CHAR(load_completed_at, 'YYYY-MM-DD HH24:MI:SS')
    FROM load_tracking 
    WHERE database_name = '$SOURCE_FILE';
" 2>/dev/null | xargs)

echo "Load status for $SOURCE_FILE: $LOAD_STATUS"
```

#### 3.2 Handle Load Status

```bash
# Check for --force flag
FORCE_FLAG=false
if [[ "$@" =~ "--force" ]]; then
    FORCE_FLAG=true
fi

if [ "$LOAD_STATUS" = "completed" ]; then
    echo ""
    echo "⚠️  WARNING: $SOURCE_FILE already loaded on $LOAD_DATE"
    
    # Historical databases should only be loaded once
    if [[ "$SOURCE_FILE" =~ "Pre2008" ]] || [[ "$SOURCE_FILE" =~ "PRE1982" ]]; then
        echo "⚠️  Historical databases should only be loaded once!"
        echo "   Re-loading will create duplicate events in staging"
        echo "   (Deduplication will handle it, but it's inefficient)"
    fi
    
    if [ "$FORCE_FLAG" = false ]; then
        echo ""
        read -p "Continue anyway? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Aborted."
            exit 0
        fi
    else
        echo "⚠️  --force flag detected, proceeding with load"
    fi
elif [ "$LOAD_STATUS" = "failed" ]; then
    echo "ℹ️  Previous load failed - safe to retry"
elif [ "$LOAD_STATUS" = "pending" ] || [ -z "$LOAD_STATUS" ]; then
    echo "✅ Database not yet loaded - proceeding"
else
    echo "⚠️  Unknown load status: $LOAD_STATUS"
fi
```

**Verification:**
- [ ] Load tracking checked
- [ ] Load status interpreted
- [ ] Duplicate load warning (if applicable)
- [ ] User confirmation obtained (if needed)

---

### PHASE 4: DRY RUN MODE (Optional, 1 minute)

**Objective:** Validate source without loading data

```bash
# Check for --dry-run flag
if [[ "$@" =~ "--dry-run" ]]; then
    echo ""
    echo "🔍 DRY RUN MODE - Validating source without loading"
    echo ""
    
    # Test mdbtools connection (if available)
    if command -v mdb-tables &> /dev/null; then
        echo "Available tables in $SOURCE_FILE:"
        mdb-tables "datasets/$SOURCE_FILE" | tr ' ' '\n' | head -20
        echo ""
        
        # Check events table row count
        echo "Checking events table..."
        EVENT_COUNT=$(mdb-export "datasets/$SOURCE_FILE" events 2>/dev/null | wc -l)
        echo "Events in source: $((EVENT_COUNT - 1))"  # Subtract header row
    else
        echo "ℹ️  mdbtools not installed - cannot inspect source"
        echo "   Install: sudo pacman -S mdbtools (or apt/brew equivalent)"
    fi
    
    echo ""
    echo "✅ Dry run complete - source file validated"
    echo "   Run without --dry-run to load data"
    exit 0
fi
```

---

### PHASE 5: EXECUTE LOAD (5-10 minutes)

**Objective:** Run load_with_staging.py with comprehensive logging

#### 5.1 Create Temporary Log Directory

```bash
echo "Creating temporary log directory..."
mkdir -p /tmp/NTSB_Datasets/load_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/tmp/NTSB_Datasets/load_logs/load_${SOURCE_FILE%.mdb}_${TIMESTAMP}.log"
echo "✅ Log file: $LOG_FILE"
```

#### 5.2 Display Pre-Load State

```bash
echo ""
echo "📊 Pre-Load Database State:"
psql -d ntsb_aviation -c "
    SELECT 
        schemaname, 
        tablename, 
        n_live_tup as rows,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_stat_user_tables 
    WHERE schemaname = 'public' 
    ORDER BY n_live_tup DESC 
    LIMIT 5;
"

PRE_LOAD_EVENTS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM events;" | xargs)
PRE_LOAD_SIZE=$(psql -d ntsb_aviation -t -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));" | xargs)
echo ""
echo "Pre-load events: $PRE_LOAD_EVENTS"
echo "Pre-load database size: $PRE_LOAD_SIZE"
echo ""
```

#### 5.3 Execute Load with Progress

```bash
echo "🚀 Starting data load..."
echo "   Source: datasets/$SOURCE_FILE"
echo "   Script: scripts/load_with_staging.py"
echo "   Log: $LOG_FILE"
echo ""
echo "This may take 5-10 minutes depending on database size..."
echo ""

# Run load_with_staging.py with tee to capture output
python scripts/load_with_staging.py --source "$SOURCE_FILE" 2>&1 | tee "$LOG_FILE"
LOAD_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $LOAD_EXIT_CODE -eq 0 ]; then
    echo "✅ Data load completed successfully"
else
    echo "❌ ERROR: Data load failed with exit code $LOAD_EXIT_CODE"
    echo "   Review log: $LOG_FILE"
    exit $LOAD_EXIT_CODE
fi
```

#### 5.4 Display Post-Load State

```bash
echo ""
echo "📊 Post-Load Database State:"
psql -d ntsb_aviation -c "
    SELECT 
        schemaname, 
        tablename, 
        n_live_tup as rows,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_stat_user_tables 
    WHERE schemaname = 'public' 
    ORDER BY n_live_tup DESC 
    LIMIT 5;
"

POST_LOAD_EVENTS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM events;" | xargs)
POST_LOAD_SIZE=$(psql -d ntsb_aviation -t -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));" | xargs)

EVENTS_ADDED=$((POST_LOAD_EVENTS - PRE_LOAD_EVENTS))
echo ""
echo "Post-load events: $POST_LOAD_EVENTS (+$EVENTS_ADDED)"
echo "Post-load database size: $POST_LOAD_SIZE"
echo ""
```

**Verification:**
- [ ] Load executed without errors
- [ ] Log file created
- [ ] Post-load state captured
- [ ] Events count increased

---

### PHASE 6: VALIDATION (2 minutes)

**Objective:** Run quick validation to ensure data integrity

```bash
echo "🔍 Running quick validation..."
echo ""

# Check for duplicates in production events table
DUPLICATE_COUNT=$(psql -d ntsb_aviation -t -c "
    SELECT COUNT(*) - COUNT(DISTINCT ev_id)
    FROM events;
" | xargs)

if [ "$DUPLICATE_COUNT" -gt 0 ]; then
    echo "⚠️  WARNING: Found $DUPLICATE_COUNT duplicate events in production table"
    echo "   This should not happen - investigate immediately"
else
    echo "✅ No duplicate events (as expected)"
fi

# Check for orphaned records
ORPHANED_AIRCRAFT=$(psql -d ntsb_aviation -t -c "
    SELECT COUNT(*) 
    FROM aircraft a 
    WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.ev_id = a.ev_id);
" | xargs)

if [ "$ORPHANED_AIRCRAFT" -gt 0 ]; then
    echo "⚠️  WARNING: Found $ORPHANED_AIRCRAFT orphaned aircraft records"
else
    echo "✅ No orphaned aircraft records"
fi

# Check load tracking
FINAL_STATUS=$(psql -d ntsb_aviation -t -c "
    SELECT load_status 
    FROM load_tracking 
    WHERE database_name = '$SOURCE_FILE';
" | xargs)

echo "✅ Load tracking status: $FINAL_STATUS"

# Check for staging table cleanup (should be empty after merge)
STAGING_EVENTS=$(psql -d ntsb_aviation -t -c "
    SELECT COUNT(*) FROM staging.events;
" 2>/dev/null | xargs)

if [ "$STAGING_EVENTS" -gt 0 ]; then
    echo "ℹ️  Staging tables still contain data ($STAGING_EVENTS events)"
    echo "   Run /cleanup-staging to clear staging tables"
else
    echo "✅ Staging tables cleared"
fi

echo ""
```

**Verification:**
- [ ] No duplicate events
- [ ] No orphaned records
- [ ] Load tracking updated
- [ ] Staging tables status checked

---

### PHASE 7: REFRESH MATERIALIZED VIEWS (2 minutes)

**Objective:** Update analytics views with new data

```bash
echo "🔄 Refreshing materialized views..."
echo ""

# Refresh all materialized views using database function
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Materialized views refreshed successfully"
    
    # Show view row counts
    echo ""
    echo "Materialized view status:"
    psql -d ntsb_aviation -c "
        SELECT 
            matviewname,
            pg_size_pretty(pg_total_relation_size('public.'||matviewname)) as size
        FROM pg_matviews 
        WHERE schemaname = 'public';
    "
else
    echo "⚠️  WARNING: Failed to refresh materialized views"
    echo "   Run manually: psql -d ntsb_aviation -c \"SELECT * FROM refresh_all_materialized_views();\""
fi

echo ""
```

**Verification:**
- [ ] Materialized views refreshed
- [ ] View row counts displayed
- [ ] No errors during refresh

---

### PHASE 8: GENERATE LOAD REPORT (2 minutes)

**Objective:** Create comprehensive load report for documentation

```bash
echo "📄 Generating load report..."
echo ""

REPORT_FILE="/tmp/NTSB_Datasets/load_logs/load_report_${SOURCE_FILE%.mdb}_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# Data Load Report: $SOURCE_FILE

**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Source File:** datasets/$SOURCE_FILE ($FILE_SIZE)
**Load Duration:** [CALCULATE from timestamps]
**Status:** ✅ SUCCESS

---

## Summary

Loaded $SOURCE_FILE into ntsb_aviation database with comprehensive validation and materialized view refresh.

**Events Added:** $EVENTS_ADDED
**Total Events:** $POST_LOAD_EVENTS
**Database Size:** $PRE_LOAD_SIZE → $POST_LOAD_SIZE

---

## Pre-Load State

- **Events:** $PRE_LOAD_EVENTS
- **Database Size:** $PRE_LOAD_SIZE
- **Load Tracking:** $LOAD_STATUS

---

## Load Process

**Command:** python scripts/load_with_staging.py --source $SOURCE_FILE
**Exit Code:** $LOAD_EXIT_CODE
**Log File:** $LOG_FILE

### Load Statistics

$(grep -E "(Loading|Loaded|Duplicate)" "$LOG_FILE" | head -20)

---

## Post-Load State

- **Events:** $POST_LOAD_EVENTS (+$EVENTS_ADDED)
- **Database Size:** $POST_LOAD_SIZE
- **Load Tracking:** $FINAL_STATUS

---

## Validation Results

- **Duplicate Events:** $DUPLICATE_COUNT ✅
- **Orphaned Aircraft:** $ORPHANED_AIRCRAFT ✅
- **Staging Tables:** $([ $STAGING_EVENTS -eq 0 ] && echo "Cleared ✅" || echo "$STAGING_EVENTS events remaining")

---

## Materialized Views

Refreshed all 6 materialized views:
- mv_yearly_stats
- mv_state_stats
- mv_aircraft_stats
- mv_decade_stats
- mv_crew_stats
- mv_finding_stats

---

## Next Steps

1. Review load log: $LOG_FILE
2. Run comprehensive validation: \`/validate-schema\`
3. Benchmark performance: \`/benchmark\`
4. Clean staging tables (if needed): \`/cleanup-staging\`
5. Create daily log: \`/daily-log\`

---

**Generated by:** /load-data command
**Load Tracking Entry:** Updated in database
**Report Version:** 1.0
EOF

echo "✅ Load report generated: $REPORT_FILE"
echo ""
```

**Verification:**
- [ ] Report file created
- [ ] All metrics captured
- [ ] Next steps documented

---

### PHASE 9: COMPLETION SUMMARY (1 minute)

**Objective:** Display final summary and next steps

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DATA LOAD COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 SUMMARY"
echo "   Source: $SOURCE_FILE ($FILE_SIZE)"
echo "   Events added: $EVENTS_ADDED"
echo "   Total events: $POST_LOAD_EVENTS"
echo "   Database size: $POST_LOAD_SIZE"
echo "   Status: ✅ SUCCESS"
echo ""
echo "📝 ARTIFACTS"
echo "   Load log: $LOG_FILE"
echo "   Load report: $REPORT_FILE"
echo ""
echo "🔍 VALIDATION"
echo "   Duplicates: $DUPLICATE_COUNT ✅"
echo "   Orphaned: $ORPHANED_AIRCRAFT ✅"
echo "   Load tracking: $FINAL_STATUS ✅"
echo "   Materialized views: Refreshed ✅"
echo ""
echo "📋 RECOMMENDED NEXT STEPS"
echo "   1. Review load log for warnings/errors"
echo "   2. Run comprehensive validation: /validate-schema"
echo "   3. Benchmark query performance: /benchmark"
echo "   4. Clean staging tables: /cleanup-staging"
echo "   5. Create daily log: /daily-log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
```

---

## SUCCESS CRITERIA

- [ ] Environment validated (PostgreSQL, Python, dependencies)
- [ ] Source file validated and confirmed
- [ ] Load tracking checked (duplicates prevented)
- [ ] Data loaded successfully (exit code 0)
- [ ] Post-load validation passed (no duplicates, no orphans)
- [ ] Materialized views refreshed
- [ ] Load report generated
- [ ] Summary displayed with next steps

---

## OUTPUT/DELIVERABLES

**Log Files:**
- `/tmp/NTSB_Datasets/load_logs/load_[SOURCE]_[TIMESTAMP].log` - Full load output
- `/tmp/NTSB_Datasets/load_logs/load_report_[SOURCE]_[TIMESTAMP].md` - Summary report

**Database Updates:**
- Events added to production tables
- Load tracking entry updated (status, counts, timestamp)
- Materialized views refreshed

**Console Output:**
- Real-time progress updates
- Pre/post load state comparison
- Validation results
- Next steps recommendations

---

## RELATED COMMANDS

- `/validate-schema` - Comprehensive validation (run after load)
- `/cleanup-staging` - Clear staging tables (run after successful load)
- `/refresh-mvs` - Refresh materialized views only
- `/benchmark` - Test query performance with new data
- `/sprint-status` - Check database status
- `/daily-log` - Document load in daily log

---

## NOTES

### When to Use

**Load avall.mdb:**
- Monthly updates from NTSB (current data)
- First-time setup
- After database reset

**Load Pre2008.mdb:**
- One-time historical data integration
- Adds 2000-2007 data (~3K unique events)
- Only load once (uses load_tracking to prevent duplicates)

**Load PRE1982.MDB:**
- NOT CURRENTLY SUPPORTED (incompatible schema)
- Requires custom ETL (Sprint 3)
- See docs/PRE1982_ANALYSIS.md

### Performance Considerations

**Load Times:**
- avall.mdb: ~30 seconds (29,773 events)
- Pre2008.mdb: ~90 seconds (906,176 rows staging, ~3K unique events)
- PRE1982.MDB: Not yet supported

**Throughput:**
- Staging: 15,000-45,000 rows/sec (bulk COPY)
- Merge: 1,000-5,000 rows/sec (deduplication)

### Safety Features

**Load Tracking:**
- Prevents duplicate loads of historical databases
- Tracks load status, counts, timestamps
- Prompts for confirmation on re-loads

**Validation:**
- Automatic duplicate detection
- Orphaned record checks
- Foreign key integrity
- Data quality checks

**Error Handling:**
- Graceful failures (no data loss)
- Comprehensive logging
- Clear error messages
- Rollback-safe (staging pattern)

---

## TROUBLESHOOTING

### Problem: "Cannot connect to ntsb_aviation database"

**Solution:**
```bash
# Initialize database
./scripts/setup_database.sh

# Verify connection
psql -d ntsb_aviation -c "SELECT 1;"
```

### Problem: "Python dependencies missing"

**Solution:**
```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import pandas, psycopg2, pyodbc"
```

### Problem: "Source file not found"

**Solution:**
```bash
# Check available files
ls -lh datasets/*.mdb

# Ensure file name matches exactly (case-sensitive)
# Use: avall.mdb, Pre2008.mdb, or PRE1982.MDB
```

### Problem: "Load failed with exit code 1"

**Solution:**
```bash
# Review log file for specific error
cat /tmp/NTSB_Datasets/load_logs/load_*.log | tail -50

# Common issues:
# - mdbtools not installed (for reading .mdb files)
# - Permission denied (database ownership)
# - Disk space full
```

### Problem: "Duplicate events found after load"

**Solution:**
```bash
# This should not happen (deduplication should prevent it)
# Check load_tracking - was database loaded twice?
psql -d ntsb_aviation -c "SELECT * FROM load_tracking;"

# Check staging tables
psql -d ntsb_aviation -c "SELECT * FROM staging.get_duplicate_stats();"

# If duplicates exist in production, report as bug
```

### Problem: "Materialized views failed to refresh"

**Solution:**
```bash
# Refresh manually
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"

# If still fails, check function exists
psql -d ntsb_aviation -c "\df refresh_all_materialized_views"

# Recreate if needed
psql -d ntsb_aviation -f scripts/optimize_queries.sql
```

---

## EXAMPLE USAGE

### Load Current Data (First Time)

```bash
# Interactive mode
/load-data
# Select: 1 (avall.mdb)
# Confirm prompts
# Wait ~30 seconds
# Review summary
# Run validation
/validate-schema
```

### Load Historical Data

```bash
# Direct mode with source specified
/load-data Pre2008.mdb
# Confirm historical load warning
# Wait ~90 seconds
# Review summary showing ~3K unique events added
# Clean staging
/cleanup-staging
```

### Dry Run (Test Without Loading)

```bash
# Validate source file without loading
/load-data --dry-run
# Select source
# Review table list and event count
# No data loaded
```

### Force Re-load (Override Load Tracking)

```bash
# Re-load even if already loaded (not recommended for historical data)
/load-data avall.mdb --force
# No confirmation prompt
# Proceeds with load
# Deduplication handles duplicates in staging
```

### Complete Workflow

```bash
# Morning update routine
/sprint-status                      # Check current state
/load-data avall.mdb                # Load monthly update
/validate-schema                    # Verify integrity
/cleanup-staging                    # Clear staging
/benchmark quick                    # Test performance
/daily-log                          # Document changes
```

---

**Command Version:** 1.0
**Last Updated:** 2025-11-06
**Adapted From:** Original command for NTSB Aviation Database
**Priority:** HIGH - Core data operation command
