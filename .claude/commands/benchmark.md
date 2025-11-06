# Benchmark - Query Performance Testing

Execute comprehensive performance benchmarks and measure query latencies for the NTSB Aviation Database.

---

## OBJECTIVE

Measure database query performance across common analytical queries:
- Test p50, p95, p99 latencies for key queries
- Validate materialized view performance
- Benchmark join operations and aggregations
- Compare against performance targets (<100ms for 95% of queries)
- Generate performance report with recommendations

**Time Estimate:** ~5-10 minutes
**Prerequisites:** PostgreSQL connection to ntsb_aviation database

---

## USAGE

```bash
/benchmark                 # Full benchmark suite
/benchmark quick           # Essential queries only (5 tests)
/benchmark views           # Materialized views only
/benchmark compare         # Compare against baseline (if exists)
```

---

## EXECUTION PHASES

### PHASE 1: VERIFY DATABASE CONNECTION (1 minute)

**Objective:** Ensure database is ready for benchmarking

```bash
echo "=========================================="
echo "Performance Benchmark - NTSB Aviation"
echo "=========================================="
echo ""

# Check psql availability
if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: psql not installed"
    exit 1
fi

# Test database connection
if ! psql -d ntsb_aviation -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    exit 1
fi

echo "✅ Connected to ntsb_aviation database"
echo ""

# Check if pg_stat_statements extension is enabled
STATS_ENABLED=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'pg_stat_statements';" | xargs)

if [ "$STATS_ENABLED" -eq 0 ]; then
    echo "⚠️  WARNING: pg_stat_statements extension not enabled"
    echo "   Performance tracking will be limited"
    echo ""
fi
```

---

### PHASE 2: PREPARE BENCHMARK ENVIRONMENT (1 minute)

**Objective:** Clear query cache and prepare for testing

```bash
echo "=========================================="
echo "Preparing Benchmark Environment"
echo "=========================================="
echo ""

# Create benchmark output directory
BENCHMARK_DIR="/tmp/NTSB_Datasets/benchmarks"
mkdir -p "$BENCHMARK_DIR"

# Generate timestamp for this benchmark run
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
REPORT_FILE="$BENCHMARK_DIR/benchmark-report-$TIMESTAMP.txt"

echo "Benchmark run: $TIMESTAMP" > "$REPORT_FILE"
echo "Database: ntsb_aviation" >> "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Warm up the database (one simple query)
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;" > /dev/null 2>&1

echo "✅ Benchmark environment ready"
echo "✅ Output directory: $BENCHMARK_DIR"
echo ""
```

---

### PHASE 3: BASIC QUERY BENCHMARKS (2 minutes)

**Objective:** Test fundamental query patterns

```bash
echo "=========================================="
echo "Basic Query Benchmarks"
echo "=========================================="
echo ""

# Function to run query multiple times and measure performance
benchmark_query() {
    local query_name="$1"
    local query="$2"
    local iterations=10
    
    echo "Testing: $query_name"
    echo "  Iterations: $iterations"
    
    # Run query multiple times and collect timings
    local total_time=0
    local times=()
    
    for i in $(seq 1 $iterations); do
        start=$(date +%s%N)
        psql -d ntsb_aviation -c "$query" > /dev/null 2>&1
        end=$(date +%s%N)
        elapsed=$((($end - $start) / 1000000))  # Convert to milliseconds
        times+=($elapsed)
        total_time=$(($total_time + $elapsed))
    done
    
    # Calculate statistics
    avg=$(($total_time / $iterations))
    
    # Sort times for percentile calculation
    IFS=$'\n' sorted_times=($(sort -n <<<"${times[*]}"))
    unset IFS
    
    # Calculate percentiles
    p50_index=$((($iterations * 50) / 100))
    p95_index=$((($iterations * 95) / 100))
    p99_index=$((($iterations * 99) / 100))
    
    p50=${sorted_times[$p50_index]}
    p95=${sorted_times[$p95_index]}
    p99=${sorted_times[$p99_index]}
    
    # Determine status
    if [ $p95 -lt 100 ]; then
        status="✅ EXCELLENT"
    elif [ $p95 -lt 500 ]; then
        status="✅ GOOD"
    elif [ $p95 -lt 1000 ]; then
        status="⚠️  ACCEPTABLE"
    else
        status="❌ SLOW"
    fi
    
    echo "  Average: ${avg}ms"
    echo "  p50: ${p50}ms"
    echo "  p95: ${p95}ms"
    echo "  p99: ${p99}ms"
    echo "  Status: $status"
    echo ""
    
    # Log to report file
    echo "$query_name,$avg,$p50,$p95,$p99,$status" >> "$BENCHMARK_DIR/timings-$TIMESTAMP.csv"
}

# Create CSV header
echo "Query,Average (ms),p50 (ms),p95 (ms),p99 (ms),Status" > "$BENCHMARK_DIR/timings-$TIMESTAMP.csv"

# Test 1: Simple count query
benchmark_query "Simple Count (events)" \
    "SELECT COUNT(*) FROM events;"

# Test 2: Filtered count with date range
benchmark_query "Filtered Count (2020-2025)" \
    "SELECT COUNT(*) FROM events WHERE ev_year >= 2020 AND ev_year <= 2025;"

# Test 3: Aggregation by state
benchmark_query "State Aggregation" \
    "SELECT ev_state, COUNT(*) as count FROM events WHERE ev_state IS NOT NULL GROUP BY ev_state ORDER BY count DESC LIMIT 10;"

# Test 4: Join query (events + aircraft)
benchmark_query "Events-Aircraft Join" \
    "SELECT e.ev_id, e.ntsb_no, a.acft_make, a.acft_model FROM events e JOIN aircraft a ON e.ev_id = a.ev_id LIMIT 1000;"

# Test 5: Complex aggregation with multiple joins
benchmark_query "Complex Multi-Join" \
    "SELECT e.ev_year, COUNT(DISTINCT e.ev_id) as accidents, COUNT(a.\"Aircraft_Key\") as aircraft_count FROM events e LEFT JOIN aircraft a ON e.ev_id = a.ev_id WHERE e.ev_year >= 2010 GROUP BY e.ev_year ORDER BY e.ev_year DESC;"
```

---

### PHASE 4: MATERIALIZED VIEW BENCHMARKS (2 minutes)

**Objective:** Test materialized view query performance

```bash
echo "=========================================="
echo "Materialized View Benchmarks"
echo "=========================================="
echo ""

# Test materialized view queries
benchmark_query "MV: Yearly Stats" \
    "SELECT * FROM mv_yearly_stats ORDER BY ev_year DESC LIMIT 10;"

benchmark_query "MV: State Stats" \
    "SELECT * FROM mv_state_stats ORDER BY total_accidents DESC LIMIT 10;"

benchmark_query "MV: Aircraft Stats" \
    "SELECT * FROM mv_aircraft_stats ORDER BY total_accidents DESC LIMIT 20;"

benchmark_query "MV: Decade Stats" \
    "SELECT * FROM mv_decade_stats ORDER BY decade DESC;"

benchmark_query "MV: Crew Stats" \
    "SELECT * FROM mv_crew_stats ORDER BY total_crew DESC LIMIT 10;"
```

---

### PHASE 5: INDEX UTILIZATION CHECK (1 minute)

**Objective:** Verify indexes are being used effectively

```bash
echo "=========================================="
echo "Index Utilization Check"
echo "=========================================="
echo ""

psql -d ntsb_aviation << 'SQL'
-- Check index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as "Index Scans",
    idx_tup_read as "Tuples Read",
    idx_tup_fetch as "Tuples Fetched",
    CASE 
        WHEN idx_scan > 0 THEN '✅ USED'
        ELSE '⚠️  UNUSED'
    END as "Status"
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC
LIMIT 20;
SQL

echo ""
```

---

### PHASE 6: QUERY PLAN ANALYSIS (1 minute)

**Objective:** Analyze query execution plans for key queries

```bash
echo "=========================================="
echo "Query Plan Analysis"
echo "=========================================="
echo ""

# Analyze a complex query plan
echo "Analyzing complex query execution plan..."
psql -d ntsb_aviation << 'SQL'
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    e.ev_year,
    e.ev_state,
    COUNT(*) as accident_count,
    COUNT(DISTINCT a."Aircraft_Key") as aircraft_count
FROM events e
LEFT JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year >= 2015
GROUP BY e.ev_year, e.ev_state
ORDER BY e.ev_year DESC, accident_count DESC
LIMIT 50;
SQL

echo ""
```

---

### PHASE 7: GENERATE PERFORMANCE REPORT (2 minutes)

**Objective:** Create comprehensive performance summary

```bash
echo "=========================================="
echo "Generating Performance Report"
echo "=========================================="
echo ""

# Read CSV and generate markdown report
cat > "$REPORT_FILE" << 'EOF'
# NTSB Aviation Database - Performance Benchmark Report

**Generated:** $(date +"%Y-%m-%d %H:%M:%S")
**Database:** ntsb_aviation
**Benchmark Run:** $TIMESTAMP

---

## Executive Summary

EOF

# Calculate overall statistics from CSV
AVG_P50=$(awk -F',' 'NR>1 {sum+=$3; count++} END {print int(sum/count)}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv")
AVG_P95=$(awk -F',' 'NR>1 {sum+=$4; count++} END {print int(sum/count)}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv")
AVG_P99=$(awk -F',' 'NR>1 {sum+=$5; count++} END {print int(sum/count)}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv")

cat >> "$REPORT_FILE" << EOF

**Performance Metrics:**
- Average p50 Latency: ${AVG_P50}ms
- Average p95 Latency: ${AVG_P95}ms
- Average p99 Latency: ${AVG_P99}ms

**Target Compliance:**
- Target: <100ms for 95% of queries
- Status: $([ $AVG_P95 -lt 100 ] && echo "✅ MEETING TARGET" || echo "⚠️  NEEDS OPTIMIZATION")

---

## Detailed Query Performance

| Query | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Status |
|-------|----------|----------|----------|----------|--------|
EOF

# Convert CSV to markdown table
tail -n +2 "$BENCHMARK_DIR/timings-$TIMESTAMP.csv" | while IFS=',' read -r query avg p50 p95 p99 status; do
    echo "| $query | $avg | $p50 | $p95 | $p99 | $status |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << 'EOF'

---

## Performance Analysis

### Fastest Queries (p95 < 50ms)
EOF

# List fastest queries
awk -F',' 'NR>1 && $4 < 50 {print "- " $1 " (" $4 "ms)"}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

### Slow Queries (p95 > 500ms)
EOF

# List slow queries
SLOW_COUNT=$(awk -F',' 'NR>1 && $4 > 500 {count++} END {print count+0}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv")

if [ $SLOW_COUNT -eq 0 ]; then
    echo "✅ No slow queries detected" >> "$REPORT_FILE"
else
    awk -F',' 'NR>1 && $4 > 500 {print "- " $1 " (" $4 "ms)"}' "$BENCHMARK_DIR/timings-$TIMESTAMP.csv" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

---

## Recommendations

EOF

# Generate recommendations based on results
if [ $AVG_P95 -lt 100 ]; then
    echo "✅ **Performance is excellent.** Database is meeting all targets." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Maintenance:**" >> "$REPORT_FILE"
    echo "- Continue regular ANALYZE operations" >> "$REPORT_FILE"
    echo "- Monitor query performance trends" >> "$REPORT_FILE"
    echo "- Keep materialized views refreshed" >> "$REPORT_FILE"
elif [ $AVG_P95 -lt 500 ]; then
    echo "✅ **Performance is acceptable** but has room for improvement." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Optimization Opportunities:**" >> "$REPORT_FILE"
    echo "- Review slow queries and add indexes" >> "$REPORT_FILE"
    echo "- Consider additional materialized views" >> "$REPORT_FILE"
    echo "- Run VACUUM ANALYZE on large tables" >> "$REPORT_FILE"
else
    echo "⚠️  **Performance needs optimization.** Several queries are slow." >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Immediate Actions:**" >> "$REPORT_FILE"
    echo "- Review query plans for slow queries (EXPLAIN ANALYZE)" >> "$REPORT_FILE"
    echo "- Add missing indexes on frequently queried columns" >> "$REPORT_FILE"
    echo "- Consider partitioning large tables (events, aircraft)" >> "$REPORT_FILE"
    echo "- Run VACUUM FULL on bloated tables" >> "$REPORT_FILE"
    echo "- Increase PostgreSQL shared_buffers if system has RAM" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

---

## Database Statistics

EOF

# Add database statistics
psql -d ntsb_aviation -t -c "SELECT 'Database Size: ' || pg_size_pretty(pg_database_size('ntsb_aviation'));" >> "$REPORT_FILE"
psql -d ntsb_aviation -t -c "SELECT 'Total Rows: ' || SUM(n_live_tup)::text FROM pg_stat_user_tables WHERE schemaname = 'public';" >> "$REPORT_FILE"
psql -d ntsb_aviation -t -c "SELECT 'Total Indexes: ' || COUNT(*)::text FROM pg_indexes WHERE schemaname = 'public';" >> "$REPORT_FILE"
psql -d ntsb_aviation -t -c "SELECT 'Materialized Views: ' || COUNT(*)::text FROM pg_matviews WHERE schemaname = 'public';" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

---

## Next Steps

1. **Review Report:** Read full performance analysis above
2. **Address Slow Queries:** Investigate and optimize queries with p95 > 500ms
3. **Run ANALYZE:** Update table statistics with `ANALYZE;`
4. **Refresh MVs:** Update materialized views with `/refresh-mvs`
5. **Validate Schema:** Check for data quality issues with `/validate-schema`

---

## Files Generated

- **Report:** $REPORT_FILE
- **Timings CSV:** $BENCHMARK_DIR/timings-$TIMESTAMP.csv
- **Query Plans:** See console output above

---

**Benchmark Complete** ✅
EOF

# Display report
cat "$REPORT_FILE"

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo ""
echo "📊 Performance Report: $REPORT_FILE"
echo "📈 Timing Data (CSV): $BENCHMARK_DIR/timings-$TIMESTAMP.csv"
echo ""
echo "Summary:"
echo "  Average p50: ${AVG_P50}ms"
echo "  Average p95: ${AVG_P95}ms"
echo "  Average p99: ${AVG_P99}ms"
echo "  Target Status: $([ $AVG_P95 -lt 100 ] && echo "✅ MEETING" || echo "⚠️  NEEDS WORK")"
echo ""
```

---

## SUCCESS CRITERIA

✅ Database connection verified
✅ 10+ queries benchmarked across multiple categories
✅ p50, p95, p99 latencies measured for each query
✅ Materialized view performance tested
✅ Index utilization checked
✅ Query plans analyzed
✅ Comprehensive performance report generated
✅ Recommendations provided based on results

---

## PERFORMANCE TARGETS

**Target Latencies:**
- **p50:** <10ms for simple queries, <50ms for complex
- **p95:** <100ms for 95% of queries
- **p99:** <500ms for heavy aggregations

**Status Indicators:**
- **✅ EXCELLENT:** p95 < 100ms
- **✅ GOOD:** p95 100-500ms
- **⚠️  ACCEPTABLE:** p95 500-1000ms
- **❌ SLOW:** p95 > 1000ms

---

## DELIVERABLES

1. **Performance Report:** Comprehensive markdown report in /tmp/NTSB_Datasets/benchmarks/
2. **Timing Data:** CSV file with all query measurements
3. **Console Output:** Real-time benchmark results
4. **Query Plans:** Execution plan analysis for complex queries
5. **Recommendations:** Actionable optimization suggestions

---

## RELATED COMMANDS

**Database Optimization:**
- `/validate-schema` - Check database integrity before benchmarking
- `/refresh-mvs` - Update materialized views to improve query performance
- `/cleanup-staging` - Vacuum and optimize database after large operations

**Workflow Integration:**
```bash
# After data load or schema changes
/validate-schema          # Verify data integrity
/benchmark                # Measure performance impact
/refresh-mvs              # Update analytics views
/benchmark                # Verify improvement

# Regular performance monitoring
/benchmark quick          # Quick check (5 tests, 2 minutes)
# Review report
# Address any slow queries

# Performance optimization cycle
/benchmark                # Baseline performance
# Add indexes, optimize queries
psql -d ntsb_aviation -f scripts/optimize_queries.sql
/benchmark                # Measure improvement
/benchmark compare        # Compare against baseline
```

---

## TROUBLESHOOTING

**Slow Benchmark Execution:**
- Reduce iterations (default 10) by modifying benchmark_query function
- Use `/benchmark quick` for faster results
- Check system load: `top` or `htop`

**Inconsistent Results:**
- Clear OS cache: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`
- Run benchmark multiple times and average
- Check for concurrent database operations

**Query Timeout:**
- Increase statement_timeout in PostgreSQL: `SET statement_timeout = '60s';`
- Skip slow queries if needed
- Investigate query plan with EXPLAIN ANALYZE

---

## NOTES

### When to Run

**Recommended Times:**
- After initial data load (establish baseline)
- After schema modifications or index additions
- After query optimizations
- Before and after major database changes
- Monthly performance monitoring

### Performance Impact

- **Duration:** 5-10 minutes for full suite
- **Database Load:** Read-only queries, moderate CPU usage
- **Safe to Run:** Can run while database is in use (queries are non-destructive)

### Quick Mode

For faster benchmarking (5 essential queries only):
```bash
/benchmark quick
```

### Baseline Comparison

To compare against a baseline:
1. Run `/benchmark` and save report
2. Make optimizations
3. Run `/benchmark compare` to see improvements

---

**EXECUTE NOW - Measure database query performance.**
