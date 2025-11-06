-- test_performance.sql - Performance Testing Queries
-- Phase 1 Sprint 1: Query Performance Validation
-- Version: 1.0.0
-- Date: 2025-11-05
--
-- Target: <100ms query response time
--
-- Usage:
--   psql -d ntsb_aviation -f scripts/test_performance.sql

\echo '============================================================'
\echo 'NTSB Aviation Database - Performance Testing'
\echo '============================================================'
\echo ''
\echo 'Target: < 100ms query response time'
\echo ''

\timing on

-- ============================================
-- 1. SIMPLE LOOKUPS (should be <10ms)
-- ============================================
\echo '1. SIMPLE LOOKUPS'
\echo '------------------------------------------------------------'

-- Single event lookup by primary key
\echo 'Test 1.1: Single event lookup by ev_id'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_state, ev_city, ev_highest_injury
FROM events
WHERE ev_id = '20230101000001';

\echo ''

-- Single aircraft lookup
\echo 'Test 1.2: Single aircraft lookup by Aircraft_Key'
EXPLAIN ANALYZE
SELECT Aircraft_Key, ev_id, acft_make, acft_model, damage
FROM aircraft
WHERE Aircraft_Key = '20230101000001001';

\echo ''

-- ============================================
-- 2. INDEXED QUERIES (should be <50ms)
-- ============================================
\echo '2. INDEXED QUERIES'
\echo '------------------------------------------------------------'

-- Query by state (indexed)
\echo 'Test 2.1: Events by state (indexed)'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_city, ev_highest_injury
FROM events
WHERE ev_state = 'CA'
LIMIT 100;

\echo ''

-- Query by date range (indexed)
\echo 'Test 2.2: Events by date range (indexed)'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_state, ev_highest_injury
FROM events
WHERE ev_date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY ev_date DESC
LIMIT 100;

\echo ''

-- Query by year (partition pruning)
\echo 'Test 2.3: Events by year (partition pruning)'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_state, ev_highest_injury
FROM events
WHERE ev_year = 2023;

\echo ''

-- Query by severity (indexed)
\echo 'Test 2.4: Fatal accidents (indexed)'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_state, ev_city, inj_tot_f
FROM events
WHERE ev_highest_injury = 'FATL'
AND ev_year = 2023
ORDER BY ev_date DESC
LIMIT 50;

\echo ''

-- ============================================
-- 3. JOIN QUERIES (should be <100ms)
-- ============================================
\echo '3. JOIN QUERIES'
\echo '------------------------------------------------------------'

-- Event + Aircraft join
\echo 'Test 3.1: Events with aircraft details'
EXPLAIN ANALYZE
SELECT
    e.ev_id,
    e.ev_date,
    e.ev_state,
    a.acft_make,
    a.acft_model,
    a.damage
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year = 2023
AND e.ev_state = 'CA'
LIMIT 100;

\echo ''

-- Event + Aircraft + Crew join
\echo 'Test 3.2: Events with aircraft and crew'
EXPLAIN ANALYZE
SELECT
    e.ev_id,
    e.ev_date,
    a.acft_make,
    a.acft_model,
    fc.pilot_tot_time,
    fc.crew_age
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
LEFT JOIN Flight_Crew fc ON a.Aircraft_Key = fc.Aircraft_Key
WHERE e.ev_year = 2023
LIMIT 100;

\echo ''

-- Event + Findings join (probable causes)
\echo 'Test 3.3: Events with probable cause findings'
EXPLAIN ANALYZE
SELECT
    e.ev_id,
    e.ev_date,
    e.ev_state,
    f.finding_code,
    f.finding_description
FROM events e
JOIN Findings f ON e.ev_id = f.ev_id
WHERE f.cm_inPC = TRUE
AND e.ev_year = 2023
LIMIT 100;

\echo ''

-- ============================================
-- 4. SPATIAL QUERIES (should be <100ms)
-- ============================================
\echo '4. SPATIAL QUERIES'
\echo '------------------------------------------------------------'

-- Spatial query: Find accidents near Los Angeles (50km radius)
\echo 'Test 4.1: Accidents within 50km of Los Angeles'
EXPLAIN ANALYZE
SELECT
    ev_id,
    ev_date,
    ev_city,
    ev_state,
    dec_latitude,
    dec_longitude,
    ST_Distance(
        location_geom,
        ST_MakePoint(-118.2437, 34.0522)::geography
    ) / 1000 AS distance_km
FROM events
WHERE location_geom IS NOT NULL
AND ST_DWithin(
    location_geom,
    ST_MakePoint(-118.2437, 34.0522)::geography,
    50000  -- 50km in meters
)
ORDER BY distance_km
LIMIT 50;

\echo ''

-- Spatial query: Bounding box search
\echo 'Test 4.2: Accidents in California bounding box'
EXPLAIN ANALYZE
SELECT ev_id, ev_date, ev_city, dec_latitude, dec_longitude
FROM events
WHERE dec_latitude BETWEEN 32.5 AND 42.0
AND dec_longitude BETWEEN -124.5 AND -114.0
AND ev_year >= 2020
LIMIT 100;

\echo ''

-- ============================================
-- 5. AGGREGATE QUERIES (should be <100ms)
-- ============================================
\echo '5. AGGREGATE QUERIES'
\echo '------------------------------------------------------------'

-- Count by state
\echo 'Test 5.1: Accident counts by state (2023)'
EXPLAIN ANALYZE
SELECT
    ev_state,
    COUNT(*) AS accident_count,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) AS fatal_count
FROM events
WHERE ev_year = 2023
AND ev_state IS NOT NULL
GROUP BY ev_state
ORDER BY accident_count DESC
LIMIT 10;

\echo ''

-- Monthly trends
\echo 'Test 5.2: Monthly accident trends (2023)'
EXPLAIN ANALYZE
SELECT
    ev_month,
    COUNT(*) AS accident_count,
    SUM(COALESCE(inj_tot_f, 0)) AS total_fatalities
FROM events
WHERE ev_year = 2023
GROUP BY ev_month
ORDER BY ev_month;

\echo ''

-- Aircraft make/model statistics
\echo 'Test 5.3: Top aircraft makes in accidents (2023)'
EXPLAIN ANALYZE
SELECT
    acft_make,
    COUNT(*) AS accident_count,
    SUM(CASE WHEN damage = 'DEST' THEN 1 ELSE 0 END) AS destroyed_count
FROM aircraft a
JOIN events e ON a.ev_id = e.ev_id
WHERE e.ev_year = 2023
AND acft_make IS NOT NULL
GROUP BY acft_make
ORDER BY accident_count DESC
LIMIT 10;

\echo ''

-- ============================================
-- 6. FULL-TEXT SEARCH (should be <200ms)
-- ============================================
\echo '6. FULL-TEXT SEARCH'
\echo '------------------------------------------------------------'

-- Search narratives
\echo 'Test 6.1: Full-text search for "engine failure"'
EXPLAIN ANALYZE
SELECT
    n.ev_id,
    e.ev_date,
    e.ev_state,
    ts_headline('english', n.narr_cause, to_tsquery('english', 'engine & failure')) AS headline
FROM narratives n
JOIN events e ON n.ev_id = e.ev_id
WHERE n.search_vector @@ to_tsquery('english', 'engine & failure')
AND e.ev_year >= 2020
ORDER BY e.ev_date DESC
LIMIT 20;

\echo ''

-- ============================================
-- 7. MATERIALIZED VIEW QUERIES (should be <10ms)
-- ============================================
\echo '7. MATERIALIZED VIEW QUERIES'
\echo '------------------------------------------------------------'

-- Yearly statistics
\echo 'Test 7.1: Yearly statistics (from materialized view)'
EXPLAIN ANALYZE
SELECT
    ev_year,
    total_accidents,
    fatal_accidents,
    total_fatalities
FROM mv_yearly_stats
WHERE ev_year >= 2020
ORDER BY ev_year DESC;

\echo ''

-- State statistics
\echo 'Test 7.2: State statistics (from materialized view)'
EXPLAIN ANALYZE
SELECT
    ev_state,
    accident_count,
    fatal_count
FROM mv_state_stats
ORDER BY accident_count DESC
LIMIT 10;

\echo ''

-- ============================================
-- 8. COMPLEX ANALYTICAL QUERIES (should be <500ms)
-- ============================================
\echo '8. COMPLEX ANALYTICAL QUERIES'
\echo '------------------------------------------------------------'

-- Accident trends with moving average
\echo 'Test 8.1: Monthly trends with 3-month moving average'
EXPLAIN ANALYZE
WITH monthly_counts AS (
    SELECT
        DATE_TRUNC('month', ev_date) AS month,
        COUNT(*) AS accident_count
    FROM events
    WHERE ev_year >= 2022
    GROUP BY DATE_TRUNC('month', ev_date)
)
SELECT
    month,
    accident_count,
    AVG(accident_count) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3month
FROM monthly_counts
ORDER BY month;

\echo ''

-- Top accident causes
\echo 'Test 8.2: Most common probable cause codes (2023)'
EXPLAIN ANALYZE
SELECT
    f.finding_code,
    f.finding_description,
    COUNT(*) AS occurrence_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM Findings f
JOIN events e ON f.ev_id = e.ev_id
WHERE f.cm_inPC = TRUE
AND e.ev_year = 2023
AND f.finding_code IS NOT NULL
GROUP BY f.finding_code, f.finding_description
ORDER BY occurrence_count DESC
LIMIT 10;

\echo ''

-- ============================================
-- 9. INDEX USAGE ANALYSIS
-- ============================================
\echo '9. INDEX USAGE STATISTICS'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC, tablename, indexname
LIMIT 20;

\echo ''

-- ============================================
-- 10. QUERY PERFORMANCE SUMMARY
-- ============================================
\echo '10. PERFORMANCE SUMMARY'
\echo '------------------------------------------------------------'

-- Table scan statistics
SELECT
    schemaname,
    relname AS table_name,
    seq_scan AS sequential_scans,
    seq_tup_read AS seq_tuples_read,
    idx_scan AS index_scans,
    idx_tup_fetch AS idx_tuples_fetched,
    ROUND(100.0 * idx_scan / NULLIF(seq_scan + idx_scan, 0), 2) AS index_usage_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY relname;

\echo ''

-- Cache hit ratio
SELECT
    'Buffer Cache Hit Ratio' AS metric,
    ROUND(
        100.0 * sum(blks_hit) / NULLIF(sum(blks_hit) + sum(blks_read), 0),
        2
    ) AS value
FROM pg_stat_database
WHERE datname = current_database();

\echo ''

\timing off

\echo '============================================================'
\echo 'Performance Testing Complete'
\echo '============================================================'
\echo ''
\echo 'Review results:'
\echo '  - Simple lookups should be < 10ms'
\echo '  - Indexed queries should be < 50ms'
\echo '  - Join queries should be < 100ms'
\echo '  - Spatial queries should be < 100ms'
\echo '  - Aggregate queries should be < 100ms'
\echo '  - Full-text search should be < 200ms'
\echo '  - Materialized views should be < 10ms'
\echo ''
\echo 'If queries exceed targets:'
\echo '  - Check EXPLAIN ANALYZE output for seq scans'
\echo '  - Verify indexes are being used'
\echo '  - Consider VACUUM ANALYZE'
\echo '  - Review partition pruning for year-based queries'
\echo ''
