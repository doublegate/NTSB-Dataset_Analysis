-- maintain_database.sql - NTSB Aviation Database Comprehensive Maintenance Script
-- Purpose: Groom and optimize database for performance and storage efficiency
-- Version: 1.0.0
-- Date: 2025-11-07
--
-- Usage:
--   psql -d ntsb_aviation -f scripts/maintain_database.sql
--   OR
--   ./scripts/maintain_database.sh (recommended - includes logging)
--
-- Recommended Frequency: Monthly (after data loads)

\echo '========================================='
\echo 'NTSB Aviation Database Maintenance Script'
\echo 'Version: 1.0.0'
\echo 'Started:' `date`
\echo '========================================='

-- ============================================
-- PHASE 1: Pre-Maintenance Metrics
-- ============================================
\echo ''
\echo 'PHASE 1: Pre-Maintenance Metrics'
\echo '---------------------------------'

-- Database size before maintenance
\echo 'Database size before maintenance:'
SELECT pg_size_pretty(pg_database_size('ntsb_aviation')) as database_size;

-- Table sizes before maintenance
\echo ''
\echo 'Table sizes before maintenance (top 10):'
SELECT
    schemaname,
    relname as tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||relname)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname) - pg_relation_size(schemaname||'.'||relname)) as index_size,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname NOT LIKE 'mv_%'  -- Exclude materialized views
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC
LIMIT 10;

-- Dead tuple summary
\echo ''
\echo 'Dead tuple summary:'
SELECT
    COUNT(*) as total_tables,
    SUM(n_live_tup) as total_live_rows,
    SUM(n_dead_tup) as total_dead_rows,
    ROUND(100.0 * SUM(n_dead_tup) / NULLIF(SUM(n_live_tup) + SUM(n_dead_tup), 0), 2) as overall_dead_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname NOT LIKE 'mv_%';

-- ============================================
-- PHASE 2: Duplicate Detection
-- ============================================
\echo ''
\echo 'PHASE 2: Duplicate Detection'
\echo '-----------------------------'

-- Check for duplicates in events table (should be 0)
\echo 'Checking for duplicate events (by ev_id)...'
SELECT COUNT(*) as duplicate_events
FROM (
    SELECT ev_id, COUNT(*) as cnt
    FROM events
    GROUP BY ev_id
    HAVING COUNT(*) > 1
) dupes;

-- Check for duplicate narratives (by ev_id - should be unique per event)
\echo 'Checking for duplicate narratives (by ev_id)...'
SELECT COUNT(*) as duplicate_narratives
FROM (
    SELECT ev_id, COUNT(*) as cnt
    FROM narratives
    GROUP BY ev_id
    HAVING COUNT(*) > 1
) dupes;

-- Note: Aircraft, Findings, Injury can have multiple records per ev_id by design
\echo ''
\echo 'Note: Aircraft, Findings, and Injury tables can have multiple records per ev_id by design.'

-- ============================================
-- PHASE 3: Data Quality Validation
-- ============================================
\echo ''
\echo 'PHASE 3: Data Quality Validation'
\echo '--------------------------------'

-- Missing critical fields
\echo 'Checking for missing critical fields...'
SELECT
    'events' as table_name,
    COUNT(*) FILTER (WHERE ev_id IS NULL) as missing_ev_id,
    COUNT(*) FILTER (WHERE ev_date IS NULL) as missing_ev_date,
    COUNT(*) as total_events
FROM events;

-- Invalid coordinates
\echo ''
\echo 'Checking for invalid coordinates...'
SELECT COUNT(*) as invalid_coords
FROM events
WHERE (dec_latitude < -90 OR dec_latitude > 90)
   OR (dec_longitude < -180 OR dec_longitude > 180);

-- Invalid dates
\echo ''
\echo 'Checking for invalid dates...'
SELECT
    COUNT(*) FILTER (WHERE ev_date < '1962-01-01') as before_1962,
    COUNT(*) FILTER (WHERE ev_date > CURRENT_DATE) as future_dates,
    COUNT(*) as total_events
FROM events;

-- Foreign key integrity checks
\echo ''
\echo 'Checking foreign key integrity (orphaned records)...'

-- Aircraft orphans
SELECT
    'aircraft' as table_name,
    COUNT(*) as orphaned_records
FROM aircraft a
LEFT JOIN events e ON a.ev_id = e.ev_id
WHERE e.ev_id IS NULL

UNION ALL

-- Findings orphans
SELECT
    'findings' as table_name,
    COUNT(*) as orphaned_records
FROM findings f
LEFT JOIN events e ON f.ev_id = e.ev_id
WHERE e.ev_id IS NULL

UNION ALL

-- Flight_Crew orphans
SELECT
    'flight_crew' as table_name,
    COUNT(*) as orphaned_records
FROM flight_crew fc
LEFT JOIN events e ON fc.ev_id = e.ev_id
WHERE e.ev_id IS NULL

UNION ALL

-- Narratives orphans
SELECT
    'narratives' as table_name,
    COUNT(*) as orphaned_records
FROM narratives n
LEFT JOIN events e ON n.ev_id = e.ev_id
WHERE e.ev_id IS NULL

UNION ALL

-- Injury orphans
SELECT
    'injury' as table_name,
    COUNT(*) as orphaned_records
FROM injury i
LEFT JOIN events e ON i.ev_id = e.ev_id
WHERE e.ev_id IS NULL;

-- ============================================
-- PHASE 4: Storage Optimization
-- ============================================
\echo ''
\echo 'PHASE 4: Storage Optimization'
\echo '-----------------------------'

-- VACUUM ANALYZE all tables
\echo 'Running VACUUM ANALYZE (this may take several minutes)...'
\echo 'This reclaims dead tuple space and updates query planner statistics.'
VACUUM ANALYZE;

\echo 'VACUUM ANALYZE completed.'

-- ============================================
-- PHASE 5: Refresh Materialized Views
-- ============================================
\echo ''
\echo 'PHASE 5: Refresh Materialized Views'
\echo '-----------------------------------'

-- Refresh all materialized views
\echo 'Refreshing all 6 materialized views...'
SELECT * FROM refresh_all_materialized_views();

-- ============================================
-- PHASE 6: Update Statistics
-- ============================================
\echo ''
\echo 'PHASE 6: Update Statistics'
\echo '-------------------------'

-- Update table statistics for query planner
\echo 'Updating table statistics for query planner...'
ANALYZE;

\echo 'Statistics updated.'

-- ============================================
-- PHASE 7: Index Health Check
-- ============================================
\echo ''
\echo 'PHASE 7: Index Health Check'
\echo '--------------------------'

-- Total index count
\echo 'Total index count:'
SELECT COUNT(*) as total_indexes
FROM pg_indexes
WHERE schemaname = 'public';

-- Unused indexes (idx_scan = 0) - excluding constraints
\echo ''
\echo 'Unused indexes (idx_scan = 0, excluding PK/unique constraints):'
SELECT
    schemaname,
    relname as table_name,
    indexrelname as index_name,
    pg_size_pretty(pg_relation_size(indexrelname::regclass)) as index_size,
    idx_scan as times_used
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND idx_scan = 0
  AND indexrelname NOT IN (
      SELECT indexrelname FROM pg_stat_user_indexes sui
      JOIN pg_index i ON sui.indexrelid = i.indexrelid
      WHERE i.indisunique OR i.indisprimary
  )
ORDER BY pg_relation_size(indexrelname::regclass) DESC;

-- Most used indexes
\echo ''
\echo 'Most used indexes (top 10):'
SELECT
    schemaname,
    relname as table_name,
    indexrelname as index_name,
    idx_scan as times_used,
    pg_size_pretty(pg_relation_size(indexrelname::regclass)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC
LIMIT 10;

-- Index size summary
\echo ''
\echo 'Index size summary:'
SELECT
    pg_size_pretty(SUM(pg_relation_size(indexrelname::regclass))) as total_index_size,
    COUNT(*) as total_indexes
FROM pg_stat_user_indexes
WHERE schemaname = 'public';

-- ============================================
-- PHASE 8: Performance Metrics
-- ============================================
\echo ''
\echo 'PHASE 8: Performance Metrics'
\echo '---------------------------'

-- Cache hit ratio (should be >90%)
\echo 'Cache hit ratio (should be >90%):'
SELECT
    ROUND(100.0 * sum(blks_hit) / NULLIF(sum(blks_hit) + sum(blks_read), 0), 2) as cache_hit_ratio
FROM pg_stat_database
WHERE datname = 'ntsb_aviation';

-- Table scan statistics
\echo ''
\echo 'Table scan statistics (top 5 by sequential scans):'
SELECT
    schemaname,
    relname as table_name,
    seq_scan as sequential_scans,
    idx_scan as index_scans,
    CASE
        WHEN seq_scan + idx_scan > 0
        THEN ROUND(100.0 * idx_scan / (seq_scan + idx_scan), 2)
        ELSE 0
    END as index_usage_pct,
    n_live_tup as rows
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname NOT LIKE 'mv_%'
ORDER BY seq_scan DESC
LIMIT 5;

-- ============================================
-- PHASE 9: Post-Maintenance Metrics
-- ============================================
\echo ''
\echo 'PHASE 9: Post-Maintenance Metrics'
\echo '---------------------------------'

-- Database size after maintenance
\echo 'Database size after maintenance:'
SELECT pg_size_pretty(pg_database_size('ntsb_aviation')) as database_size;

-- Table sizes after maintenance
\echo ''
\echo 'Table sizes after maintenance (top 10):'
SELECT
    schemaname,
    relname as tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname NOT LIKE 'mv_%'
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC
LIMIT 10;

-- Materialized view sizes
\echo ''
\echo 'Materialized view sizes:'
SELECT
    schemaname,
    relname as mv_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    n_live_tup as rows
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname LIKE 'mv_%'
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC;

-- ============================================
-- PHASE 10: Maintenance Summary
-- ============================================
\echo ''
\echo 'PHASE 10: Maintenance Summary'
\echo '----------------------------'

-- Row count summary
\echo 'Row count summary:'
SELECT
    'events' AS table_name, COUNT(*) AS row_count FROM events
UNION ALL
SELECT 'aircraft', COUNT(*) FROM aircraft
UNION ALL
SELECT 'flight_crew', COUNT(*) FROM flight_crew
UNION ALL
SELECT 'injury', COUNT(*) FROM injury
UNION ALL
SELECT 'findings', COUNT(*) FROM findings
UNION ALL
SELECT 'narratives', COUNT(*) FROM narratives
UNION ALL
SELECT 'engines', COUNT(*) FROM engines
UNION ALL
SELECT 'ntsb_admin', COUNT(*) FROM ntsb_admin
UNION ALL
SELECT 'events_sequence', COUNT(*) FROM events_sequence
ORDER BY row_count DESC;

\echo ''
\echo '========================================='
\echo 'Maintenance Complete!'
\echo 'Completed:' `date`
\echo '========================================='
\echo ''
\echo 'Summary:'
\echo '  ✓ Storage optimized (VACUUM ANALYZE)'
\echo '  ✓ Materialized views refreshed (6 views)'
\echo '  ✓ Statistics updated (ANALYZE)'
\echo '  ✓ Data quality validated'
\echo '  ✓ Index health checked'
\echo '  ✓ Performance metrics collected'
\echo ''
\echo 'Next steps:'
\echo '  - Review unused indexes for potential removal'
\echo '  - Monitor cache hit ratio (target: >90%)'
\echo '  - Check for orphaned records (should be 0)'
\echo '  - Run this script monthly after data loads'
\echo ''
