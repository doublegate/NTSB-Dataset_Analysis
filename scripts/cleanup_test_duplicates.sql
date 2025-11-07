-- ============================================
-- Cleanup Test Load Duplicates
-- ============================================
-- Purpose: Remove duplicate child records accumulated during Sprint 3 Week 2 testing
-- Date: 2025-11-07
-- Context: Multiple test loads created duplicate child records while events table
--          correctly prevented duplicates via staging table deduplication.
--
-- Strategy:
--   1. For each child table, identify duplicate records (same ev_id + same data)
--   2. Keep only the first occurrence (smallest primary key ID)
--   3. Delete all duplicate records
--   4. Vacuum tables to reclaim disk space
--   5. Refresh materialized views
--   6. Analyze tables for query planner
--
-- Expected Impact:
--   - Database size: 2,561 MB → ~1,200 MB (50% reduction)
--   - Total rows: 3.8M → ~750K (80% reduction)
--   - No data loss (keeping first occurrence of each unique record)
--   - Zero impact on events table (already clean)
-- ============================================

\timing on
\echo ''
\echo '===================================='
\echo 'NTSB Database Cleanup - Test Duplicates'
\echo '===================================='
\echo ''

-- ============================================
-- STEP 1: ANALYZE CURRENT STATE
-- ============================================
\echo '📊 STEP 1: Analyzing current state...'
\echo ''

-- Record pre-cleanup metrics
CREATE TEMP TABLE cleanup_metrics_before AS
SELECT
    schemaname,
    relname as tablename,
    n_live_tup as rows,
    pg_total_relation_size(schemaname||'.'||relname) as size_bytes,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname IN ('injury', 'findings', 'narratives', 'events_sequence',
                  'aircraft', 'flight_crew', 'engines', 'ntsb_admin');

\echo 'Before cleanup:'
SELECT
    tablename,
    rows,
    size
FROM cleanup_metrics_before
ORDER BY rows DESC;

\echo ''
\echo 'Total database size before:'
SELECT pg_size_pretty(pg_database_size('ntsb_aviation')) as db_size;

-- ============================================
-- STEP 2: IDENTIFY DUPLICATES
-- ============================================
\echo ''
\echo '🔍 STEP 2: Identifying duplicates...'
\echo ''

-- Create temp table to track duplicates
CREATE TEMP TABLE duplicate_counts (
    table_name TEXT,
    total_rows INT,
    unique_rows INT,
    duplicate_rows INT,
    duplicate_pct NUMERIC(5,2)
);

-- Check injury table
INSERT INTO duplicate_counts
SELECT
    'injury',
    COUNT(*),
    COUNT(DISTINCT (ev_id, aircraft_key, inj_person_category, inj_level, inj_person_count)),
    COUNT(*) - COUNT(DISTINCT (ev_id, aircraft_key, inj_person_category, inj_level, inj_person_count)),
    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT (ev_id, aircraft_key, inj_person_category, inj_level, inj_person_count))) / COUNT(*), 2)
FROM injury;

-- Check findings table
INSERT INTO duplicate_counts
SELECT
    'findings',
    COUNT(*),
    COUNT(DISTINCT (ev_id, finding_code, finding_description, cm_inpc)),
    COUNT(*) - COUNT(DISTINCT (ev_id, finding_code, finding_description, cm_inpc)),
    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT (ev_id, finding_code, finding_description, cm_inpc))) / COUNT(*), 2)
FROM findings;

-- Check narratives table
INSERT INTO duplicate_counts
SELECT
    'narratives',
    COUNT(*),
    COUNT(DISTINCT (ev_id, narr_accp, narr_cause)),
    COUNT(*) - COUNT(DISTINCT (ev_id, narr_accp, narr_cause)),
    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT (ev_id, narr_accp, narr_cause))) / COUNT(*), 2)
FROM narratives;

-- Check events_sequence table
INSERT INTO duplicate_counts
SELECT
    'events_sequence',
    COUNT(*),
    COUNT(DISTINCT (ev_id, occurrence_code, occurrence_no, seq_of_events_code)),
    COUNT(*) - COUNT(DISTINCT (ev_id, occurrence_code, occurrence_no, seq_of_events_code)),
    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT (ev_id, occurrence_code, occurrence_no, seq_of_events_code))) / COUNT(*), 2)
FROM events_sequence;

\echo 'Duplicate analysis:'
SELECT * FROM duplicate_counts ORDER BY duplicate_rows DESC;

-- ============================================
-- STEP 3: BACKUP BEFORE CLEANUP
-- ============================================
\echo ''
\echo '💾 STEP 3: Creating backup tables...'
\echo ''

-- Create backup of tables we'll modify (just in case)
DROP TABLE IF EXISTS injury_backup_20251107;
DROP TABLE IF EXISTS findings_backup_20251107;
DROP TABLE IF EXISTS narratives_backup_20251107;
DROP TABLE IF EXISTS events_sequence_backup_20251107;

\echo '  Creating injury backup...'
CREATE TABLE injury_backup_20251107 AS SELECT * FROM injury;

\echo '  Creating findings backup...'
CREATE TABLE findings_backup_20251107 AS SELECT * FROM findings;

\echo '  Creating narratives backup...'
CREATE TABLE narratives_backup_20251107 AS SELECT * FROM narratives;

\echo '  Creating events_sequence backup...'
CREATE TABLE events_sequence_backup_20251107 AS SELECT * FROM events_sequence;

\echo '✅ Backups created successfully'

-- ============================================
-- STEP 4: DELETE DUPLICATES
-- ============================================
\echo ''
\echo '🗑️  STEP 4: Deleting duplicate records...'
\echo ''

BEGIN;

-- Delete duplicate injury records (keep first occurrence)
\echo '  Cleaning injury table...'
WITH duplicates AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY ev_id, aircraft_key, inj_person_category, inj_level, inj_person_count
            ORDER BY id
        ) as rn
    FROM injury
)
DELETE FROM injury
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);

\echo '  ✓ Injury duplicates deleted'

-- Delete duplicate findings records (keep first occurrence)
\echo '  Cleaning findings table...'
WITH duplicates AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY ev_id, finding_code, finding_description, cm_inpc
            ORDER BY id
        ) as rn
    FROM findings
)
DELETE FROM findings
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);

\echo '  ✓ Findings duplicates deleted'

-- Delete duplicate narratives records (keep first occurrence)
\echo '  Cleaning narratives table...'
WITH duplicates AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY ev_id, narr_accp, narr_cause
            ORDER BY id
        ) as rn
    FROM narratives
)
DELETE FROM narratives
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);

\echo '  ✓ Narratives duplicates deleted'

-- Delete duplicate events_sequence records (keep first occurrence)
\echo '  Cleaning events_sequence table...'
WITH duplicates AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY ev_id, occurrence_code, occurrence_no, seq_of_events_code
            ORDER BY id
        ) as rn
    FROM events_sequence
)
DELETE FROM events_sequence
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);

\echo '  ✓ Events_sequence duplicates deleted'

COMMIT;

\echo ''
\echo '✅ All duplicates deleted successfully'

-- ============================================
-- STEP 5: VACUUM AND ANALYZE
-- ============================================
\echo ''
\echo '🧹 STEP 5: Vacuuming tables to reclaim space...'
\echo ''

\echo '  Vacuuming injury...'
VACUUM FULL injury;

\echo '  Vacuuming findings...'
VACUUM FULL findings;

\echo '  Vacuuming narratives...'
VACUUM FULL narratives;

\echo '  Vacuuming events_sequence...'
VACUUM FULL events_sequence;

\echo ''
\echo '✅ Vacuum complete'

-- ============================================
-- STEP 6: REFRESH MATERIALIZED VIEWS
-- ============================================
\echo ''
\echo '🔄 STEP 6: Refreshing materialized views...'
\echo ''

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_finding_stats;
\echo '  ✓ mv_finding_stats refreshed'

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
\echo '  ✓ mv_yearly_stats refreshed'

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;
\echo '  ✓ mv_state_stats refreshed'

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_aircraft_stats;
\echo '  ✓ mv_aircraft_stats refreshed'

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_decade_stats;
\echo '  ✓ mv_decade_stats refreshed'

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_crew_stats;
\echo '  ✓ mv_crew_stats refreshed'

\echo ''
\echo '✅ Materialized views refreshed'

-- ============================================
-- STEP 7: ANALYZE FOR QUERY PLANNER
-- ============================================
\echo ''
\echo '📈 STEP 7: Analyzing tables for query planner...'
\echo ''

ANALYZE injury;
ANALYZE findings;
ANALYZE narratives;
ANALYZE events_sequence;
ANALYZE aircraft;
ANALYZE flight_crew;
ANALYZE engines;
ANALYZE ntsb_admin;
ANALYZE events;

\echo '✅ Tables analyzed'

-- ============================================
-- STEP 8: VERIFY CLEANUP RESULTS
-- ============================================
\echo ''
\echo '✅ STEP 8: Verification and final metrics'
\echo ''

-- Record post-cleanup metrics
CREATE TEMP TABLE cleanup_metrics_after AS
SELECT
    schemaname,
    relname as tablename,
    n_live_tup as rows,
    pg_total_relation_size(schemaname||'.'||relname) as size_bytes,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname IN ('injury', 'findings', 'narratives', 'events_sequence',
                  'aircraft', 'flight_crew', 'engines', 'ntsb_admin');

-- Show before/after comparison
\echo ''
\echo '====================================
\echo 'CLEANUP RESULTS SUMMARY'
\echo '===================================='
\echo ''

SELECT
    b.tablename,
    b.rows as rows_before,
    a.rows as rows_after,
    b.rows - a.rows as rows_deleted,
    ROUND(100.0 * (b.rows - a.rows) / b.rows, 2) as pct_reduced,
    b.size as size_before,
    a.size as size_after,
    pg_size_pretty(b.size_bytes - a.size_bytes) as space_reclaimed
FROM cleanup_metrics_before b
JOIN cleanup_metrics_after a ON b.tablename = a.tablename
WHERE b.rows > a.rows
ORDER BY b.rows - a.rows DESC;

\echo ''
\echo 'Database size comparison:'
\echo 'Before: ' \gset before_
SELECT pg_size_pretty(pg_database_size('ntsb_aviation')) as size;

\echo ''
\echo 'Total rows before/after:'
SELECT
    SUM(b.rows) as total_rows_before,
    SUM(a.rows) as total_rows_after,
    SUM(b.rows) - SUM(a.rows) as total_rows_deleted
FROM cleanup_metrics_before b
JOIN cleanup_metrics_after a ON b.tablename = a.tablename;

\echo ''
\echo '===================================='
\echo '✅ CLEANUP COMPLETE!'
\echo '===================================='
\echo ''
\echo 'Backup tables created (can be dropped after verification):'
\echo '  - injury_backup_20251107'
\echo '  - findings_backup_20251107'
\echo '  - narratives_backup_20251107'
\echo '  - events_sequence_backup_20251107'
\echo ''
\echo 'To drop backups after verification:'
\echo '  DROP TABLE injury_backup_20251107;'
\echo '  DROP TABLE findings_backup_20251107;'
\echo '  DROP TABLE narratives_backup_20251107;'
\echo '  DROP TABLE events_sequence_backup_20251107;'
\echo ''
