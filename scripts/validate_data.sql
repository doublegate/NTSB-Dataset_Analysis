-- validate_data.sql - Data Quality Validation Queries
-- Phase 1 Sprint 1: Data Quality Checks
-- Version: 1.0.0
-- Date: 2025-11-05
--
-- Usage:
--   psql -d ntsb_aviation -f scripts/validate_data.sql

\echo '============================================================'
\echo 'NTSB Aviation Database - Data Quality Validation'
\echo '============================================================'
\echo ''

-- ============================================
-- 1. ROW COUNTS
-- ============================================
\echo '1. ROW COUNTS'
\echo '------------------------------------------------------------'

SELECT 'events' AS table_name, COUNT(*) AS row_count FROM events
UNION ALL
SELECT 'aircraft', COUNT(*) FROM aircraft
UNION ALL
SELECT 'Flight_Crew', COUNT(*) FROM Flight_Crew
UNION ALL
SELECT 'injury', COUNT(*) FROM injury
UNION ALL
SELECT 'Findings', COUNT(*) FROM Findings
UNION ALL
SELECT 'Occurrences', COUNT(*) FROM Occurrences
UNION ALL
SELECT 'seq_of_events', COUNT(*) FROM seq_of_events
UNION ALL
SELECT 'Events_Sequence', COUNT(*) FROM Events_Sequence
UNION ALL
SELECT 'engines', COUNT(*) FROM engines
UNION ALL
SELECT 'narratives', COUNT(*) FROM narratives
UNION ALL
SELECT 'NTSB_Admin', COUNT(*) FROM NTSB_Admin
ORDER BY table_name;

\echo ''

-- ============================================
-- 2. PRIMARY KEY VALIDATION
-- ============================================
\echo '2. PRIMARY KEY VALIDATION'
\echo '------------------------------------------------------------'

-- Events: Check for duplicate ev_id
SELECT
    'events.ev_id' AS check_name,
    COUNT(*) AS total_records,
    COUNT(DISTINCT ev_id) AS unique_values,
    COUNT(*) - COUNT(DISTINCT ev_id) AS duplicates,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT ev_id) THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM events;

-- Aircraft: Check for duplicate Aircraft_Key
SELECT
    'aircraft.Aircraft_Key' AS check_name,
    COUNT(*) AS total_records,
    COUNT(DISTINCT Aircraft_Key) AS unique_values,
    COUNT(*) - COUNT(DISTINCT Aircraft_Key) AS duplicates,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT Aircraft_Key) THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM aircraft;

\echo ''

-- ============================================
-- 3. NULL VALUE CHECKS
-- ============================================
\echo '3. NULL VALUE CHECKS (Key Fields)'
\echo '------------------------------------------------------------'

-- Events: Critical fields
SELECT
    'events' AS table_name,
    ROUND(100.0 * COUNT(*) FILTER (WHERE ev_id IS NULL) / COUNT(*), 2) AS ev_id_null_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE ev_date IS NULL) / COUNT(*), 2) AS ev_date_null_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE dec_latitude IS NULL) / COUNT(*), 2) AS latitude_null_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE dec_longitude IS NULL) / COUNT(*), 2) AS longitude_null_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE ev_highest_injury IS NULL) / COUNT(*), 2) AS injury_null_pct
FROM events;

\echo ''

-- ============================================
-- 4. DATA INTEGRITY CHECKS
-- ============================================
\echo '4. DATA INTEGRITY CHECKS'
\echo '------------------------------------------------------------'

-- Coordinate validity
SELECT
    'Coordinate Validity' AS check_name,
    COUNT(*) FILTER (
        WHERE dec_latitude IS NOT NULL
        AND dec_longitude IS NOT NULL
    ) AS total_with_coords,
    COUNT(*) FILTER (
        WHERE dec_latitude BETWEEN -90 AND 90
        AND dec_longitude BETWEEN -180 AND 180
    ) AS valid_coords,
    COUNT(*) FILTER (
        WHERE (dec_latitude NOT BETWEEN -90 AND 90 OR dec_longitude NOT BETWEEN -180 AND 180)
        AND dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
    ) AS invalid_coords,
    CASE
        WHEN COUNT(*) FILTER (
            WHERE (dec_latitude NOT BETWEEN -90 AND 90 OR dec_longitude NOT BETWEEN -180 AND 180)
            AND dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
        ) = 0 THEN 'PASS ✓'
        ELSE 'WARN ⚠'
    END AS status
FROM events;

-- Date validity (1962-present)
SELECT
    'Date Range Validity' AS check_name,
    COUNT(*) AS total_events,
    MIN(ev_date) AS earliest_date,
    MAX(ev_date) AS latest_date,
    COUNT(*) FILTER (WHERE ev_date < '1962-01-01') AS before_1962,
    COUNT(*) FILTER (WHERE ev_date > CURRENT_DATE + INTERVAL '1 year') AS future_dates,
    CASE
        WHEN COUNT(*) FILTER (
            WHERE ev_date < '1962-01-01' OR ev_date > CURRENT_DATE + INTERVAL '1 year'
        ) = 0 THEN 'PASS ✓'
        ELSE 'WARN ⚠'
    END AS status
FROM events;

\echo ''

-- ============================================
-- 5. FOREIGN KEY VALIDATION
-- ============================================
\echo '5. FOREIGN KEY VALIDATION'
\echo '------------------------------------------------------------'

-- Aircraft -> Events
SELECT
    'aircraft.ev_id -> events.ev_id' AS relationship,
    COUNT(*) AS aircraft_records,
    COUNT(DISTINCT a.ev_id) AS unique_ev_ids,
    COUNT(*) FILTER (WHERE e.ev_id IS NULL) AS orphaned_records,
    CASE
        WHEN COUNT(*) FILTER (WHERE e.ev_id IS NULL) = 0 THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM aircraft a
LEFT JOIN events e ON a.ev_id = e.ev_id;

-- Flight_Crew -> Events
SELECT
    'Flight_Crew.ev_id -> events.ev_id' AS relationship,
    COUNT(*) AS crew_records,
    COUNT(DISTINCT fc.ev_id) AS unique_ev_ids,
    COUNT(*) FILTER (WHERE e.ev_id IS NULL) AS orphaned_records,
    CASE
        WHEN COUNT(*) FILTER (WHERE e.ev_id IS NULL) = 0 THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM Flight_Crew fc
LEFT JOIN events e ON fc.ev_id = e.ev_id;

-- Findings -> Events
SELECT
    'Findings.ev_id -> events.ev_id' AS relationship,
    COUNT(*) AS findings_records,
    COUNT(DISTINCT f.ev_id) AS unique_ev_ids,
    COUNT(*) FILTER (WHERE e.ev_id IS NULL) AS orphaned_records,
    CASE
        WHEN COUNT(*) FILTER (WHERE e.ev_id IS NULL) = 0 THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM Findings f
LEFT JOIN events e ON f.ev_id = e.ev_id;

-- Narratives -> Events
SELECT
    'narratives.ev_id -> events.ev_id' AS relationship,
    COUNT(*) AS narrative_records,
    COUNT(DISTINCT n.ev_id) AS unique_ev_ids,
    COUNT(*) FILTER (WHERE e.ev_id IS NULL) AS orphaned_records,
    CASE
        WHEN COUNT(*) FILTER (WHERE e.ev_id IS NULL) = 0 THEN 'PASS ✓'
        ELSE 'FAIL ✗'
    END AS status
FROM narratives n
LEFT JOIN events e ON n.ev_id = e.ev_id;

\echo ''

-- ============================================
-- 6. PARTITION VALIDATION
-- ============================================
\echo '6. PARTITION VALIDATION (Events by Year)'
\echo '------------------------------------------------------------'

SELECT
    CASE
        WHEN ev_year BETWEEN 1960 AND 1969 THEN 'events_1960s'
        WHEN ev_year BETWEEN 1970 AND 1979 THEN 'events_1970s'
        WHEN ev_year BETWEEN 1980 AND 1989 THEN 'events_1980s'
        WHEN ev_year BETWEEN 1990 AND 1999 THEN 'events_1990s'
        WHEN ev_year BETWEEN 2000 AND 2009 THEN 'events_2000s'
        WHEN ev_year BETWEEN 2010 AND 2019 THEN 'events_2010s'
        WHEN ev_year BETWEEN 2020 AND 2029 THEN 'events_2020s'
        ELSE 'unpartitioned'
    END AS partition_name,
    COUNT(*) AS record_count,
    MIN(ev_date) AS earliest_date,
    MAX(ev_date) AS latest_date
FROM events
GROUP BY
    CASE
        WHEN ev_year BETWEEN 1960 AND 1969 THEN 'events_1960s'
        WHEN ev_year BETWEEN 1970 AND 1979 THEN 'events_1970s'
        WHEN ev_year BETWEEN 1980 AND 1989 THEN 'events_1980s'
        WHEN ev_year BETWEEN 1990 AND 1999 THEN 'events_1990s'
        WHEN ev_year BETWEEN 2000 AND 2009 THEN 'events_2000s'
        WHEN ev_year BETWEEN 2010 AND 2019 THEN 'events_2010s'
        WHEN ev_year BETWEEN 2020 AND 2029 THEN 'events_2020s'
        ELSE 'unpartitioned'
    END
ORDER BY partition_name;

\echo ''

-- ============================================
-- 7. INDEX VALIDATION
-- ============================================
\echo '7. INDEX STATUS'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_indexes
JOIN pg_class ON pg_indexes.indexname = pg_class.relname
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

\echo ''

-- ============================================
-- 8. DATA QUALITY SUMMARY
-- ============================================
\echo '8. DATA QUALITY SUMMARY'
\echo '------------------------------------------------------------'

SELECT
    'Total Events' AS metric,
    COUNT(*)::text AS value
FROM events
UNION ALL
SELECT
    'Events with Coordinates',
    COUNT(*) FILTER (WHERE dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL)::text
FROM events
UNION ALL
SELECT
    'Fatal Accidents',
    COUNT(*) FILTER (WHERE ev_highest_injury = 'FATL')::text
FROM events
UNION ALL
SELECT
    'Total Fatalities',
    SUM(COALESCE(inj_tot_f, 0))::text
FROM events
UNION ALL
SELECT
    'Aircraft Records',
    COUNT(*)::text
FROM aircraft
UNION ALL
SELECT
    'Crew Records',
    COUNT(*)::text
FROM Flight_Crew
UNION ALL
SELECT
    'Finding Records',
    COUNT(*)::text
FROM Findings
UNION ALL
SELECT
    'Narrative Records',
    COUNT(*)::text
FROM narratives
UNION ALL
SELECT
    'Events with Narratives',
    COUNT(DISTINCT ev_id)::text
FROM narratives
UNION ALL
SELECT
    'Date Range',
    MIN(ev_date)::text || ' to ' || MAX(ev_date)::text
FROM events;

\echo ''

-- ============================================
-- 9. GENERATED COLUMN VALIDATION
-- ============================================
\echo '9. GENERATED COLUMNS'
\echo '------------------------------------------------------------'

-- Check ev_year and ev_month generation
SELECT
    'ev_year/ev_month Generation' AS check_name,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE ev_year IS NOT NULL) AS ev_year_populated,
    COUNT(*) FILTER (WHERE ev_month IS NOT NULL) AS ev_month_populated,
    CASE
        WHEN COUNT(*) = COUNT(*) FILTER (WHERE ev_year IS NOT NULL)
        AND COUNT(*) = COUNT(*) FILTER (WHERE ev_month IS NOT NULL)
        THEN 'PASS ✓'
        ELSE 'WARN ⚠'
    END AS status
FROM events
WHERE ev_date IS NOT NULL;

-- Check location_geom generation
SELECT
    'location_geom Generation' AS check_name,
    COUNT(*) FILTER (
        WHERE dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
    ) AS coords_available,
    COUNT(*) FILTER (WHERE location_geom IS NOT NULL) AS geom_populated,
    CASE
        WHEN COUNT(*) FILTER (
            WHERE dec_latitude IS NOT NULL
            AND dec_longitude IS NOT NULL
        ) = COUNT(*) FILTER (WHERE location_geom IS NOT NULL)
        THEN 'PASS ✓'
        ELSE 'WARN ⚠'
    END AS status
FROM events;

\echo ''

-- ============================================
-- 10. DATABASE SIZE
-- ============================================
\echo '10. DATABASE SIZE'
\echo '------------------------------------------------------------'

SELECT
    pg_database.datname AS database_name,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS database_size
FROM pg_database
WHERE datname = current_database();

\echo ''

SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS indexes_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 15;

\echo ''
\echo '============================================================'
\echo 'Validation Complete'
\echo '============================================================'
