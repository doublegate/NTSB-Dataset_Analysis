-- optimize_queries.sql - Query Optimization & Performance Tuning
-- Phase 1 Sprint 2: Performance Infrastructure
-- Version: 1.0.0
-- Date: 2025-11-06
--
-- Usage:
--   psql -d ntsb_aviation -f scripts/optimize_queries.sql

\echo '============================================================'
\echo 'NTSB Aviation Database - Query Optimization'
\echo '============================================================'
\echo ''

-- ============================================
-- 1. QUERY PERFORMANCE MONITORING
-- ============================================
\echo '1. Query Performance Monitoring'
\echo '------------------------------------------------------------'

-- Note: pg_stat_statements extension must be enabled by superuser first
-- Check if available and reset statistics
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements') THEN
        PERFORM pg_stat_statements_reset();
        RAISE NOTICE 'pg_stat_statements enabled and reset';
    ELSE
        RAISE NOTICE 'pg_stat_statements not available (requires superuser to enable)';
    END IF;
END $$;

\echo ''

-- ============================================
-- 2. DROP EXISTING MATERIALIZED VIEWS (if any)
-- ============================================
\echo '2. Cleaning Up Existing Materialized Views'
\echo '------------------------------------------------------------'

DROP MATERIALIZED VIEW IF EXISTS mv_yearly_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_state_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_aircraft_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_decade_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_crew_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_finding_stats CASCADE;

\echo '✓ Existing materialized views dropped'
\echo ''

-- ============================================
-- 3. CREATE MATERIALIZED VIEWS
-- ============================================
\echo '3. Creating Materialized Views for Common Aggregations'
\echo '------------------------------------------------------------'

-- Yearly Statistics (optimized from schema.sql)
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT
    ev_year,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    ROUND(100.0 * SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_accident_rate,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
    ROUND(AVG(COALESCE(inj_tot_f, 0)), 2) as avg_fatalities_per_accident,
    SUM(CASE WHEN ev_highest_injury = 'SERS' THEN 1 ELSE 0 END) as serious_injury_accidents,
    SUM(COALESCE(inj_tot_s, 0)) as total_serious_injuries,
    SUM(COALESCE(inj_tot_m, 0)) as total_minor_injuries,
    SUM(COALESCE(inj_tot_n, 0)) as total_no_injuries,
    COUNT(DISTINCT CASE WHEN dec_latitude IS NOT NULL THEN ev_id END) as events_with_coords,
    COUNT(DISTINCT CASE WHEN probable_cause IS NOT NULL THEN ev_id END) as events_with_cause
FROM events e
GROUP BY ev_year
ORDER BY ev_year;

CREATE UNIQUE INDEX idx_mv_yearly_stats_year ON mv_yearly_stats(ev_year);

\echo '  ✓ mv_yearly_stats created'

-- State-Level Statistics (improved from schema.sql)
CREATE MATERIALIZED VIEW mv_state_stats AS
SELECT
    ev_state,
    COUNT(*) as accident_count,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
    ROUND(100.0 * SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_rate,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
    ROUND(AVG(dec_latitude), 4) as avg_latitude,
    ROUND(AVG(dec_longitude), 4) as avg_longitude,
    MIN(ev_date) as earliest_accident,
    MAX(ev_date) as latest_accident,
    COUNT(DISTINCT ev_year) as years_with_accidents
FROM events
WHERE ev_state IS NOT NULL
GROUP BY ev_state
ORDER BY accident_count DESC;

CREATE UNIQUE INDEX idx_mv_state_stats_state ON mv_state_stats(ev_state);

\echo '  ✓ mv_state_stats created'

-- Aircraft Statistics (NEW)
CREATE MATERIALIZED VIEW mv_aircraft_stats AS
SELECT
    acft_make,
    acft_model,
    COUNT(DISTINCT a.ev_id) as accident_count,
    SUM(CASE WHEN e.ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(CASE WHEN damage = 'DEST' THEN 1 ELSE 0 END) as destroyed_count,
    SUM(CASE WHEN damage = 'SUBS' THEN 1 ELSE 0 END) as substantial_damage_count,
    SUM(CASE WHEN damage = 'MINR' THEN 1 ELSE 0 END) as minor_damage_count,
    ROUND(AVG(COALESCE(cert_max_gr_wt, 0)), 0) as avg_max_weight,
    ROUND(AVG(COALESCE(num_eng, 0)), 1) as avg_num_engines,
    MIN(e.ev_date) as first_accident,
    MAX(e.ev_date) as latest_accident
FROM aircraft a
INNER JOIN events e ON a.ev_id = e.ev_id
WHERE acft_make IS NOT NULL AND acft_model IS NOT NULL
GROUP BY acft_make, acft_model
HAVING COUNT(DISTINCT a.ev_id) >= 5  -- Only aircraft with 5+ accidents
ORDER BY accident_count DESC;

CREATE UNIQUE INDEX idx_mv_aircraft_stats_make_model ON mv_aircraft_stats(acft_make, acft_model);
CREATE INDEX idx_mv_aircraft_stats_count ON mv_aircraft_stats(accident_count DESC);

\echo '  ✓ mv_aircraft_stats created'

-- Decade Statistics (NEW)
CREATE MATERIALIZED VIEW mv_decade_stats AS
SELECT
    FLOOR(ev_year / 10) * 10 as decade_start,
    (FLOOR(ev_year / 10) * 10 + 9) as decade_end,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
    ROUND(AVG(COALESCE(inj_tot_f, 0)), 2) as avg_fatalities_per_accident,
    COUNT(DISTINCT ev_state) as states_affected,
    COUNT(DISTINCT CASE WHEN dec_latitude IS NOT NULL THEN ev_id END) as events_with_coords,
    MIN(ev_date) as period_start,
    MAX(ev_date) as period_end
FROM events
GROUP BY FLOOR(ev_year / 10)
ORDER BY decade_start;

CREATE UNIQUE INDEX idx_mv_decade_stats_decade ON mv_decade_stats(decade_start);

\echo '  ✓ mv_decade_stats created'

-- Crew Statistics (NEW)
CREATE MATERIALIZED VIEW mv_crew_stats AS
SELECT
    crew_category,
    COUNT(*) as crew_count,
    ROUND(AVG(COALESCE(crew_age, 0)), 1) as avg_age,
    MIN(crew_age) as min_age,
    MAX(crew_age) as max_age,
    ROUND(AVG(COALESCE(pilot_tot_time, 0)), 0) as avg_total_hours,
    ROUND(AVG(COALESCE(pilot_make_time, 0)), 0) as avg_make_model_hours,
    COUNT(CASE WHEN pilot_cert LIKE '%ATP%' THEN 1 END) as atp_certified,
    COUNT(CASE WHEN pilot_cert LIKE '%COM%' THEN 1 END) as commercial_certified,
    COUNT(CASE WHEN pilot_cert LIKE '%PVT%' THEN 1 END) as private_certified,
    COUNT(CASE WHEN pilot_cert LIKE '%STU%' THEN 1 END) as student_pilots
FROM flight_crew
WHERE crew_category IS NOT NULL
GROUP BY crew_category
ORDER BY crew_count DESC;

CREATE UNIQUE INDEX idx_mv_crew_stats_category ON mv_crew_stats(crew_category);

\echo '  ✓ mv_crew_stats created'

-- Finding Statistics (NEW)
CREATE MATERIALIZED VIEW mv_finding_stats AS
SELECT
    finding_code,
    finding_description,
    COUNT(*) as occurrence_count,
    COUNT(CASE WHEN cm_inPC = TRUE THEN 1 END) as cited_in_probable_cause,
    ROUND(100.0 * COUNT(CASE WHEN cm_inPC = TRUE THEN 1 END) / COUNT(*), 2) as probable_cause_rate,
    COUNT(DISTINCT f.ev_id) as unique_events,
    MIN(e.ev_date) as first_occurrence,
    MAX(e.ev_date) as latest_occurrence
FROM findings f
INNER JOIN events e ON f.ev_id = e.ev_id
WHERE finding_code IS NOT NULL
GROUP BY finding_code, finding_description
HAVING COUNT(*) >= 10  -- Only findings with 10+ occurrences
ORDER BY occurrence_count DESC;

CREATE UNIQUE INDEX idx_mv_finding_stats_code ON mv_finding_stats(finding_code, finding_description);
CREATE INDEX idx_mv_finding_stats_count ON mv_finding_stats(occurrence_count DESC);

\echo '  ✓ mv_finding_stats created'
\echo ''

-- ============================================
-- 4. ADDITIONAL PERFORMANCE INDEXES
-- ============================================
\echo '4. Creating Additional Performance Indexes'
\echo '------------------------------------------------------------'

-- Events table composite indexes
CREATE INDEX IF NOT EXISTS idx_events_year_severity ON events(ev_year, ev_highest_injury);
CREATE INDEX IF NOT EXISTS idx_events_state_year ON events(ev_state, ev_year) WHERE ev_state IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_fatal ON events(ev_id) WHERE ev_highest_injury = 'FATL';
CREATE INDEX IF NOT EXISTS idx_events_has_coords ON events(ev_id) WHERE dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL;

\echo '  ✓ Events composite indexes created'

-- Aircraft table indexes
CREATE INDEX IF NOT EXISTS idx_aircraft_damage ON aircraft(damage);
CREATE INDEX IF NOT EXISTS idx_aircraft_category ON aircraft(acft_category);

\echo '  ✓ Aircraft indexes created'

-- Crew table indexes
CREATE INDEX IF NOT EXISTS idx_crew_age_not_null ON flight_crew(crew_age) WHERE crew_age IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_crew_cert ON flight_crew(pilot_cert) WHERE pilot_cert IS NOT NULL;

\echo '  ✓ Crew indexes created'

-- Findings table indexes
CREATE INDEX IF NOT EXISTS idx_findings_in_pc_true ON findings(ev_id, finding_code) WHERE cm_inPC = TRUE;

\echo '  ✓ Findings indexes created'
\echo ''

-- ============================================
-- 5. CREATE REFRESH FUNCTION
-- ============================================
\echo '5. Creating Materialized View Refresh Function'
\echo '------------------------------------------------------------'

CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS TABLE(view_name TEXT, refresh_time INTERVAL) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
BEGIN
    -- mv_yearly_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_yearly_stats;
    end_time := clock_timestamp();
    view_name := 'mv_yearly_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- mv_state_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_state_stats;
    end_time := clock_timestamp();
    view_name := 'mv_state_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- mv_aircraft_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_aircraft_stats;
    end_time := clock_timestamp();
    view_name := 'mv_aircraft_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- mv_decade_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_decade_stats;
    end_time := clock_timestamp();
    view_name := 'mv_decade_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- mv_crew_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_crew_stats;
    end_time := clock_timestamp();
    view_name := 'mv_crew_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;

    -- mv_finding_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_finding_stats;
    end_time := clock_timestamp();
    view_name := 'mv_finding_stats';
    refresh_time := end_time - start_time;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

\echo '✓ refresh_all_materialized_views() function created'
\echo ''

-- ============================================
-- 6. ANALYZE TABLES FOR QUERY PLANNER
-- ============================================
\echo '6. Analyzing Tables for Query Planner'
\echo '------------------------------------------------------------'

ANALYZE events;
ANALYZE aircraft;
ANALYZE flight_crew;
ANALYZE injury;
ANALYZE findings;
ANALYZE occurrences;
ANALYZE seq_of_events;
ANALYZE events_sequence;
ANALYZE engines;
ANALYZE narratives;
ANALYZE ntsb_admin;

-- Analyze staging schema too
ANALYZE staging.events;
ANALYZE staging.aircraft;
ANALYZE staging.flight_crew;

-- Analyze materialized views
ANALYZE mv_yearly_stats;
ANALYZE mv_state_stats;
ANALYZE mv_aircraft_stats;
ANALYZE mv_decade_stats;
ANALYZE mv_crew_stats;
ANALYZE mv_finding_stats;

\echo '✓ All tables and views analyzed'
\echo ''

-- ============================================
-- 7. SUMMARY
-- ============================================
\echo '7. Optimization Summary'
\echo '------------------------------------------------------------'

SELECT
    'Materialized Views' as component,
    COUNT(*) as count
FROM pg_matviews
WHERE schemaname = 'public'
UNION ALL
SELECT
    'Indexes',
    COUNT(*)
FROM pg_indexes
WHERE schemaname = 'public'
UNION ALL
SELECT
    'Tables',
    COUNT(*)
FROM pg_tables
WHERE schemaname = 'public';

\echo ''
\echo '============================================================'
\echo 'Query Optimization Complete!'
\echo ''
\echo 'Next Steps:'
\echo '  - Run performance benchmarks: psql -d ntsb_aviation -f scripts/benchmark_queries.sql'
\echo '  - Refresh materialized views: SELECT * FROM refresh_all_materialized_views();'
\echo '  - Monitor queries: SELECT * FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;'
\echo '============================================================'
