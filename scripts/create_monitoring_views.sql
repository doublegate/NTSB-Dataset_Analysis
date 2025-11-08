-- ============================================================================
-- NTSB Aviation Database Monitoring Views
-- ============================================================================
-- Created: 2025-11-07
-- Purpose: Provide metrics for monitoring dashboard and alerting
-- ============================================================================

-- View 1: Database Metrics - Size and Row Count Trends
-- ============================================================================
CREATE OR REPLACE VIEW vw_database_metrics AS
SELECT
    NOW() as metric_timestamp,
    schemaname,
    relname as tablename,
    n_live_tup as row_count,
    n_dead_tup as dead_rows,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as table_size_pretty,
    pg_total_relation_size(schemaname||'.'||relname) as table_size_bytes,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;

COMMENT ON VIEW vw_database_metrics IS
'Real-time database size and row count metrics for monitoring dashboard';


-- View 2: Data Quality Checks
-- ============================================================================
CREATE OR REPLACE VIEW vw_data_quality_checks AS
WITH recent_events AS (
    SELECT * FROM events
    WHERE ev_date >= CURRENT_DATE - INTERVAL '35 days'
       OR created_at >= CURRENT_DATE - INTERVAL '35 days'
),
quality_metrics AS (
    SELECT
        'recent_events' as metric_name,
        COUNT(*)::bigint as value,
        'Total events in last 35 days' as description
    FROM recent_events

    UNION ALL

    SELECT
        'missing_ev_id',
        COUNT(*)::bigint,
        'Events missing ev_id (critical)'
    FROM recent_events
    WHERE ev_id IS NULL

    UNION ALL

    SELECT
        'missing_ev_date',
        COUNT(*)::bigint,
        'Events missing ev_date (critical)'
    FROM recent_events
    WHERE ev_date IS NULL

    UNION ALL

    SELECT
        'missing_coordinates',
        COUNT(*)::bigint,
        'Events missing lat/lon coordinates'
    FROM recent_events
    WHERE dec_latitude IS NULL OR dec_longitude IS NULL

    UNION ALL

    SELECT
        'invalid_coordinates',
        COUNT(*)::bigint,
        'Events with invalid coordinates'
    FROM recent_events
    WHERE dec_latitude < -90 OR dec_latitude > 90
       OR dec_longitude < -180 OR dec_longitude > 180

    UNION ALL

    SELECT
        'orphaned_aircraft',
        COUNT(*)::bigint,
        'Aircraft records without matching event'
    FROM aircraft a
    LEFT JOIN events e ON a.ev_id = e.ev_id
    WHERE e.ev_id IS NULL

    UNION ALL

    SELECT
        'orphaned_findings',
        COUNT(*)::bigint,
        'Finding records without matching event'
    FROM findings f
    LEFT JOIN events e ON f.ev_id = e.ev_id
    WHERE e.ev_id IS NULL

    UNION ALL

    SELECT
        'orphaned_narratives',
        COUNT(*)::bigint,
        'Narrative records without matching event'
    FROM narratives n
    LEFT JOIN events e ON n.ev_id = e.ev_id
    WHERE e.ev_id IS NULL

    UNION ALL

    SELECT
        'duplicate_events',
        COUNT(*)::bigint,
        'Duplicate ev_id values (should be 0)'
    FROM (
        SELECT ev_id
        FROM events
        GROUP BY ev_id
        HAVING COUNT(*) > 1
    ) dups
)
SELECT
    NOW() as check_timestamp,
    metric_name,
    value as metric_value,
    description,
    CASE
        WHEN metric_name IN ('missing_ev_id', 'missing_ev_date', 'duplicate_events')
            AND value > 0 THEN 'CRITICAL'
        WHEN metric_name IN ('invalid_coordinates', 'orphaned_aircraft', 'orphaned_findings')
            AND value > 0 THEN 'WARNING'
        ELSE 'OK'
    END as severity
FROM quality_metrics;

COMMENT ON VIEW vw_data_quality_checks IS
'Data quality metrics for monitoring anomalies and integrity issues';


-- View 3: Monthly Event Trends
-- ============================================================================
CREATE OR REPLACE VIEW vw_monthly_event_trends AS
SELECT
    DATE_TRUNC('month', ev_date) as month,
    COUNT(*) as event_count,
    COUNT(DISTINCT ev_state) as states_affected,
    COUNT(*) FILTER (WHERE ev_highest_injury = 'FATL') as fatal_accidents,
    COUNT(*) FILTER (WHERE ev_highest_injury = 'SERS') as serious_injuries,
    COUNT(*) FILTER (WHERE ev_highest_injury IN ('MINR', 'NONE')) as minor_or_none
FROM events
WHERE ev_date >= CURRENT_DATE - INTERVAL '24 months'
GROUP BY DATE_TRUNC('month', ev_date)
ORDER BY month DESC;

COMMENT ON VIEW vw_monthly_event_trends IS
'Monthly event statistics for trend analysis and anomaly detection';


-- View 4: Database Health Summary
-- ============================================================================
CREATE OR REPLACE VIEW vw_database_health AS
SELECT
    NOW() as health_check_timestamp,
    pg_size_pretty(pg_database_size(current_database())) as database_size,
    pg_database_size(current_database()) as database_size_bytes,
    (SELECT COUNT(*) FROM events) as total_events,
    (SELECT COUNT(*) FROM aircraft) as total_aircraft,
    (SELECT COUNT(*) FROM findings) as total_findings,
    (SELECT COUNT(*) FROM narratives) as total_narratives,
    (SELECT COUNT(*) FROM flight_crew) as total_crew_records,
    (SELECT COUNT(*) FROM events WHERE ev_date >= CURRENT_DATE - INTERVAL '30 days') as events_last_30_days,
    (SELECT COUNT(*) FROM events WHERE ev_date >= CURRENT_DATE - INTERVAL '365 days') as events_last_year,
    (SELECT MIN(ev_date) FROM events) as earliest_event_date,
    (SELECT MAX(ev_date) FROM events) as latest_event_date,
    (SELECT COUNT(*) FROM pg_stat_user_tables WHERE schemaname = 'public') as table_count,
    (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public') as index_count,
    (SELECT COUNT(*) FROM pg_views WHERE schemaname = 'public') as view_count;

COMMENT ON VIEW vw_database_health IS
'Overall database health and statistics summary for monitoring dashboard';


-- ============================================================================
-- Verification Queries
-- ============================================================================
\echo 'Monitoring views created successfully!'
\echo ''
\echo 'Test queries:'
\echo 'SELECT * FROM vw_database_metrics LIMIT 5;'
\echo 'SELECT * FROM vw_data_quality_checks;'
\echo 'SELECT * FROM vw_monthly_event_trends LIMIT 12;'
\echo 'SELECT * FROM vw_database_health;'
