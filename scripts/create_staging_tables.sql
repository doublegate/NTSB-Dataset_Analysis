-- create_staging_tables.sql - Staging Table Infrastructure
-- Phase 1 Sprint 2: Historical Data Integration with Staging Tables
-- Version: 2.0.0
-- Date: 2025-11-06
--
-- Purpose: Create temporary staging tables for bulk data loads with deduplication
--          Allows loading data without constraints, then merging only unique records

-- ============================================
-- CREATE STAGING SCHEMA
-- ============================================

CREATE SCHEMA IF NOT EXISTS staging;

COMMENT ON SCHEMA staging IS 'Temporary staging area for bulk data loads with deduplication logic';

-- ============================================
-- DROP EXISTING STAGING TABLES (if any)
-- ============================================

-- Drop in reverse dependency order
DROP TABLE IF EXISTS staging.ntsb_admin CASCADE;
DROP TABLE IF EXISTS staging.narratives CASCADE;
DROP TABLE IF EXISTS staging.engines CASCADE;
DROP TABLE IF EXISTS staging.events_sequence CASCADE;
DROP TABLE IF EXISTS staging.seq_of_events CASCADE;
DROP TABLE IF EXISTS staging.occurrences CASCADE;
DROP TABLE IF EXISTS staging.findings CASCADE;
DROP TABLE IF EXISTS staging.injury CASCADE;
DROP TABLE IF EXISTS staging.flight_crew CASCADE;
DROP TABLE IF EXISTS staging.aircraft CASCADE;
DROP TABLE IF EXISTS staging.events CASCADE;

-- ============================================
-- CREATE STAGING TABLES (NO CONSTRAINTS)
-- ============================================

-- Staging tables copy structure from public schema but WITHOUT:
-- - Foreign key constraints (allows loading before validation)
-- - CHECK constraints (allows invalid data for analysis)
-- - Triggers (no auto-generated columns)
-- - Indexes (added separately for performance)

-- 1. Events staging table
CREATE TABLE staging.events (LIKE public.events INCLUDING DEFAULTS);

-- 2. Aircraft staging table
CREATE TABLE staging.aircraft (LIKE public.aircraft INCLUDING DEFAULTS);

-- 3. Flight crew staging table
CREATE TABLE staging.flight_crew (LIKE public.flight_crew INCLUDING DEFAULTS);

-- 4. Injury staging table
CREATE TABLE staging.injury (LIKE public.injury INCLUDING DEFAULTS);

-- 5. Findings staging table
CREATE TABLE staging.findings (LIKE public.findings INCLUDING DEFAULTS);

-- 6. Occurrences staging table
CREATE TABLE staging.occurrences (LIKE public.occurrences INCLUDING DEFAULTS);

-- 7. Sequence of events staging table
CREATE TABLE staging.seq_of_events (LIKE public.seq_of_events INCLUDING DEFAULTS);

-- 8. Events sequence staging table
CREATE TABLE staging.events_sequence (LIKE public.events_sequence INCLUDING DEFAULTS);

-- 9. Engines staging table
CREATE TABLE staging.engines (LIKE public.engines INCLUDING DEFAULTS);

-- 10. Narratives staging table
CREATE TABLE staging.narratives (LIKE public.narratives INCLUDING DEFAULTS);

-- 11. NTSB Admin staging table
CREATE TABLE staging.ntsb_admin (LIKE public.ntsb_admin INCLUDING DEFAULTS);

-- ============================================
-- CREATE STAGING INDEXES (for faster lookups)
-- ============================================

-- Events indexes (critical for duplicate detection)
CREATE INDEX idx_staging_events_ev_id ON staging.events(ev_id);
CREATE INDEX idx_staging_events_ev_date ON staging.events(ev_date);

-- Aircraft indexes (for foreign key verification)
CREATE INDEX idx_staging_aircraft_ev_id ON staging.aircraft(ev_id);
CREATE INDEX idx_staging_aircraft_key ON staging.aircraft(ev_id, aircraft_key);

-- Flight crew indexes
CREATE INDEX idx_staging_flight_crew_ev_id ON staging.flight_crew(ev_id);

-- Child table indexes for ev_id lookups
CREATE INDEX idx_staging_injury_ev_id ON staging.injury(ev_id);
CREATE INDEX idx_staging_findings_ev_id ON staging.findings(ev_id);
CREATE INDEX idx_staging_occurrences_ev_id ON staging.occurrences(ev_id);
CREATE INDEX idx_staging_seq_of_events_ev_id ON staging.seq_of_events(ev_id);
CREATE INDEX idx_staging_events_sequence_ev_id ON staging.events_sequence(ev_id);
CREATE INDEX idx_staging_engines_ev_id ON staging.engines(ev_id);
CREATE INDEX idx_staging_narratives_ev_id ON staging.narratives(ev_id);
CREATE INDEX idx_staging_ntsb_admin_ev_id ON staging.ntsb_admin(ev_id);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to truncate all staging tables at once
CREATE OR REPLACE FUNCTION staging.truncate_all_tables()
RETURNS void AS $$
BEGIN
    TRUNCATE TABLE staging.ntsb_admin CASCADE;
    TRUNCATE TABLE staging.narratives CASCADE;
    TRUNCATE TABLE staging.engines CASCADE;
    TRUNCATE TABLE staging.events_sequence CASCADE;
    TRUNCATE TABLE staging.seq_of_events CASCADE;
    TRUNCATE TABLE staging.occurrences CASCADE;
    TRUNCATE TABLE staging.findings CASCADE;
    TRUNCATE TABLE staging.injury CASCADE;
    TRUNCATE TABLE staging.flight_crew CASCADE;
    TRUNCATE TABLE staging.aircraft CASCADE;
    TRUNCATE TABLE staging.events CASCADE;

    RAISE NOTICE '✓ All staging tables truncated';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION staging.truncate_all_tables() IS 'Truncate all staging tables in correct dependency order';

-- Function to get row counts from all staging tables
CREATE OR REPLACE FUNCTION staging.get_row_counts()
RETURNS TABLE(table_name TEXT, row_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT 'events'::TEXT, COUNT(*)::BIGINT FROM staging.events
    UNION ALL
    SELECT 'aircraft'::TEXT, COUNT(*)::BIGINT FROM staging.aircraft
    UNION ALL
    SELECT 'flight_crew'::TEXT, COUNT(*)::BIGINT FROM staging.flight_crew
    UNION ALL
    SELECT 'injury'::TEXT, COUNT(*)::BIGINT FROM staging.injury
    UNION ALL
    SELECT 'findings'::TEXT, COUNT(*)::BIGINT FROM staging.findings
    UNION ALL
    SELECT 'occurrences'::TEXT, COUNT(*)::BIGINT FROM staging.occurrences
    UNION ALL
    SELECT 'seq_of_events'::TEXT, COUNT(*)::BIGINT FROM staging.seq_of_events
    UNION ALL
    SELECT 'events_sequence'::TEXT, COUNT(*)::BIGINT FROM staging.events_sequence
    UNION ALL
    SELECT 'engines'::TEXT, COUNT(*)::BIGINT FROM staging.engines
    UNION ALL
    SELECT 'narratives'::TEXT, COUNT(*)::BIGINT FROM staging.narratives
    UNION ALL
    SELECT 'ntsb_admin'::TEXT, COUNT(*)::BIGINT FROM staging.ntsb_admin
    ORDER BY 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION staging.get_row_counts() IS 'Get row counts from all staging tables';

-- Function to identify duplicate events between staging and production
CREATE OR REPLACE FUNCTION staging.identify_duplicate_events()
RETURNS TABLE(
    ev_id VARCHAR(20),
    in_staging BOOLEAN,
    in_production BOOLEAN,
    ev_date DATE,
    ev_city VARCHAR(100),
    ev_state CHAR(2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(s.ev_id, p.ev_id) as ev_id,
        (s.ev_id IS NOT NULL) as in_staging,
        (p.ev_id IS NOT NULL) as in_production,
        COALESCE(s.ev_date, p.ev_date) as ev_date,
        COALESCE(s.ev_city, p.ev_city) as ev_city,
        COALESCE(s.ev_state, p.ev_state) as ev_state
    FROM staging.events s
    FULL OUTER JOIN public.events p ON s.ev_id = p.ev_id
    WHERE s.ev_id IS NOT NULL OR p.ev_id IS NOT NULL
    ORDER BY COALESCE(s.ev_date, p.ev_date) DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION staging.identify_duplicate_events() IS 'Identify events that exist in both staging and production tables';

-- Function to get duplicate event statistics
CREATE OR REPLACE FUNCTION staging.get_duplicate_stats()
RETURNS TABLE(
    total_in_staging BIGINT,
    total_in_production BIGINT,
    duplicates BIGINT,
    unique_in_staging BIGINT,
    unique_in_production BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT
            COUNT(DISTINCT s.ev_id) as staging_count,
            COUNT(DISTINCT p.ev_id) as production_count,
            COUNT(DISTINCT CASE WHEN s.ev_id IS NOT NULL AND p.ev_id IS NOT NULL THEN s.ev_id END) as duplicate_count
        FROM staging.events s
        FULL OUTER JOIN public.events p ON s.ev_id = p.ev_id
    )
    SELECT
        staging_count,
        production_count,
        duplicate_count,
        staging_count - duplicate_count as unique_staging,
        production_count - duplicate_count as unique_production
    FROM stats;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION staging.get_duplicate_stats() IS 'Get statistics on duplicate events between staging and production';

-- ============================================
-- TABLE COMMENTS
-- ============================================

COMMENT ON TABLE staging.events IS 'Staging table for events - no constraints for bulk loading';
COMMENT ON TABLE staging.aircraft IS 'Staging table for aircraft - no foreign keys';
COMMENT ON TABLE staging.flight_crew IS 'Staging table for flight crew';
COMMENT ON TABLE staging.injury IS 'Staging table for injury records';
COMMENT ON TABLE staging.findings IS 'Staging table for investigation findings';
COMMENT ON TABLE staging.occurrences IS 'Staging table for occurrence events';
COMMENT ON TABLE staging.seq_of_events IS 'Staging table for event sequences';
COMMENT ON TABLE staging.events_sequence IS 'Staging table for event ordering';
COMMENT ON TABLE staging.engines IS 'Staging table for engine details';
COMMENT ON TABLE staging.narratives IS 'Staging table for narratives';
COMMENT ON TABLE staging.ntsb_admin IS 'Staging table for NTSB administrative data';

-- ============================================
-- COMPLETION MESSAGE
-- ============================================

DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    function_count INTEGER;
BEGIN
    -- Count objects created
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'staging';

    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'staging';

    SELECT COUNT(*) INTO function_count
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'staging';

    RAISE NOTICE '';
    RAISE NOTICE '╔══════════════════════════════════════════════════════════╗';
    RAISE NOTICE '║     Staging Table Infrastructure Created Successfully    ║';
    RAISE NOTICE '╚══════════════════════════════════════════════════════════╝';
    RAISE NOTICE '';
    RAISE NOTICE 'Schema: staging';
    RAISE NOTICE '  - Tables: % tables created', table_count;
    RAISE NOTICE '  - Indexes: % indexes created', index_count;
    RAISE NOTICE '  - Functions: % helper functions', function_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Helper Functions:';
    RAISE NOTICE '  - staging.truncate_all_tables()       - Clear all staging tables';
    RAISE NOTICE '  - staging.get_row_counts()             - Show row counts';
    RAISE NOTICE '  - staging.identify_duplicate_events()  - Find duplicates';
    RAISE NOTICE '  - staging.get_duplicate_stats()        - Duplicate statistics';
    RAISE NOTICE '';
    RAISE NOTICE 'Usage Example:';
    RAISE NOTICE '  SELECT * FROM staging.get_row_counts();';
    RAISE NOTICE '  SELECT * FROM staging.get_duplicate_stats();';
    RAISE NOTICE '';
END $$;
