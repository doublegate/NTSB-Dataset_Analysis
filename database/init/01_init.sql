-- NTSB Aviation Accident Database - Initialization Script
-- This script runs automatically on first container startup
-- Creates extensions and prepares database for data loading

\echo 'Starting NTSB Aviation Database initialization...'

-- ============================================
-- Create Extensions
-- ============================================

-- PostGIS for geospatial queries (aircraft locations, crash sites)
CREATE EXTENSION IF NOT EXISTS postgis;
\echo 'PostGIS extension created'

-- pg_trgm for fuzzy text search (aircraft registration, NTSB numbers)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
\echo 'pg_trgm extension created'

-- pgcrypto for UUID generation and hashing
CREATE EXTENSION IF NOT EXISTS pgcrypto;
\echo 'pgcrypto extension created'

-- pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
\echo 'pg_stat_statements extension created'

-- ============================================
-- Verify Extensions
-- ============================================
SELECT
    extname AS extension_name,
    extversion AS version
FROM pg_extension
WHERE extname IN ('postgis', 'pg_trgm', 'pgcrypto', 'pg_stat_statements')
ORDER BY extname;

-- ============================================
-- Database Metadata
-- ============================================
COMMENT ON DATABASE ntsb_aviation IS
'NTSB Aviation Accident Database (1962-2025) - 179,809 events across 64 years.
Data source: National Transportation Safety Board (NTSB).
Schema: 13 tables, 59 indexes, 6 materialized views, PostGIS geospatial support.';

\echo 'Database initialization complete!'
\echo 'Ready for schema import (scripts/schema.sql) and data loading (scripts/load_with_staging.py)'
