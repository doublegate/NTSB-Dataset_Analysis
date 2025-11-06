-- create_load_tracking.sql - Database Load Tracking System
-- Phase 1 Sprint 2: Historical Data Integration with Staging Tables
-- Version: 2.0.0
-- Date: 2025-11-06
--
-- Purpose: Track which historical databases have been loaded to prevent
--          accidental re-loading of static historical datasets (Pre2008, PRE1982)

-- Create load_tracking table
CREATE TABLE IF NOT EXISTS load_tracking (
    id SERIAL PRIMARY KEY,
    database_name VARCHAR(50) UNIQUE NOT NULL,
    database_path TEXT,
    load_status VARCHAR(20) CHECK (load_status IN ('pending', 'in_progress', 'completed', 'failed')) NOT NULL,

    -- Load statistics
    events_loaded INTEGER DEFAULT 0,
    total_rows_loaded INTEGER DEFAULT 0,
    duplicate_events_found INTEGER DEFAULT 0,
    new_events_added INTEGER DEFAULT 0,

    -- Timing information
    load_started_at TIMESTAMP,
    load_completed_at TIMESTAMP,
    load_duration_seconds INTEGER,

    -- Additional metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on status for fast lookups
CREATE INDEX IF NOT EXISTS idx_load_tracking_status ON load_tracking(load_status);
CREATE INDEX IF NOT EXISTS idx_load_tracking_db_name ON load_tracking(database_name);

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_load_tracking_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_load_tracking_timestamp
    BEFORE UPDATE ON load_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_load_tracking_timestamp();

-- Add comments for documentation
COMMENT ON TABLE load_tracking IS 'Tracks which historical NTSB databases have been loaded to prevent duplicate loading';
COMMENT ON COLUMN load_tracking.database_name IS 'Name of the MDB file (avall.mdb, Pre2008.mdb, PRE1982.MDB)';
COMMENT ON COLUMN load_tracking.load_status IS 'Current status: pending, in_progress, completed, failed';
COMMENT ON COLUMN load_tracking.duplicate_events_found IS 'Number of events that already existed in production database';
COMMENT ON COLUMN load_tracking.new_events_added IS 'Number of new unique events inserted into production database';
COMMENT ON COLUMN load_tracking.total_rows_loaded IS 'Total rows loaded across all tables (events + child tables)';

-- Insert initial tracking records for all three databases
INSERT INTO load_tracking (database_name, database_path, load_status, notes) VALUES
    ('avall.mdb', 'datasets/avall.mdb', 'completed', 'Initial load completed in Sprint 1'),
    ('Pre2008.mdb', 'datasets/Pre2008.mdb', 'completed', 'Historical load completed - contains 2000-2007 data'),
    ('PRE1982.MDB', 'datasets/PRE1982.MDB', 'pending', 'Different schema - requires investigation before loading')
ON CONFLICT (database_name) DO NOTHING;

-- Update avall.mdb status with Sprint 1 statistics
UPDATE load_tracking
SET
    load_status = 'completed',
    events_loaded = 29773,
    total_rows_loaded = 478631,
    duplicate_events_found = 0,
    new_events_added = 29773,
    load_completed_at = '2025-11-05 23:59:59',
    notes = 'Sprint 1 initial load - 2008-2025 data, 11 tables, 100% success rate'
WHERE database_name = 'avall.mdb';

-- Update Pre2008.mdb status (already loaded in Sprint 2, but without staging tables)
UPDATE load_tracking
SET
    load_status = 'completed',
    events_loaded = 62999,  -- Approximate from Sprint 2 Progress Report
    total_rows_loaded = 906176,  -- Total extracted rows
    duplicate_events_found = 0,  -- Will be recalculated with staging table approach
    new_events_added = 62999,  -- Estimated
    load_completed_at = CURRENT_TIMESTAMP,
    notes = 'Sprint 2 historical load - 2000-2007 data, loaded without staging table system. May need re-verification.'
WHERE database_name = 'Pre2008.mdb';

-- Display current tracking status
SELECT
    database_name,
    load_status,
    events_loaded,
    total_rows_loaded,
    duplicate_events_found,
    new_events_added,
    load_completed_at,
    notes
FROM load_tracking
ORDER BY
    CASE load_status
        WHEN 'completed' THEN 1
        WHEN 'in_progress' THEN 2
        WHEN 'failed' THEN 3
        WHEN 'pending' THEN 4
    END,
    load_completed_at DESC NULLS LAST;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✓ Load tracking system created successfully';
    RAISE NOTICE '  - Table: load_tracking';
    RAISE NOTICE '  - Indexes: 2 indexes created';
    RAISE NOTICE '  - Triggers: updated_at auto-update enabled';
    RAISE NOTICE '  - Records: 3 database entries initialized';
    RAISE NOTICE '';
    RAISE NOTICE 'Current status:';
    RAISE NOTICE '  - avall.mdb:    COMPLETED (Sprint 1)';
    RAISE NOTICE '  - Pre2008.mdb:  COMPLETED (Sprint 2, needs staging re-verification)';
    RAISE NOTICE '  - PRE1982.MDB:  PENDING (requires investigation)';
END $$;
