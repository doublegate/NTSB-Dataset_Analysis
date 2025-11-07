-- Migration: Add file_size and file_checksum columns to load_tracking
-- Date: 2025-11-07
-- Sprint: Phase 1 Sprint 3 Week 2
-- Reason: Required by monthly_sync_dag.py for update detection
-- Status: Already applied during testing (columns exist)

BEGIN;

-- Add file_size column (stores size in bytes)
ALTER TABLE load_tracking
ADD COLUMN IF NOT EXISTS file_size BIGINT;

-- Add file_checksum column (stores SHA256 hash)
ALTER TABLE load_tracking
ADD COLUMN IF NOT EXISTS file_checksum VARCHAR(64);

-- Add comments
COMMENT ON COLUMN load_tracking.file_size IS 'File size in bytes for update detection';
COMMENT ON COLUMN load_tracking.file_checksum IS 'SHA256 checksum for file integrity verification';

-- Verify columns added
SELECT
    column_name,
    data_type,
    character_maximum_length,
    column_default,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'load_tracking'
AND column_name IN ('file_size', 'file_checksum')
ORDER BY column_name;

COMMIT;

-- Expected Output:
-- column_name   | data_type         | character_maximum_length | column_default | is_nullable
-- --------------+-------------------+--------------------------+----------------+-------------
-- file_checksum | character varying | 64                       | NULL           | YES
-- file_size     | bigint            | NULL                     | NULL           | YES
