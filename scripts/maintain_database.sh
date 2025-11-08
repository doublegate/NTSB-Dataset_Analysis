#!/bin/bash
# maintain_database.sh - NTSB Aviation Database Maintenance Wrapper Script
# Version: 1.0.0
# Date: 2025-11-07
#
# Usage:
#   ./scripts/maintain_database.sh [db_name]
#
# Description:
#   Runs comprehensive database maintenance with logging
#   - VACUUM ANALYZE (storage optimization)
#   - Refresh materialized views
#   - Update statistics
#   - Data quality validation
#   - Index health check
#   - Performance metrics collection
#
# Recommended Frequency: Monthly (after data loads)

set -e  # Exit on error

# Default database name
DB_NAME="${1:-ntsb_aviation}"

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log directory and file
LOG_DIR="logs/maintenance"
LOG_FILE="$LOG_DIR/maintenance_$TIMESTAMP.log"

# Create logs directory if not exists
mkdir -p "$LOG_DIR"

echo "========================================="
echo "NTSB Aviation Database Maintenance"
echo "========================================="
echo "Database: $DB_NAME"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo "========================================="
echo ""

# Check if database exists
if ! psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "ERROR: Database '$DB_NAME' does not exist."
    exit 1
fi

# Run maintenance script and log output
echo "Running maintenance script..."
echo "(This may take several minutes depending on database size)"
echo ""

if psql -d "$DB_NAME" -f scripts/maintain_database.sql 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "========================================="
    echo "Maintenance completed successfully!"
    echo "========================================="
    echo "Log saved to: $LOG_FILE"
    echo ""

    # Display log file size
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo "Log file size: $LOG_SIZE"

    # Extract key metrics from log
    echo ""
    echo "Key Metrics Summary:"
    echo "-------------------"

    # Database size
    grep "database_size" "$LOG_FILE" | head -1 | tail -1

    # Total events
    grep "events" "$LOG_FILE" | grep "row_count" | head -1

    echo ""
    echo "For full details, see: $LOG_FILE"

    exit 0
else
    echo ""
    echo "ERROR: Maintenance script failed!"
    echo "See log file for details: $LOG_FILE"
    exit 1
fi
