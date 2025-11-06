#!/bin/bash
# setup_database.sh - Complete database setup for NTSB Aviation Database
# Run as regular user - only requires sudo for initial database creation and extensions
#
# Phase 1 Sprint 2: Simplified setup with minimal sudo requirements
# Version: 2.0.0
# Date: 2025-11-06
#
# Usage:
#   ./scripts/setup_database.sh
#   ./scripts/setup_database.sh custom_db_name custom_user

set -e  # Exit on error

# ============================================
# CONFIGURATION
# ============================================
DB_NAME="${1:-ntsb_aviation}"
DB_USER="${2:-$USER}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "NTSB Aviation Database - Complete Setup"
echo "============================================================"
echo "Database: $DB_NAME"
echo "Owner: $DB_USER"
echo ""

# ============================================
# 1. CHECK POSTGRESQL
# ============================================
echo -e "${GREEN}Step 1/8: Checking PostgreSQL${NC}"
echo "------------------------------------------------------------"

if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: PostgreSQL is not installed${NC}"
    echo ""
    echo "Install PostgreSQL:"
    echo "  Arch Linux:  sudo pacman -S postgresql postgis"
    echo "  Ubuntu:      sudo apt install postgresql postgresql-contrib postgis"
    echo "  macOS:       brew install postgresql postgis"
    exit 1
fi

PG_VERSION=$(psql --version | awk '{print $3}')
echo -e "${BLUE}✓ PostgreSQL ${PG_VERSION} installed${NC}"

if ! pg_isready -q 2>/dev/null; then
    echo -e "${YELLOW}Warning: PostgreSQL is not running${NC}"
    echo ""
    echo "Start PostgreSQL service:"
    echo "  sudo systemctl start postgresql"
    echo "  sudo systemctl enable postgresql  # Start on boot"
    echo ""
    read -p "Start PostgreSQL now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start postgresql
        sleep 2
        if ! pg_isready -q; then
            echo -e "${RED}Error: Could not start PostgreSQL${NC}"
            exit 1
        fi
    else
        echo "Please start PostgreSQL manually and re-run this script."
        exit 1
    fi
fi

echo -e "${GREEN}✓ PostgreSQL is running${NC}"
echo ""

# ============================================
# 2. CHECK/INITIALIZE POSTGRES DATA DIR
# ============================================
echo -e "${GREEN}Step 2/8: Checking PostgreSQL Initialization${NC}"
echo "------------------------------------------------------------"

if ! sudo -u postgres test -f /var/lib/postgres/data/PG_VERSION 2>/dev/null; then
    echo -e "${YELLOW}PostgreSQL data directory not initialized${NC}"
    echo "Initializing PostgreSQL..."
    sudo -u postgres initdb -D /var/lib/postgres/data
    sudo systemctl restart postgresql
    sleep 2
    echo -e "${GREEN}✓ PostgreSQL initialized${NC}"
else
    echo -e "${BLUE}✓ PostgreSQL already initialized${NC}"
fi

echo ""

# ============================================
# 3. CREATE DATABASE
# ============================================
echo -e "${GREEN}Step 3/8: Creating Database${NC}"
echo "------------------------------------------------------------"

if psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Database $DB_NAME already exists${NC}"
    read -p "Drop and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        dropdb "$DB_NAME" 2>/dev/null || sudo -u postgres dropdb "$DB_NAME"
        echo "✓ Dropped existing database"
    else
        echo -e "${BLUE}Using existing database${NC}"
    fi
fi

if ! psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME" 2>/dev/null; then
    echo "Creating database $DB_NAME..."

    # Try as current user first
    if createdb -O "$DB_USER" "$DB_NAME" 2>/dev/null; then
        echo -e "${GREEN}✓ Database created by $DB_USER${NC}"
    else
        # Fall back to postgres user if needed
        echo "Requires postgres user privileges..."
        sudo -u postgres createdb -O "$DB_USER" "$DB_NAME"
        echo -e "${GREEN}✓ Database created by postgres user${NC}"
    fi
else
    echo -e "${BLUE}✓ Database already exists${NC}"
fi

echo ""

# ============================================
# 4. ENABLE EXTENSIONS
# ============================================
echo -e "${GREEN}Step 4/8: Enabling PostgreSQL Extensions${NC}"
echo "------------------------------------------------------------"

echo "Enabling PostGIS, pg_trgm, pgcrypto, pg_stat_statements..."
echo "(Requires superuser privileges - this is the ONLY superuser step)"

sudo -u postgres psql -d "$DB_NAME" <<EOF
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
EOF

echo -e "${GREEN}✓ Extensions enabled${NC}"
echo ""

# ============================================
# 5. TRANSFER OWNERSHIP
# ============================================
echo -e "${GREEN}Step 5/8: Transferring Ownership to $DB_USER${NC}"
echo "------------------------------------------------------------"

if [ ! -f "$SCRIPT_DIR/transfer_ownership.sql" ]; then
    echo -e "${RED}Error: transfer_ownership.sql not found${NC}"
    echo "Expected at: $SCRIPT_DIR/transfer_ownership.sql"
    exit 1
fi

sudo -u postgres psql -d "$DB_NAME" -f "$SCRIPT_DIR/transfer_ownership.sql" | grep -E "(✓|Transferred|owner)"

echo ""

# ============================================
# 6. CREATE SCHEMA
# ============================================
echo -e "${GREEN}Step 6/8: Creating Database Schema${NC}"
echo "------------------------------------------------------------"

if [ ! -f "$SCRIPT_DIR/schema.sql" ]; then
    echo -e "${RED}Error: schema.sql not found${NC}"
    echo "Expected at: $SCRIPT_DIR/schema.sql"
    exit 1
fi

psql -d "$DB_NAME" -f "$SCRIPT_DIR/schema.sql" | grep -E "(CREATE|✓|NOTICE)"

echo -e "${GREEN}✓ Schema created (11 core tables)${NC}"
echo ""

# ============================================
# 7. CREATE STAGING INFRASTRUCTURE
# ============================================
echo -e "${GREEN}Step 7/8: Creating Staging Infrastructure${NC}"
echo "------------------------------------------------------------"

if [ ! -f "$SCRIPT_DIR/create_staging_tables.sql" ]; then
    echo -e "${YELLOW}Warning: create_staging_tables.sql not found${NC}"
    echo "Skipping staging table creation"
else
    psql -d "$DB_NAME" -f "$SCRIPT_DIR/create_staging_tables.sql" | grep -E "(CREATE|✓)"
    echo -e "${GREEN}✓ Staging tables created${NC}"
fi

echo ""

# ============================================
# 8. CREATE LOAD TRACKING
# ============================================
echo -e "${GREEN}Step 8/8: Creating Load Tracking System${NC}"
echo "------------------------------------------------------------"

if [ ! -f "$SCRIPT_DIR/create_load_tracking.sql" ]; then
    echo -e "${YELLOW}Warning: create_load_tracking.sql not found${NC}"
    echo "Skipping load tracking creation"
else
    psql -d "$DB_NAME" -f "$SCRIPT_DIR/create_load_tracking.sql" | grep -E "(CREATE|INSERT|✓)"
    echo -e "${GREEN}✓ Load tracking system created${NC}"
fi

echo ""

# ============================================
# COMPLETION SUMMARY
# ============================================
echo "============================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Database: $DB_NAME"
echo "Owner: $DB_USER"
echo ""

echo "Database Objects:"
psql -d "$DB_NAME" -t -c "
SELECT
    'Tables: ' || COUNT(*)::text
FROM pg_tables
WHERE schemaname = 'public'
UNION ALL
SELECT
    'Sequences: ' || COUNT(*)::text
FROM pg_sequences
WHERE schemaname = 'public'
UNION ALL
SELECT
    'Functions: ' || COUNT(*)::text
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public' AND p.prokind = 'f'
UNION ALL
SELECT
    'Extensions: ' || COUNT(*)::text
FROM pg_extension
WHERE extname IN ('postgis', 'pg_trgm', 'pgcrypto', 'pg_stat_statements');
" | sed 's/^[ \t]*/  /'

echo ""
echo "Database Size:"
psql -d "$DB_NAME" -t -c "SELECT '  ' || pg_size_pretty(pg_database_size('$DB_NAME'));"

echo ""
echo "============================================================"
echo "Next Steps:"
echo "============================================================"
echo ""
echo "1. Activate Python environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Load Current Data (2008-present):"
echo "   python scripts/load_with_staging.py --source avall.mdb"
echo ""
echo "3. Load Historical Data (2000-2007):"
echo "   python scripts/load_with_staging.py --source Pre2008.mdb"
echo ""
echo "4. Optimize Database:"
echo "   psql -d $DB_NAME -f scripts/optimize_queries.sql"
echo ""
echo "5. Validate Data Quality:"
echo "   psql -d $DB_NAME -f scripts/validate_data.sql"
echo ""
echo "6. Performance Testing (optional):"
echo "   psql -d $DB_NAME -f scripts/test_performance.sql"
echo ""
echo "============================================================"
echo -e "${GREEN}All future operations run as regular user (no sudo!)${NC}"
echo "============================================================"
echo ""
