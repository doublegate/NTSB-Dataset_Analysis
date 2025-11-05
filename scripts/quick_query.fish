#!/usr/bin/env fish
# Quick SQL query on CSV data using DuckDB

set query $argv[1]

if test -z "$query"
    echo "Usage: ./scripts/quick_query.fish <SQL_query>"
    echo ""
    echo "Examples:"
    echo "  ./scripts/quick_query.fish \"SELECT COUNT(*) FROM 'data/events.csv'\""
    echo "  ./scripts/quick_query.fish \"SELECT ev_state, COUNT(*) as count FROM 'data/events.csv' GROUP BY ev_state ORDER BY count DESC LIMIT 10\""
    echo ""
    echo "Common queries:"
    echo "  # Total events"
    echo "  ./scripts/quick_query.fish \"SELECT COUNT(*) as total_events FROM 'data/events.csv'\""
    echo ""
    echo "  # Events by year"
    echo "  ./scripts/quick_query.fish \"SELECT ev_year, COUNT(*) as count FROM 'data/events.csv' GROUP BY ev_year ORDER BY ev_year DESC\""
    echo ""
    echo "  # Fatal accidents in 2023"
    echo "  ./scripts/quick_query.fish \"SELECT * FROM 'data/events.csv' WHERE ev_year = 2023 AND inj_tot_f > 0\""
    exit 1
end

# Check if DuckDB is installed
if not command -v duckdb > /dev/null 2>&1
    echo "❌ Error: DuckDB is not installed"
    echo ""
    echo "Install with:"
    echo "  sudo pacman -S duckdb"
    exit 1
end

# Check if data directory exists
if not test -d data
    echo "⚠️  Warning: data/ directory not found"
    echo "   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb"
    echo ""
end

echo "🔍 Executing query..."
echo ""

duckdb -c "$query"
