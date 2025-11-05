#!/usr/bin/env fish
# Show information about MDB database

set database $argv[1]

if test -z "$database"
    echo "Usage: ./scripts/show_database_info.fish <database.mdb>"
    echo ""
    echo "Example:"
    echo "  ./scripts/show_database_info.fish datasets/avall.mdb"
    exit 1
end

if not test -f "$database"
    echo "Error: Database file not found: $database"
    exit 1
end

echo "📊 Database Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "File: $database"
set file_size (du -h "$database" | cut -f1)
echo "Size: $file_size"
echo ""

echo "📋 Tables:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
set tables (mdb-tables "$database")
set table_count (count $tables)
echo "Total tables: $table_count"
echo ""

# List all tables
for table in $tables
    echo "  • $table"
end

echo ""
echo "💡 Tip: Extract a table with:"
echo "   ./scripts/extract_table.fish $database <table_name>"
