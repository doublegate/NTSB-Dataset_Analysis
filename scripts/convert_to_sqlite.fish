#!/usr/bin/env fish
# Convert MDB database to SQLite for easier querying

set database $argv[1]
set output_db $argv[2]

if test (count $argv) -lt 1
    echo "Usage: ./scripts/convert_to_sqlite.fish <database.mdb> [output.db]"
    echo ""
    echo "Example:"
    echo "  ./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db"
    exit 1
end

if not test -f "$database"
    echo "Error: Database file not found: $database"
    exit 1
end

# Set default output if not provided
if test -z "$output_db"
    set output_db (basename "$database" .mdb).db
    set output_db (basename "$output_db" .MDB).db
    set output_db "data/$output_db"
end

echo "🔄 Converting MDB to SQLite..."
echo "Source: $database"
echo "Output: $output_db"
echo ""

# Create output directory
mkdir -p (dirname "$output_db")

# Remove existing database
if test -f "$output_db"
    echo "⚠️  Removing existing database: $output_db"
    rm "$output_db"
end

# Get all tables
set tables (mdb-tables "$database")
set total (count $tables)
set current 0

echo "📊 Converting $total tables..."
echo ""

for table in $tables
    set current (math $current + 1)
    echo "[$current/$total] Converting $table..."

    # Export to CSV and import to SQLite
    mdb-export "$database" "$table" | sqlite3 "$output_db" ".mode csv" ".import '|cat -' $table" 2>/dev/null

    if test $status -ne 0
        # Fallback: export to temp CSV then import
        set temp_csv "/tmp/ntsb_temp_$table.csv"
        mdb-export "$database" "$table" > "$temp_csv"

        sqlite3 "$output_db" << EOF
.mode csv
.import $temp_csv $table
EOF
        rm "$temp_csv"
    end
end

echo ""
echo "✅ Conversion complete!"
echo "📁 SQLite database: $output_db"
echo ""
echo "💡 Query with:"
echo "   sqlite3 $output_db"
echo "   sqlite> SELECT * FROM events LIMIT 10;"
