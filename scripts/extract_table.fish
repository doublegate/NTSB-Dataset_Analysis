#!/usr/bin/env fish
# Extract a specific table from MDB to CSV

set database $argv[1]
set table_name $argv[2]

if test (count $argv) -lt 2
    echo "Usage: ./scripts/extract_table.fish <database.mdb> <table_name>"
    echo ""
    echo "Example:"
    echo "  ./scripts/extract_table.fish datasets/avall.mdb events"
    exit 1
end

if not test -f "$database"
    echo "Error: Database file not found: $database"
    exit 1
end

# Get database name for prefix (avall, Pre2008, or PRE1982)
set db_basename (basename "$database" .mdb)
set db_basename (basename "$db_basename" .MDB)

# Check if table exists in database
# Split the space-separated table list into an array
set tables (mdb-tables "$database" | string split ' ')
if not contains "$table_name" $tables
    echo "Error: Table '$table_name' not found in database"
    echo ""
    echo "Available tables:"
    for t in $tables
        if test -n "$t"
            echo "  - $t"
        end
    end
    exit 1
end

echo "📊 Extracting table '$table_name' from $database..."

mkdir -p data

set output_file "data/$db_basename-$table_name.csv"
mdb-export "$database" "$table_name" > "$output_file"

if test $status -eq 0
    echo "✅ Success! Exported to: $output_file"
    set row_count (wc -l < "$output_file")
    echo "📈 Rows: "(math $row_count - 1)" (plus header)"
    set file_size (du -h "$output_file" | cut -f1)
    echo "💾 Size: $file_size"
else
    echo "❌ Error exporting table"
    exit 1
end
