#!/usr/bin/env fish
# Extract all tables from MDB to CSV

set database $argv[1]
if test -z "$database"
    echo "Usage: ./scripts/extract_all_tables.fish <path_to_mdb_file>"
    exit 1
end

if not test -f "$database"
    echo "Error: Database file not found: $database"
    exit 1
end

echo "📊 Extracting tables from $database..."

# Get database name for prefix (avall, Pre2008, or PRE1982)
set db_basename (basename "$database" .mdb)
set db_basename (basename "$db_basename" .MDB)

mkdir -p data

# Get all table names and split by whitespace
set tables (mdb-tables "$database" | string split ' ')

set count 0
for table in $tables
    # Skip empty entries
    if test -n "$table"
        echo "  → Exporting $table..."
        mdb-export "$database" "$table" > "data/$db_basename-$table.csv"
        set count (math $count + 1)
    end
end

echo "✅ Done! Exported $count tables to data/"
echo "📁 Sample of exported files:"
ls -lh data/ | head -10
