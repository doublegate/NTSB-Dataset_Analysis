#!/usr/bin/env fish
# Search for text in CSV files

set search_term $argv[1]
set column $argv[2]

if test -z "$search_term"
    echo "Usage: ./scripts/search_data.fish <search_term> [column_name]"
    echo ""
    echo "Examples:"
    echo "  # Search all columns for 'Boeing'"
    echo "  ./scripts/search_data.fish Boeing"
    echo ""
    echo "  # Search specific column"
    echo "  ./scripts/search_data.fish \"Los Angeles\" ev_city"
    echo ""
    echo "  # Search in specific CSV"
    echo "  ./scripts/search_data.fish Cessna | grep events.csv"
    exit 1
end

if not test -d data
    echo "❌ Error: data/ directory not found"
    echo "   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb"
    exit 1
end

echo "🔍 Searching for: '$search_term'"
if test -n "$column"
    echo "📋 In column: $column"
end
echo ""

set found 0

for csv_file in data/*.csv
    if test -f "$csv_file"
        set filename (basename "$csv_file")

        if test -n "$column"
            # Search specific column using csvgrep
            if command -v csvgrep > /dev/null 2>&1
                set results (csvgrep -c "$column" -m "$search_term" "$csv_file" 2>/dev/null | wc -l)
                if test $results -gt 1  # Greater than 1 because of header
                    set matches (math $results - 1)
                    echo "✓ $filename: $matches match(es) in column '$column'"
                    set found (math $found + $matches)
                end
            else
                echo "⚠️  csvgrep not available. Install with: pip install csvkit"
                break
            end
        else
            # Search all columns
            set results (grep -i "$search_term" "$csv_file" 2>/dev/null | wc -l)
            if test $results -gt 0
                echo "✓ $filename: $results match(es)"
                set found (math $found + $results)
            end
        end
    end
end

echo ""
if test $found -gt 0
    echo "✅ Total matches found: $found"
else
    echo "❌ No matches found"
end
