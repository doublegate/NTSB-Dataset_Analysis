#!/usr/bin/env fish
# Analyze CSV file with statistics

set csv_file $argv[1]

if test -z "$csv_file"
    echo "Usage: ./scripts/analyze_csv.fish <csv_file>"
    echo ""
    echo "Example:"
    echo "  ./scripts/analyze_csv.fish data/events.csv"
    exit 1
end

if not test -f "$csv_file"
    echo "Error: File not found: $csv_file"
    exit 1
end

echo "📊 Analyzing: $csv_file"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# File info
set file_size (du -h "$csv_file" | cut -f1)
echo "📁 File size: $file_size"

# Row count
set total_lines (wc -l < "$csv_file")
set data_rows (math $total_lines - 1)
echo "📈 Rows: $data_rows (plus 1 header)"

# Column info
echo ""
echo "📋 Columns:"
head -n 1 "$csv_file" | tr ',' '\n' | nl -w2 -s'. '

# Use csvstat if available for detailed stats
if command -v csvstat > /dev/null 2>&1
    echo ""
    echo "📊 Statistical Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    csvstat "$csv_file" 2>/dev/null
else if command -v xsv > /dev/null 2>&1
    echo ""
    echo "📊 Statistical Summary (xsv):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    xsv stats "$csv_file" | xsv table
else
    echo ""
    echo "💡 Install csvkit or xsv for detailed statistics:"
    echo "   pip install csvkit"
    echo "   cargo install xsv"
end
