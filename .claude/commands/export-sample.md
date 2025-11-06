# Export Sample - Sample Data Export for Testing

Export filtered sample datasets from the NTSB Aviation Database for testing, sharing, demos, and development purposes with comprehensive relationship preservation.

---

## OBJECTIVE

Generate sample datasets that:
- Filter by multiple criteria (year range, state, aircraft type, injury severity)
- Preserve relational integrity (events + related tables)
- Export to CSV format (one file per table)
- Create metadata file documenting filters and contents
- Package as compressed archive
- Suitable for testing, demos, documentation, and sharing

**Time Estimate:** 2-5 minutes (depending on sample size)
**Output:** ZIP archive with CSV files + metadata

---

## CONTEXT

**Project:** NTSB Aviation Database (PostgreSQL data repository)
**Repository:** /home/parobek/Code/NTSB_Datasets
**Database:** ntsb_aviation
**Use Cases:** Testing, sharing, demos, documentation, development

**Sample Sizes:**
- **tiny** - 100 events (~2,000 rows total, <1MB)
- **small** - 1,000 events (~20,000 rows total, ~5MB)
- **medium** - 10,000 events (~200,000 rows total, ~50MB)
- **custom** - User-specified count

---

## USAGE

```bash
/export-sample                    # Interactive mode (prompts for filters)
/export-sample tiny               # Export 100 events
/export-sample small              # Export 1,000 events
/export-sample medium             # Export 10,000 events
/export-sample --count 500        # Export exactly 500 events
/export-sample --year 2020-2023   # Filter by year range
/export-sample --state CA,TX,FL   # Filter by states
/export-sample --fatal            # Only fatal accidents
/export-sample --recent           # Last 5 years only
```

---

## EXECUTION PHASES

### PHASE 1: ENVIRONMENT CHECKS (1 minute)

**Objective:** Verify prerequisites and prepare environment

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 SAMPLE DATA EXPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check PostgreSQL connection
echo "Checking database connection..."
if ! psql -d ntsb_aviation -c "SELECT 1;" &> /dev/null; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    exit 1
fi
echo "✅ Database connection verified"

# Create export directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPORT_DIR="/tmp/NTSB_Datasets/exports/sample_${TIMESTAMP}"
mkdir -p "$EXPORT_DIR"
echo "✅ Export directory: $EXPORT_DIR"
echo ""
```

---

### PHASE 2: PARSE FILTERS (1 minute)

**Objective:** Determine sample size and filter criteria

```bash
# Default values
SAMPLE_SIZE=""
YEAR_MIN=""
YEAR_MAX=""
STATES=""
FATAL_ONLY=""
RECENT_ONLY=""
AIRCRAFT_TYPE=""
CUSTOM_COUNT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        tiny)
            SAMPLE_SIZE="tiny"
            CUSTOM_COUNT=100
            ;;
        small)
            SAMPLE_SIZE="small"
            CUSTOM_COUNT=1000
            ;;
        medium)
            SAMPLE_SIZE="medium"
            CUSTOM_COUNT=10000
            ;;
        --count)
            CUSTOM_COUNT="$2"
            shift
            ;;
        --year)
            # Format: 2020-2023 or 2020
            if [[ "$2" =~ ^([0-9]{4})-([0-9]{4})$ ]]; then
                YEAR_MIN="${BASH_REMATCH[1]}"
                YEAR_MAX="${BASH_REMATCH[2]}"
            else
                YEAR_MIN="$2"
                YEAR_MAX="$2"
            fi
            shift
            ;;
        --state)
            STATES="$2"
            shift
            ;;
        --fatal)
            FATAL_ONLY="true"
            ;;
        --recent)
            RECENT_ONLY="true"
            YEAR_MIN=$(($(date +%Y) - 5))
            YEAR_MAX=$(date +%Y)
            ;;
        --aircraft)
            AIRCRAFT_TYPE="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Interactive mode if no arguments
if [ -z "$SAMPLE_SIZE" ] && [ -z "$CUSTOM_COUNT" ]; then
    echo "Select sample size:"
    echo "  1) Tiny   (100 events, <1MB)"
    echo "  2) Small  (1,000 events, ~5MB)"
    echo "  3) Medium (10,000 events, ~50MB)"
    echo "  4) Custom (specify count)"
    echo ""
    read -p "Choice (1-4): " choice
    
    case $choice in
        1) SAMPLE_SIZE="tiny"; CUSTOM_COUNT=100 ;;
        2) SAMPLE_SIZE="small"; CUSTOM_COUNT=1000 ;;
        3) SAMPLE_SIZE="medium"; CUSTOM_COUNT=10000 ;;
        4) 
            read -p "Enter event count: " CUSTOM_COUNT
            SAMPLE_SIZE="custom"
            ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
    
    # Optional filters
    echo ""
    read -p "Filter by year range? (e.g., 2020-2023 or press Enter to skip): " year_filter
    if [ -n "$year_filter" ]; then
        if [[ "$year_filter" =~ ^([0-9]{4})-([0-9]{4})$ ]]; then
            YEAR_MIN="${BASH_REMATCH[1]}"
            YEAR_MAX="${BASH_REMATCH[2]}"
        else
            YEAR_MIN="$year_filter"
            YEAR_MAX="$year_filter"
        fi
    fi
    
    read -p "Filter by states? (e.g., CA,TX,FL or press Enter to skip): " STATES
    read -p "Fatal accidents only? (yes/no): " fatal_choice
    if [ "$fatal_choice" = "yes" ]; then
        FATAL_ONLY="true"
    fi
fi

echo "Export configuration:"
echo "  Sample size: ${SAMPLE_SIZE:-custom} ($CUSTOM_COUNT events)"
[ -n "$YEAR_MIN" ] && echo "  Year range: $YEAR_MIN - $YEAR_MAX"
[ -n "$STATES" ] && echo "  States: $STATES"
[ "$FATAL_ONLY" = "true" ] && echo "  Fatal only: Yes"
[ -n "$AIRCRAFT_TYPE" ] && echo "  Aircraft type: $AIRCRAFT_TYPE"
echo ""
```

---

### PHASE 3: BUILD FILTER QUERY (1 minute)

**Objective:** Construct SQL WHERE clause from filters

```bash
echo "Building filter query..."

# Start WHERE clause
WHERE_CLAUSE="WHERE 1=1"

# Year filter
if [ -n "$YEAR_MIN" ]; then
    if [ "$YEAR_MIN" = "$YEAR_MAX" ]; then
        WHERE_CLAUSE="$WHERE_CLAUSE AND EXTRACT(YEAR FROM ev_date) = $YEAR_MIN"
    else
        WHERE_CLAUSE="$WHERE_CLAUSE AND EXTRACT(YEAR FROM ev_date) BETWEEN $YEAR_MIN AND $YEAR_MAX"
    fi
fi

# State filter (comma-separated)
if [ -n "$STATES" ]; then
    # Convert CA,TX,FL to ('CA','TX','FL')
    STATE_LIST=$(echo "$STATES" | sed "s/,/','/g" | sed "s/^/('/" | sed "s/$/')/" )
    WHERE_CLAUSE="$WHERE_CLAUSE AND ev_state IN $STATE_LIST"
fi

# Fatal accidents only
if [ "$FATAL_ONLY" = "true" ]; then
    WHERE_CLAUSE="$WHERE_CLAUSE AND inj_tot_f > 0"
fi

# Aircraft type filter (for later join)
AIRCRAFT_WHERE=""
if [ -n "$AIRCRAFT_TYPE" ]; then
    AIRCRAFT_WHERE="AND (acft_make ILIKE '%$AIRCRAFT_TYPE%' OR acft_model ILIKE '%$AIRCRAFT_TYPE%')"
fi

echo "✅ Filter query constructed"
echo ""
```

---

### PHASE 4: EXTRACT SAMPLE EVENTS (1 minute)

**Objective:** Select sample events matching filters

```bash
echo "Extracting sample events..."

# Create temporary table with sample event IDs
psql -d ntsb_aviation << EOF
-- Create temporary table for sample events
DROP TABLE IF EXISTS temp_sample_events;
CREATE TEMP TABLE temp_sample_events AS
SELECT ev_id
FROM events
$WHERE_CLAUSE
ORDER BY ev_date DESC
LIMIT $CUSTOM_COUNT;

-- Show sample statistics
SELECT 
    COUNT(*) as sample_events,
    MIN(EXTRACT(YEAR FROM ev_date)) as min_year,
    MAX(EXTRACT(YEAR FROM ev_date)) as max_year,
    COUNT(DISTINCT ev_state) as states_count,
    SUM(inj_tot_f) as total_fatalities
FROM events
WHERE ev_id IN (SELECT ev_id FROM temp_sample_events);
EOF

# Capture sample count
ACTUAL_COUNT=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM temp_sample_events;" | xargs)

if [ "$ACTUAL_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No events match the specified filters"
    exit 1
fi

if [ "$ACTUAL_COUNT" -lt "$CUSTOM_COUNT" ]; then
    echo "⚠️  WARNING: Only $ACTUAL_COUNT events match filters (requested $CUSTOM_COUNT)"
fi

echo "✅ Sample events selected: $ACTUAL_COUNT"
echo ""
```

---

### PHASE 5: EXPORT EVENTS TABLE (1 minute)

**Objective:** Export events table for sample

```bash
echo "Exporting events table..."

psql -d ntsb_aviation -c "\\copy (
    SELECT * FROM events 
    WHERE ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY ev_date DESC
) TO '$EXPORT_DIR/events.csv' WITH CSV HEADER"

EVENT_COUNT=$(wc -l < "$EXPORT_DIR/events.csv")
EVENT_COUNT=$((EVENT_COUNT - 1))  # Subtract header
EVENT_SIZE=$(du -h "$EXPORT_DIR/events.csv" | cut -f1)

echo "✅ Events exported: $EVENT_COUNT rows ($EVENT_SIZE)"
```

---

### PHASE 6: EXPORT RELATED TABLES (2 minutes)

**Objective:** Export all related tables maintaining referential integrity

```bash
echo "Exporting related tables..."

# Aircraft
echo "  Exporting aircraft..."
psql -d ntsb_aviation -c "\\copy (
    SELECT a.* FROM aircraft a
    WHERE a.ev_id IN (SELECT ev_id FROM temp_sample_events)
    $AIRCRAFT_WHERE
    ORDER BY a.ev_id, a.aircraft_key
) TO '$EXPORT_DIR/aircraft.csv' WITH CSV HEADER"
AIRCRAFT_COUNT=$(tail -n +2 "$EXPORT_DIR/aircraft.csv" | wc -l)

# Flight Crew
echo "  Exporting flight_crew..."
psql -d ntsb_aviation -c "\\copy (
    SELECT fc.* FROM flight_crew fc
    WHERE fc.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY fc.ev_id, fc.crew_no
) TO '$EXPORT_DIR/flight_crew.csv' WITH CSV HEADER"
CREW_COUNT=$(tail -n +2 "$EXPORT_DIR/flight_crew.csv" | wc -l)

# Injury
echo "  Exporting injury..."
psql -d ntsb_aviation -c "\\copy (
    SELECT i.* FROM injury i
    WHERE i.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY i.ev_id
) TO '$EXPORT_DIR/injury.csv' WITH CSV HEADER"
INJURY_COUNT=$(tail -n +2 "$EXPORT_DIR/injury.csv" | wc -l)

# Findings
echo "  Exporting findings..."
psql -d ntsb_aviation -c "\\copy (
    SELECT f.* FROM findings f
    WHERE f.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY f.ev_id, f.finding_no
) TO '$EXPORT_DIR/findings.csv' WITH CSV HEADER"
FINDINGS_COUNT=$(tail -n +2 "$EXPORT_DIR/findings.csv" | wc -l)

# Narratives
echo "  Exporting narratives..."
psql -d ntsb_aviation -c "\\copy (
    SELECT n.* FROM narratives n
    WHERE n.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY n.ev_id
) TO '$EXPORT_DIR/narratives.csv' WITH CSV HEADER"
NARRATIVES_COUNT=$(tail -n +2 "$EXPORT_DIR/narratives.csv" | wc -l)

# Engines
echo "  Exporting engines..."
psql -d ntsb_aviation -c "\\copy (
    SELECT e.* FROM engines e
    WHERE e.aircraft_key IN (
        SELECT aircraft_key FROM aircraft 
        WHERE ev_id IN (SELECT ev_id FROM temp_sample_events)
    )
    ORDER BY e.aircraft_key, e.eng_no
) TO '$EXPORT_DIR/engines.csv' WITH CSV HEADER"
ENGINES_COUNT=$(tail -n +2 "$EXPORT_DIR/engines.csv" | wc -l)

# Events Sequence
echo "  Exporting events_sequence..."
psql -d ntsb_aviation -c "\\copy (
    SELECT es.* FROM events_sequence es
    WHERE es.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY es.ev_id, es.occurrence_no
) TO '$EXPORT_DIR/events_sequence.csv' WITH CSV HEADER"
SEQUENCE_COUNT=$(tail -n +2 "$EXPORT_DIR/events_sequence.csv" | wc -l)

# NTSB Admin
echo "  Exporting ntsb_admin..."
psql -d ntsb_aviation -c "\\copy (
    SELECT na.* FROM ntsb_admin na
    WHERE na.ev_id IN (SELECT ev_id FROM temp_sample_events)
    ORDER BY na.ev_id
) TO '$EXPORT_DIR/ntsb_admin.csv' WITH CSV HEADER"
ADMIN_COUNT=$(tail -n +2 "$EXPORT_DIR/ntsb_admin.csv" | wc -l)

# Calculate total rows
TOTAL_ROWS=$((EVENT_COUNT + AIRCRAFT_COUNT + CREW_COUNT + INJURY_COUNT + FINDINGS_COUNT + NARRATIVES_COUNT + ENGINES_COUNT + SEQUENCE_COUNT + ADMIN_COUNT))

echo ""
echo "✅ Related tables exported:"
echo "   events: $EVENT_COUNT"
echo "   aircraft: $AIRCRAFT_COUNT"
echo "   flight_crew: $CREW_COUNT"
echo "   injury: $INJURY_COUNT"
echo "   findings: $FINDINGS_COUNT"
echo "   narratives: $NARRATIVES_COUNT"
echo "   engines: $ENGINES_COUNT"
echo "   events_sequence: $SEQUENCE_COUNT"
echo "   ntsb_admin: $ADMIN_COUNT"
echo "   TOTAL: $TOTAL_ROWS rows"
echo ""
```

---

### PHASE 7: GENERATE METADATA (1 minute)

**Objective:** Create comprehensive metadata file documenting export

```bash
echo "Generating metadata..."

cat > "$EXPORT_DIR/METADATA.md" << EOF
# NTSB Aviation Database - Sample Export

**Generated:** $(date +"%Y-%m-%d %H:%M:%S")
**Database:** ntsb_aviation
**Sample Type:** ${SAMPLE_SIZE:-custom}
**Export Directory:** $EXPORT_DIR

---

## Export Configuration

**Sample Size:** $ACTUAL_COUNT events (requested: $CUSTOM_COUNT)

### Filters Applied

EOF

if [ -n "$YEAR_MIN" ]; then
    echo "- **Year Range:** $YEAR_MIN - $YEAR_MAX" >> "$EXPORT_DIR/METADATA.md"
else
    echo "- **Year Range:** All years" >> "$EXPORT_DIR/METADATA.md"
fi

if [ -n "$STATES" ]; then
    echo "- **States:** $STATES" >> "$EXPORT_DIR/METADATA.md"
else
    echo "- **States:** All states" >> "$EXPORT_DIR/METADATA.md"
fi

if [ "$FATAL_ONLY" = "true" ]; then
    echo "- **Fatalities:** Fatal accidents only" >> "$EXPORT_DIR/METADATA.md"
else
    echo "- **Fatalities:** All accident types" >> "$EXPORT_DIR/METADATA.md"
fi

if [ -n "$AIRCRAFT_TYPE" ]; then
    echo "- **Aircraft Type:** $AIRCRAFT_TYPE" >> "$EXPORT_DIR/METADATA.md"
else
    echo "- **Aircraft Type:** All types" >> "$EXPORT_DIR/METADATA.md"
fi

cat >> "$EXPORT_DIR/METADATA.md" << EOF

---

## Export Statistics

| Table | Rows | File Size |
|-------|------|-----------|
| events | $EVENT_COUNT | $(du -h "$EXPORT_DIR/events.csv" | cut -f1) |
| aircraft | $AIRCRAFT_COUNT | $(du -h "$EXPORT_DIR/aircraft.csv" | cut -f1) |
| flight_crew | $CREW_COUNT | $(du -h "$EXPORT_DIR/flight_crew.csv" | cut -f1) |
| injury | $INJURY_COUNT | $(du -h "$EXPORT_DIR/injury.csv" | cut -f1) |
| findings | $FINDINGS_COUNT | $(du -h "$EXPORT_DIR/findings.csv" | cut -f1) |
| narratives | $NARRATIVES_COUNT | $(du -h "$EXPORT_DIR/narratives.csv" | cut -f1) |
| engines | $ENGINES_COUNT | $(du -h "$EXPORT_DIR/engines.csv" | cut -f1) |
| events_sequence | $SEQUENCE_COUNT | $(du -h "$EXPORT_DIR/events_sequence.csv" | cut -f1) |
| ntsb_admin | $ADMIN_COUNT | $(du -h "$EXPORT_DIR/ntsb_admin.csv" | cut -f1) |
| **TOTAL** | **$TOTAL_ROWS** | **$(du -sh "$EXPORT_DIR" | cut -f1)** |

---

## Sample Characteristics

$(psql -d ntsb_aviation -t -c "
SELECT 
    'Year Range: ' || MIN(EXTRACT(YEAR FROM ev_date)) || ' - ' || MAX(EXTRACT(YEAR FROM ev_date)) ||
    E'\nStates Represented: ' || COUNT(DISTINCT ev_state) ||
    E'\nTotal Fatalities: ' || COALESCE(SUM(inj_tot_f), 0) ||
    E'\nTotal Injuries: ' || COALESCE(SUM(inj_tot_t), 0) ||
    E'\nAverage Fatalities per Event: ' || ROUND(AVG(COALESCE(inj_tot_f, 0)), 2) ||
    E'\nFatal Accidents: ' || COUNT(CASE WHEN inj_tot_f > 0 THEN 1 END) || ' (' || 
        ROUND(COUNT(CASE WHEN inj_tot_f > 0 THEN 1 END) * 100.0 / COUNT(*), 1) || '%)'
FROM events
WHERE ev_id IN (SELECT ev_id FROM temp_sample_events);
")

---

## File Descriptions

### events.csv
Master table containing accident event details. Primary key: \`ev_id\`

**Key Fields:**
- \`ev_id\` - Unique event identifier
- \`ev_date\` - Accident date
- \`ev_state\` - State where accident occurred
- \`latitude\`, \`longitude\` - Accident location
- \`inj_tot_f\` - Total fatalities
- \`inj_tot_t\` - Total injuries
- \`ev_type\` - Event type (ACC, INC)

### aircraft.csv
Aircraft involved in accidents. Foreign key: \`ev_id\` → events

**Key Fields:**
- \`aircraft_key\` - Unique aircraft identifier
- \`ev_id\` - Links to events table
- \`acft_make\` - Aircraft manufacturer
- \`acft_model\` - Aircraft model
- \`acft_category\` - Aircraft category (AIR, HELI, etc.)
- \`damage\` - Damage severity (DEST, SUBS, MINR, NONE)

### flight_crew.csv
Flight crew information. Foreign key: \`ev_id\` → events

### injury.csv
Injury details for crew and passengers. Foreign key: \`ev_id\` → events

### findings.csv
Investigation findings and probable causes. Foreign key: \`ev_id\` → events

**Key Fields:**
- \`finding_code\` - NTSB finding code
- \`cm_inPC\` - TRUE if in probable cause, FALSE if contributing factor

### narratives.csv
Textual accident narratives. Foreign key: \`ev_id\` → events

### engines.csv
Engine details. Foreign key: \`aircraft_key\` → aircraft

### events_sequence.csv
Event sequencing information. Foreign key: \`ev_id\` → events

### ntsb_admin.csv
Administrative metadata. Foreign key: \`ev_id\` → events

---

## Usage Examples

### Load into PostgreSQL

\`\`\`sql
CREATE TABLE events_sample (LIKE events INCLUDING ALL);
\\copy events_sample FROM 'events.csv' WITH CSV HEADER;

CREATE TABLE aircraft_sample (LIKE aircraft INCLUDING ALL);
\\copy aircraft_sample FROM 'aircraft.csv' WITH CSV HEADER;

-- Repeat for other tables...
\`\`\`

### Load into Python (pandas)

\`\`\`python
import pandas as pd

events = pd.read_csv('events.csv')
aircraft = pd.read_csv('aircraft.csv')
findings = pd.read_csv('findings.csv')

# Merge for analysis
merged = events.merge(aircraft, on='ev_id', how='left')
\`\`\`

### Load into R

\`\`\`r
events <- read.csv('events.csv')
aircraft <- read.csv('aircraft.csv')

# Merge
merged <- merge(events, aircraft, by='ev_id', all.x=TRUE)
\`\`\`

---

## Referential Integrity

All exported data maintains referential integrity:
- All \`ev_id\` values in child tables exist in events.csv
- All \`aircraft_key\` values in engines.csv exist in aircraft.csv
- No orphaned records

---

## Data Quality

- **NULL Handling:** NULL values exported as empty strings in CSV
- **Date Format:** ISO 8601 (YYYY-MM-DD)
- **Encoding:** UTF-8
- **Line Endings:** Unix (LF)
- **Delimiter:** Comma (,)
- **Quoting:** RFC 4180 compliant

---

## License & Attribution

**Source:** National Transportation Safety Board (NTSB)
**Database:** NTSB Aviation Accident Database
**URL:** https://data.ntsb.gov/avdata

**Usage:**
- Free to use for research, analysis, and educational purposes
- Attribution to NTSB required
- Not official NTSB data (sample/filtered subset)

---

## Limitations

- **Sample Only:** This is a filtered subset, not the complete database
- **Point-in-Time:** Data as of export date (may not include most recent updates)
- **Filters Applied:** See "Filters Applied" section above
- **Schema Simplified:** Some fields may be omitted for brevity

---

## Contact & Support

**Project:** NTSB Aviation Database
**Repository:** github.com/[username]/NTSB_Datasets
**Issues:** Report data quality issues or questions via GitHub Issues

---

**Generated by:** /export-sample command
**Export Version:** 1.0
EOF

echo "✅ Metadata generated: $EXPORT_DIR/METADATA.md"
echo ""
```

---

### PHASE 8: CREATE COMPRESSED ARCHIVE (1 minute)

**Objective:** Package all CSV files and metadata into ZIP archive

```bash
echo "Creating compressed archive..."

# Create ZIP archive
ARCHIVE_NAME="ntsb_sample_${SAMPLE_SIZE:-custom}_${ACTUAL_COUNT}events_${TIMESTAMP}.zip"
ARCHIVE_PATH="/tmp/NTSB_Datasets/exports/$ARCHIVE_NAME"

cd "$EXPORT_DIR/.."
zip -q -r "$ARCHIVE_PATH" "$(basename $EXPORT_DIR)"
cd - > /dev/null

ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo "✅ Archive created: $ARCHIVE_PATH ($ARCHIVE_SIZE)"
echo ""
```

---

### PHASE 9: GENERATE SUMMARY (1 minute)

**Objective:** Display comprehensive summary

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SAMPLE EXPORT COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 EXPORT SUMMARY"
echo "   Sample type: ${SAMPLE_SIZE:-custom}"
echo "   Events exported: $ACTUAL_COUNT"
echo "   Total rows: $TOTAL_ROWS"
echo "   Tables exported: 9"
echo ""
echo "📝 ARTIFACTS"
echo "   Export directory: $EXPORT_DIR"
echo "   Archive file: $ARCHIVE_PATH"
echo "   Archive size: $ARCHIVE_SIZE"
echo ""
echo "📂 FILES EXPORTED"
echo "   events.csv ($EVENT_COUNT rows)"
echo "   aircraft.csv ($AIRCRAFT_COUNT rows)"
echo "   flight_crew.csv ($CREW_COUNT rows)"
echo "   injury.csv ($INJURY_COUNT rows)"
echo "   findings.csv ($FINDINGS_COUNT rows)"
echo "   narratives.csv ($NARRATIVES_COUNT rows)"
echo "   engines.csv ($ENGINES_COUNT rows)"
echo "   events_sequence.csv ($SEQUENCE_COUNT rows)"
echo "   ntsb_admin.csv ($ADMIN_COUNT rows)"
echo "   METADATA.md (documentation)"
echo ""
if [ -n "$YEAR_MIN" ] || [ -n "$STATES" ] || [ "$FATAL_ONLY" = "true" ]; then
echo "🔍 FILTERS APPLIED"
[ -n "$YEAR_MIN" ] && echo "   Year range: $YEAR_MIN - $YEAR_MAX"
[ -n "$STATES" ] && echo "   States: $STATES"
[ "$FATAL_ONLY" = "true" ] && echo "   Fatal accidents only"
echo ""
fi
echo "📋 NEXT STEPS"
echo "   1. Review metadata: cat $EXPORT_DIR/METADATA.md"
echo "   2. Extract archive: unzip $ARCHIVE_PATH"
echo "   3. Load into analysis tool (pandas, R, PostgreSQL)"
echo "   4. Share archive via email, cloud storage, or repository"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
```

---

## SUCCESS CRITERIA

- [ ] Database connection verified
- [ ] Filters parsed and validated
- [ ] Sample events selected (matching filters)
- [ ] Events table exported
- [ ] All related tables exported (8 tables)
- [ ] Referential integrity maintained
- [ ] Metadata file generated
- [ ] ZIP archive created
- [ ] Summary displayed

---

## OUTPUT/DELIVERABLES

**Export Directory:**
- `/tmp/NTSB_Datasets/exports/sample_[TIMESTAMP]/` - Uncompressed CSV files + metadata

**Archive File:**
- `/tmp/NTSB_Datasets/exports/ntsb_sample_[SIZE]_[COUNT]events_[TIMESTAMP].zip` - Compressed archive

**Files in Export:**
- `events.csv` - Master event table
- `aircraft.csv` - Aircraft involved
- `flight_crew.csv` - Crew information
- `injury.csv` - Injury details
- `findings.csv` - Investigation findings
- `narratives.csv` - Accident narratives
- `engines.csv` - Engine specifications
- `events_sequence.csv` - Event sequencing
- `ntsb_admin.csv` - Administrative metadata
- `METADATA.md` - Comprehensive documentation

---

## RELATED COMMANDS

- `/data-coverage` - Analyze full database coverage before sampling
- `/validate-schema` - Validate database before export
- `/load-data` - Load full datasets
- `/geographic-analysis` - Geospatial analysis (can use sample)
- `/finding-analysis` - Finding code analysis (can use sample)

---

## NOTES

### When to Use

**Testing & Development:**
- Unit tests for ETL pipelines
- Query development and optimization
- Schema design validation
- Performance testing (without full dataset)

**Demos & Documentation:**
- Tutorials and workshops
- Documentation examples
- Presentations and reports
- GitHub repository examples

**Sharing & Collaboration:**
- Send to colleagues without database access
- Provide to students/researchers
- Share for code review
- Collaborate on analysis approaches

**Quick Analysis:**
- Exploratory data analysis
- Proof-of-concept work
- Algorithm prototyping
- Feature engineering experiments

### Sample Size Guidelines

**Tiny (100 events):**
- Quick tests and demos
- Documentation examples
- Fast load/iteration cycles
- <1MB size (email-friendly)

**Small (1,000 events):**
- Statistical validity for trends
- Comprehensive testing
- Workshop datasets
- ~5MB size

**Medium (10,000 events):**
- Robust analysis
- Machine learning training
- Production-like testing
- ~50MB size

**Custom:**
- Specific research needs
- Targeted analysis
- Filtered subsets

### Performance Considerations

**Export Times:**
- Tiny: <30 seconds
- Small: 1-2 minutes
- Medium: 3-5 minutes
- Custom (large): 5-10 minutes

**Disk Space:**
- Tiny: <1MB
- Small: ~5MB
- Medium: ~50MB
- Proportional to event count

---

## TROUBLESHOOTING

### Problem: "No events match filters"

**Solution:**
```bash
# Check filter validity
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events WHERE ev_state IN ('XX');"

# Relax filters or check available data
/data-coverage geographic  # See which states have data
/data-coverage temporal    # See year range
```

### Problem: "Sample smaller than requested"

**Solution:**
```bash
# Normal if filters are restrictive
# Check total events matching filters:
psql -d ntsb_aviation -c "
    SELECT COUNT(*) 
    FROM events 
    WHERE EXTRACT(YEAR FROM ev_date) BETWEEN 2020 AND 2023;
"

# Adjust filters or reduce sample size
```

### Problem: "CSV files empty or missing"

**Solution:**
```bash
# Check temporary table
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM temp_sample_events;"

# Ensure export directory exists
ls -la /tmp/NTSB_Datasets/exports/

# Check disk space
df -h /tmp/
```

### Problem: "ZIP creation failed"

**Solution:**
```bash
# Check zip utility installed
which zip || sudo pacman -S zip

# Check disk space
df -h /tmp/

# Create archive manually
cd /tmp/NTSB_Datasets/exports/sample_*/
zip -r ../sample.zip .
```

---

## EXAMPLE USAGE

### Export Tiny Sample for Testing

```bash
# Quick test dataset
/export-sample tiny

# Extract and test
unzip /tmp/NTSB_Datasets/exports/ntsb_sample_*.zip
cd sample_*/
wc -l *.csv
```

### Export Recent Fatal Accidents

```bash
# Last 5 years, fatal only
/export-sample small --recent --fatal

# Review fatalities
cut -d',' -f17 events.csv | tail -n +2 | paste -sd+ | bc
```

### Export Specific States and Years

```bash
# California accidents 2020-2023
/export-sample --count 500 --year 2020-2023 --state CA

# Review state distribution
cut -d',' -f9 events.csv | tail -n +2 | sort | uniq -c
```

### Export for Aircraft Analysis

```bash
# Cessna accidents
/export-sample medium --aircraft Cessna

# Check aircraft types
cut -d',' -f8,9 aircraft.csv | tail -n +2 | sort | uniq -c | sort -rn
```

### Complete Workflow

```bash
# 1. Check coverage
/data-coverage temporal

# 2. Export recent sample
/export-sample small --recent

# 3. Extract archive
cd /tmp/NTSB_Datasets/exports/
unzip ntsb_sample_*.zip

# 4. Load into Python
python3 << EOF
import pandas as pd
events = pd.read_csv('sample_*/events.csv')
print(events.describe())
EOF
```

---

**Command Version:** 1.0
**Last Updated:** 2025-11-06
**Adapted From:** Original command for NTSB Aviation Database
**Priority:** HIGH - Essential for testing, demos, and collaboration
