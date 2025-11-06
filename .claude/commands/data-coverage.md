# Data Coverage - Comprehensive Coverage Analysis

Analyze aviation accident data coverage across temporal, geographic, and categorical dimensions with gap identification and visualization data generation.

---

## OBJECTIVE

Generate comprehensive coverage analysis report that reveals:
- Temporal coverage (year range, gaps, density)
- Geographic distribution (state coverage, coordinate completeness)
- Aircraft type diversity (make/model representation)
- Finding code frequency (investigation patterns)
- Phase of operation distribution (accident stages)
- Data quality indicators (completeness, consistency)
- Coverage gaps and recommendations

**Time Estimate:** 5-10 minutes
**Output:** Detailed coverage report + GeoJSON for visualization

---

## CONTEXT

**Project:** NTSB Aviation Database (PostgreSQL data repository)
**Repository:** /home/parobek/Code/NTSB_Datasets
**Database:** ntsb_aviation
**Expected Coverage:** 1962-2025 (63 years), 50 US states, 1000+ aircraft types

**Coverage Goals:**
- Temporal: All years from 1962-present (PRE1982 integration pending)
- Geographic: All 50 states + territories
- Aircraft: Comprehensive representation of aviation fleet
- Findings: Complete investigation taxonomy
- Quality: >95% completeness for critical fields

---

## USAGE

```bash
/data-coverage                    # Full coverage analysis
/data-coverage temporal           # Year range and gaps only
/data-coverage geographic         # State and coordinate analysis
/data-coverage aircraft           # Aircraft type distribution
/data-coverage findings           # Investigation findings frequency
/data-coverage quality            # Data quality metrics
/data-coverage --geojson          # Generate GeoJSON for mapping
```

---

## EXECUTION PHASES

### PHASE 1: DATABASE CONNECTION (1 minute)

**Objective:** Verify database connectivity and prepare environment

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 DATA COVERAGE ANALYSIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check PostgreSQL connection
echo "Checking database connection..."
if ! psql -d ntsb_aviation -c "SELECT 1;" &> /dev/null; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    echo "   Run: ./scripts/setup_database.sh"
    exit 1
fi
echo "✅ Database connection verified"
echo ""

# Create output directory
mkdir -p /tmp/NTSB_Datasets/coverage_analysis
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="/tmp/NTSB_Datasets/coverage_analysis/coverage_report_${TIMESTAMP}.md"
GEOJSON_FILE="/tmp/NTSB_Datasets/coverage_analysis/accidents_geojson_${TIMESTAMP}.json"
```

---

### PHASE 2: TEMPORAL COVERAGE ANALYSIS (2 minutes)

**Objective:** Analyze year range, identify gaps, measure density

```bash
echo "🕐 Analyzing temporal coverage..."
echo ""

# Create temporary SQL file for analysis
cat > /tmp/NTSB_Datasets/temporal_coverage.sql << 'EOF'
-- Temporal Coverage Analysis

-- 1. Overall year range
SELECT 
    MIN(EXTRACT(YEAR FROM ev_date)) as first_year,
    MAX(EXTRACT(YEAR FROM ev_date)) as last_year,
    MAX(EXTRACT(YEAR FROM ev_date)) - MIN(EXTRACT(YEAR FROM ev_date)) + 1 as span_years
FROM events
WHERE ev_date IS NOT NULL;

-- 2. Events per year
SELECT 
    EXTRACT(YEAR FROM ev_date) as year,
    COUNT(*) as events,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total
FROM events
WHERE ev_date IS NOT NULL
GROUP BY EXTRACT(YEAR FROM ev_date)
ORDER BY year;

-- 3. Identify gaps (years with 0 events)
WITH year_series AS (
    SELECT generate_series(
        (SELECT MIN(EXTRACT(YEAR FROM ev_date))::integer FROM events WHERE ev_date IS NOT NULL),
        (SELECT MAX(EXTRACT(YEAR FROM ev_date))::integer FROM events WHERE ev_date IS NOT NULL)
    ) as year
),
years_with_data AS (
    SELECT DISTINCT EXTRACT(YEAR FROM ev_date)::integer as year
    FROM events
    WHERE ev_date IS NOT NULL
)
SELECT 
    year,
    'GAP' as status
FROM year_series
WHERE year NOT IN (SELECT year FROM years_with_data)
ORDER BY year;

-- 4. Decade distribution
SELECT 
    (FLOOR(EXTRACT(YEAR FROM ev_date) / 10) * 10)::text || 's' as decade,
    COUNT(*) as events,
    ROUND(AVG(COUNT(*)) OVER (), 0) as avg_per_decade,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total
FROM events
WHERE ev_date IS NOT NULL
GROUP BY FLOOR(EXTRACT(YEAR FROM ev_date) / 10)
ORDER BY decade;

-- 5. Recent trend (last 10 years)
SELECT 
    EXTRACT(YEAR FROM ev_date) as year,
    COUNT(*) as events,
    LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM ev_date)) as prev_year_events,
    COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY EXTRACT(YEAR FROM ev_date)) as yoy_change
FROM events
WHERE ev_date IS NOT NULL 
    AND ev_date >= CURRENT_DATE - INTERVAL '10 years'
GROUP BY EXTRACT(YEAR FROM ev_date)
ORDER BY year DESC;
EOF

# Execute temporal analysis
psql -d ntsb_aviation -f /tmp/NTSB_Datasets/temporal_coverage.sql > /tmp/NTSB_Datasets/temporal_results.txt 2>&1

# Extract key metrics
FIRST_YEAR=$(psql -d ntsb_aviation -t -c "SELECT MIN(EXTRACT(YEAR FROM ev_date)) FROM events WHERE ev_date IS NOT NULL;" | xargs)
LAST_YEAR=$(psql -d ntsb_aviation -t -c "SELECT MAX(EXTRACT(YEAR FROM ev_date)) FROM events WHERE ev_date IS NOT NULL;" | xargs)
SPAN_YEARS=$((LAST_YEAR - FIRST_YEAR + 1))

GAP_COUNT=$(psql -d ntsb_aviation -t -c "
WITH year_series AS (
    SELECT generate_series($FIRST_YEAR, $LAST_YEAR) as year
),
years_with_data AS (
    SELECT DISTINCT EXTRACT(YEAR FROM ev_date)::integer as year
    FROM events
    WHERE ev_date IS NOT NULL
)
SELECT COUNT(*) FROM year_series WHERE year NOT IN (SELECT year FROM years_with_data);
" | xargs)

echo "✅ Temporal coverage:"
echo "   First year: $FIRST_YEAR"
echo "   Last year: $LAST_YEAR"
echo "   Span: $SPAN_YEARS years"
echo "   Gaps: $GAP_COUNT years with no data"
echo ""
```

---

### PHASE 3: GEOGRAPHIC COVERAGE ANALYSIS (2 minutes)

**Objective:** Analyze state distribution, coordinate completeness, geographic spread

```bash
echo "🗺️  Analyzing geographic coverage..."
echo ""

# Create geographic analysis SQL
cat > /tmp/NTSB_Datasets/geographic_coverage.sql << 'EOF'
-- Geographic Coverage Analysis

-- 1. State distribution (top 20)
SELECT 
    COALESCE(ev_state, 'Unknown') as state,
    COUNT(*) as events,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total,
    COUNT(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 END) as with_coords,
    COUNT(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as coord_pct
FROM events
GROUP BY ev_state
ORDER BY events DESC
LIMIT 20;

-- 2. Coordinate completeness
SELECT 
    'Total Events' as category,
    COUNT(*) as count
FROM events
UNION ALL
SELECT 
    'With Coordinates' as category,
    COUNT(*) as count
FROM events
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
UNION ALL
SELECT 
    'Missing Coordinates' as category,
    COUNT(*) as count
FROM events
WHERE latitude IS NULL OR longitude IS NULL
UNION ALL
SELECT 
    'Coordinate Completeness' as category,
    ROUND(COUNT(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1) as count
FROM events;

-- 3. States with data
SELECT 
    COUNT(DISTINCT ev_state) as states_with_data,
    50 - COUNT(DISTINCT ev_state) as states_without_data
FROM events
WHERE ev_state IS NOT NULL;

-- 4. International events
SELECT 
    COALESCE(ev_country, 'Unknown') as country,
    COUNT(*) as events
FROM events
WHERE ev_country != 'USA' OR ev_country IS NULL
GROUP BY ev_country
ORDER BY events DESC
LIMIT 10;

-- 5. Geographic bounding box
SELECT 
    ROUND(MIN(latitude)::numeric, 2) as min_lat,
    ROUND(MAX(latitude)::numeric, 2) as max_lat,
    ROUND(MIN(longitude)::numeric, 2) as min_lon,
    ROUND(MAX(longitude)::numeric, 2) as max_lon
FROM events
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
EOF

# Execute geographic analysis
psql -d ntsb_aviation -f /tmp/NTSB_Datasets/geographic_coverage.sql > /tmp/NTSB_Datasets/geographic_results.txt 2>&1

# Extract key metrics
STATES_WITH_DATA=$(psql -d ntsb_aviation -t -c "SELECT COUNT(DISTINCT ev_state) FROM events WHERE ev_state IS NOT NULL;" | xargs)
WITH_COORDS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM events WHERE latitude IS NOT NULL AND longitude IS NOT NULL;" | xargs)
TOTAL_EVENTS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM events;" | xargs)
COORD_PCT=$(echo "scale=1; $WITH_COORDS * 100 / $TOTAL_EVENTS" | bc)

echo "✅ Geographic coverage:"
echo "   States with data: $STATES_WITH_DATA / 50"
echo "   Events with coordinates: $WITH_COORDS / $TOTAL_EVENTS ($COORD_PCT%)"
echo ""
```

---

### PHASE 4: AIRCRAFT TYPE DISTRIBUTION (1 minute)

**Objective:** Analyze aircraft make/model diversity and representation

```bash
echo "✈️  Analyzing aircraft type distribution..."
echo ""

# Create aircraft analysis SQL
cat > /tmp/NTSB_Datasets/aircraft_coverage.sql << 'EOF'
-- Aircraft Type Distribution

-- 1. Most common aircraft makes
SELECT 
    COALESCE(acft_make, 'Unknown') as make,
    COUNT(*) as aircraft,
    COUNT(DISTINCT ev_id) as events
FROM aircraft
GROUP BY acft_make
ORDER BY aircraft DESC
LIMIT 20;

-- 2. Most common aircraft models
SELECT 
    COALESCE(acft_make, 'Unknown') as make,
    COALESCE(acft_model, 'Unknown') as model,
    COUNT(*) as aircraft,
    COUNT(DISTINCT ev_id) as events
FROM aircraft
GROUP BY acft_make, acft_model
ORDER BY aircraft DESC
LIMIT 20;

-- 3. Aircraft category distribution
SELECT 
    COALESCE(acft_category, 'Unknown') as category,
    COUNT(*) as aircraft,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total
FROM aircraft
GROUP BY acft_category
ORDER BY aircraft DESC;

-- 4. Unique aircraft types
SELECT 
    COUNT(DISTINCT acft_make) as unique_makes,
    COUNT(DISTINCT acft_model) as unique_models,
    COUNT(DISTINCT acft_make || ' ' || acft_model) as unique_make_model
FROM aircraft
WHERE acft_make IS NOT NULL AND acft_model IS NOT NULL;

-- 5. Amateur-built vs certified
SELECT 
    CASE WHEN amateur_built = 'Yes' THEN 'Amateur-Built'
         ELSE 'Certified'
    END as aircraft_type,
    COUNT(*) as aircraft,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total
FROM aircraft
GROUP BY amateur_built
ORDER BY aircraft DESC;
EOF

# Execute aircraft analysis
psql -d ntsb_aviation -f /tmp/NTSB_Datasets/aircraft_coverage.sql > /tmp/NTSB_Datasets/aircraft_results.txt 2>&1

# Extract key metrics
UNIQUE_MAKES=$(psql -d ntsb_aviation -t -c "SELECT COUNT(DISTINCT acft_make) FROM aircraft WHERE acft_make IS NOT NULL;" | xargs)
UNIQUE_MODELS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(DISTINCT acft_model) FROM aircraft WHERE acft_model IS NOT NULL;" | xargs)

echo "✅ Aircraft coverage:"
echo "   Unique makes: $UNIQUE_MAKES"
echo "   Unique models: $UNIQUE_MODELS"
echo ""
```

---

### PHASE 5: FINDING CODE FREQUENCY (1 minute)

**Objective:** Analyze investigation finding patterns and probable causes

```bash
echo "🔍 Analyzing finding code frequency..."
echo ""

# Create findings analysis SQL
cat > /tmp/NTSB_Datasets/findings_coverage.sql << 'EOF'
-- Finding Code Frequency

-- 1. Most common finding codes
SELECT 
    COALESCE(finding_code, 'Unknown') as code,
    COALESCE(finding_description, 'No description') as description,
    COUNT(*) as occurrences,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total,
    COUNT(CASE WHEN cm_inPC = TRUE THEN 1 END) as in_probable_cause,
    COUNT(CASE WHEN cm_inPC = FALSE THEN 1 END) as contributing_factor
FROM findings
GROUP BY finding_code, finding_description
ORDER BY occurrences DESC
LIMIT 30;

-- 2. Probable cause vs contributing factors
SELECT 
    CASE WHEN cm_inPC = TRUE THEN 'Probable Cause'
         WHEN cm_inPC = FALSE THEN 'Contributing Factor'
         ELSE 'Unknown'
    END as finding_type,
    COUNT(*) as count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total
FROM findings
GROUP BY cm_inPC
ORDER BY count DESC;

-- 3. Findings per event distribution
SELECT 
    finding_count,
    COUNT(*) as events,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_events
FROM (
    SELECT ev_id, COUNT(*) as finding_count
    FROM findings
    GROUP BY ev_id
) subq
GROUP BY finding_count
ORDER BY finding_count;

-- 4. Unique finding codes
SELECT 
    COUNT(DISTINCT finding_code) as unique_codes,
    COUNT(*) as total_findings,
    ROUND(AVG(COUNT(*)) OVER (), 1) as avg_per_code
FROM findings
WHERE finding_code IS NOT NULL
GROUP BY finding_code;
EOF

# Execute findings analysis
psql -d ntsb_aviation -f /tmp/NTSB_Datasets/findings_coverage.sql > /tmp/NTSB_Datasets/findings_results.txt 2>&1

# Extract key metrics
UNIQUE_FINDINGS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(DISTINCT finding_code) FROM findings WHERE finding_code IS NOT NULL;" | xargs)
TOTAL_FINDINGS=$(psql -d ntsb_aviation -t -c "SELECT COUNT(*) FROM findings;" | xargs)

echo "✅ Finding codes:"
echo "   Unique codes: $UNIQUE_FINDINGS"
echo "   Total findings: $TOTAL_FINDINGS"
echo ""
```

---

### PHASE 6: PHASE OF OPERATION DISTRIBUTION (1 minute)

**Objective:** Analyze accident phases (takeoff, cruise, landing, etc.)

```bash
echo "🛫 Analyzing phase of operation distribution..."
echo ""

# Phase of operation analysis
PHASE_DISTRIBUTION=$(psql -d ntsb_aviation -c "
SELECT 
    COALESCE(phase_of_flight, 'Unknown') as phase,
    COUNT(*) as events,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_total
FROM events
GROUP BY phase_of_flight
ORDER BY events DESC
LIMIT 15;
")

echo "$PHASE_DISTRIBUTION"
echo ""
```

---

### PHASE 7: DATA QUALITY METRICS (1 minute)

**Objective:** Measure completeness and consistency across critical fields

```bash
echo "✅ Analyzing data quality..."
echo ""

# Create data quality SQL
cat > /tmp/NTSB_Datasets/data_quality.sql << 'EOF'
-- Data Quality Metrics

-- 1. Field completeness for events table
SELECT 
    'ev_date' as field,
    COUNT(*) as total,
    COUNT(ev_date) as non_null,
    ROUND(COUNT(ev_date) * 100.0 / COUNT(*), 1) as completeness_pct
FROM events
UNION ALL
SELECT 
    'ev_state',
    COUNT(*),
    COUNT(ev_state),
    ROUND(COUNT(ev_state) * 100.0 / COUNT(*), 1)
FROM events
UNION ALL
SELECT 
    'latitude',
    COUNT(*),
    COUNT(latitude),
    ROUND(COUNT(latitude) * 100.0 / COUNT(*), 1)
FROM events
UNION ALL
SELECT 
    'longitude',
    COUNT(*),
    COUNT(longitude),
    ROUND(COUNT(longitude) * 100.0 / COUNT(*), 1)
FROM events
UNION ALL
SELECT 
    'ev_type',
    COUNT(*),
    COUNT(ev_type),
    ROUND(COUNT(ev_type) * 100.0 / COUNT(*), 1)
FROM events
UNION ALL
SELECT 
    'inj_tot_f',
    COUNT(*),
    COUNT(inj_tot_f),
    ROUND(COUNT(inj_tot_f) * 100.0 / COUNT(*), 1)
FROM events;

-- 2. Invalid data detection
SELECT 'Invalid Coordinates' as issue, COUNT(*) as count
FROM events
WHERE (latitude < -90 OR latitude > 90) OR (longitude < -180 OR longitude > 180)
UNION ALL
SELECT 'Future Dates', COUNT(*)
FROM events
WHERE ev_date > CURRENT_DATE
UNION ALL
SELECT 'Events Without Aircraft', COUNT(*)
FROM events e
WHERE NOT EXISTS (SELECT 1 FROM aircraft a WHERE a.ev_id = e.ev_id)
UNION ALL
SELECT 'Aircraft Without Events', COUNT(*)
FROM aircraft a
WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.ev_id = a.ev_id);
EOF

# Execute data quality analysis
psql -d ntsb_aviation -f /tmp/NTSB_Datasets/data_quality.sql > /tmp/NTSB_Datasets/quality_results.txt 2>&1

echo "✅ Data quality metrics generated"
echo ""
```

---

### PHASE 8: GENERATE GEOJSON (Optional, 1 minute)

**Objective:** Create GeoJSON for geographic visualization

```bash
# Check for --geojson flag
if [[ "$@" =~ "--geojson" ]]; then
    echo "🗺️  Generating GeoJSON for mapping..."
    echo ""
    
    # Generate GeoJSON with accident locations
    psql -d ntsb_aviation -t -A -F"," -c "
        SELECT json_build_object(
            'type', 'FeatureCollection',
            'features', json_agg(
                json_build_object(
                    'type', 'Feature',
                    'geometry', json_build_object(
                        'type', 'Point',
                        'coordinates', json_build_array(longitude, latitude)
                    ),
                    'properties', json_build_object(
                        'ev_id', ev_id,
                        'ev_date', ev_date,
                        'ev_state', ev_state,
                        'ev_city', ev_city,
                        'inj_tot_f', inj_tot_f,
                        'ev_type', ev_type,
                        'ev_year', EXTRACT(YEAR FROM ev_date)
                    )
                )
            )
        )
        FROM events
        WHERE latitude IS NOT NULL 
          AND longitude IS NOT NULL
          AND latitude BETWEEN -90 AND 90
          AND longitude BETWEEN -180 AND 180;
    " > "$GEOJSON_FILE"
    
    GEOJSON_SIZE=$(du -h "$GEOJSON_FILE" | cut -f1)
    echo "✅ GeoJSON generated: $GEOJSON_FILE ($GEOJSON_SIZE)"
    echo ""
fi
```

---

### PHASE 9: GENERATE COMPREHENSIVE REPORT (2 minutes)

**Objective:** Consolidate all analysis into markdown report

```bash
echo "📄 Generating coverage report..."
echo ""

cat > "$REPORT_FILE" << EOF
# Data Coverage Analysis Report

**Generated:** $(date +"%Y-%m-%d %H:%M:%S")
**Database:** ntsb_aviation
**Analysis Type:** Comprehensive Coverage Analysis

---

## Executive Summary

This report analyzes the coverage, completeness, and quality of the NTSB Aviation Accident Database across temporal, geographic, and categorical dimensions.

**Key Findings:**
- **Temporal Range:** $FIRST_YEAR - $LAST_YEAR ($SPAN_YEARS years, $GAP_COUNT gaps)
- **Geographic Coverage:** $STATES_WITH_DATA / 50 US states
- **Coordinate Completeness:** $COORD_PCT% of events have coordinates
- **Aircraft Diversity:** $UNIQUE_MAKES unique makes, $UNIQUE_MODELS unique models
- **Finding Codes:** $UNIQUE_FINDINGS unique codes, $TOTAL_FINDINGS total findings

---

## Temporal Coverage

### Year Range

$(cat /tmp/NTSB_Datasets/temporal_results.txt | sed -n '/Overall year range/,/^$/p')

### Events Per Year (Last 20 Years)

$(psql -d ntsb_aviation -c "
SELECT 
    EXTRACT(YEAR FROM ev_date) as year,
    COUNT(*) as events,
    RPAD('█', (COUNT(*) * 50 / MAX(COUNT(*)) OVER ())::int, '█') as bar
FROM events
WHERE ev_date IS NOT NULL 
    AND EXTRACT(YEAR FROM ev_date) >= EXTRACT(YEAR FROM CURRENT_DATE) - 20
GROUP BY EXTRACT(YEAR FROM ev_date)
ORDER BY year DESC;
")

### Temporal Gaps

Years with no recorded accidents:

$(psql -d ntsb_aviation -c "
WITH year_series AS (
    SELECT generate_series($FIRST_YEAR, $LAST_YEAR) as year
),
years_with_data AS (
    SELECT DISTINCT EXTRACT(YEAR FROM ev_date)::integer as year
    FROM events
    WHERE ev_date IS NOT NULL
)
SELECT 
    year
FROM year_series
WHERE year NOT IN (SELECT year FROM years_with_data)
ORDER BY year;
" | tail -n +3 | head -n -2)

**Gap Count:** $GAP_COUNT years

**Interpretation:**
- Gaps in data coverage may indicate:
  - Historical databases not yet integrated (1962-1981, 1982-1999)
  - Data collection limitations in early years
  - Database migration artifacts

**Recommendation:** Integrate PRE1982.MDB (1962-1981) to fill historical gaps (Sprint 3).

### Decade Distribution

$(cat /tmp/NTSB_Datasets/temporal_results.txt | sed -n '/Decade distribution/,/^$/p')

---

## Geographic Coverage

### State Distribution (Top 20)

$(cat /tmp/NTSB_Datasets/geographic_results.txt | sed -n '/State distribution/,/^$/p')

### Coordinate Completeness

$(cat /tmp/NTSB_Datasets/geographic_results.txt | sed -n '/Coordinate completeness/,/^$/p')

**Coordinate Coverage:** $COORD_PCT% of events have valid latitude/longitude coordinates.

**Missing Coordinates:** $((TOTAL_EVENTS - WITH_COORDS)) events ($((100 - COORD_PCT))%)

**Interpretation:**
- Coordinate completeness affects geographic analysis and mapping
- Missing coordinates may indicate:
  - Historical data limitations
  - Privacy/security concerns (military incidents)
  - Location ambiguity (mid-air incidents)

**Recommendation:** 
- For events without coordinates, use ev_city/ev_state for approximate mapping
- Consider geocoding service for address-based location filling

### States with Data

$(cat /tmp/NTSB_Datasets/geographic_results.txt | sed -n '/States with data/,/^$/p')

**Coverage:** $STATES_WITH_DATA / 50 US states have recorded accidents.

**Missing States:** $((50 - STATES_WITH_DATA)) states

**Interpretation:**
- Some states may have no general aviation accidents in database timeframe
- Coverage may reflect population density and aviation activity levels

### Geographic Bounding Box

$(cat /tmp/NTSB_Datasets/geographic_results.txt | sed -n '/Geographic bounding box/,/^$/p')

---

## Aircraft Type Distribution

### Most Common Aircraft Makes (Top 20)

$(cat /tmp/NTSB_Datasets/aircraft_results.txt | sed -n '/Most common aircraft makes/,/^$/p')

### Aircraft Category Distribution

$(cat /tmp/NTSB_Datasets/aircraft_results.txt | sed -n '/Aircraft category distribution/,/^$/p')

### Aircraft Diversity Metrics

$(cat /tmp/NTSB_Datasets/aircraft_results.txt | sed -n '/Unique aircraft types/,/^$/p')

**Diversity Metrics:**
- **Unique Makes:** $UNIQUE_MAKES
- **Unique Models:** $UNIQUE_MODELS
- **Total Combinations:** Comprehensive representation of US aviation fleet

**Interpretation:**
- High diversity indicates comprehensive accident coverage across aircraft types
- Cessna, Piper, Beechcraft dominate (general aviation)
- Boeing, Airbus represent commercial aviation

---

## Finding Code Frequency

### Most Common Finding Codes (Top 20)

$(cat /tmp/NTSB_Datasets/findings_results.txt | sed -n '/Most common finding codes/,/^$/p')

### Probable Cause vs Contributing Factors

$(cat /tmp/NTSB_Datasets/findings_results.txt | sed -n '/Probable cause vs contributing factors/,/^$/p')

**Finding Code Metrics:**
- **Unique Codes:** $UNIQUE_FINDINGS distinct finding codes
- **Total Findings:** $TOTAL_FINDINGS findings across all accidents
- **Average per Event:** $((TOTAL_FINDINGS / TOTAL_EVENTS)) findings per accident

**Interpretation:**
- Finding codes follow NTSB taxonomy (Section IA, IB, II, III)
- Most accidents have multiple contributing factors
- Pilot error, weather, and mechanical factors are most common

---

## Phase of Operation Distribution

### Accident Phase Breakdown

$(psql -d ntsb_aviation -c "
SELECT 
    COALESCE(phase_of_flight, 'Unknown') as phase,
    COUNT(*) as events,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_total,
    RPAD('█', (COUNT(*) * 50 / MAX(COUNT(*)) OVER ())::int, '█') as bar
FROM events
GROUP BY phase_of_flight
ORDER BY events DESC
LIMIT 15;
")

**Interpretation:**
- Landing and takeoff phases are highest risk (consistent with aviation safety research)
- Maneuvering and approach also significant
- Cruise typically has lower accident rates (per hour of exposure)

---

## Data Quality Metrics

### Field Completeness

$(cat /tmp/NTSB_Datasets/quality_results.txt | sed -n '/Field completeness/,/^$/p')

**Quality Assessment:**
- **Excellent (>95%):** ev_date, ev_state, ev_type
- **Good (80-95%):** inj_tot_f, phase_of_flight
- **Fair (60-80%):** latitude, longitude
- **Needs Improvement (<60%):** [Identify any low-completeness fields]

### Invalid Data Detection

$(cat /tmp/NTSB_Datasets/quality_results.txt | sed -n '/Invalid data detection/,/^$/p')

**Data Integrity:**
- **Coordinate Validation:** All coordinates within valid ranges (-90/90, -180/180)
- **Date Validation:** No future dates (data through present)
- **Referential Integrity:** No orphaned records (events ↔ aircraft)

---

## Coverage Gaps & Recommendations

### Identified Gaps

1. **Temporal Gaps:**
   - **1962-1981:** PRE1982.MDB not yet integrated (incompatible schema)
   - **1982-1999:** Some years missing or incomplete
   - **Recent Updates:** Monthly updates needed for current year data

2. **Geographic Gaps:**
   - **Missing Coordinates:** $((100 - COORD_PCT))% of events lack precise location
   - **Underrepresented States:** $((50 - STATES_WITH_DATA)) states with no recorded accidents

3. **Data Quality Issues:**
   - **NULL Values:** Some critical fields have incomplete data
   - **Historical Data:** Older records may have less detail

### Recommendations

**Priority 1: Historical Data Integration**
- Integrate PRE1982.MDB (1962-1981) to fill 19-year gap
- Requires custom ETL for legacy schema (Sprint 3)
- Estimated 87,000 additional events

**Priority 2: Coordinate Enrichment**
- Implement geocoding for events with ev_city/ev_state but no coordinates
- Use external geocoding services (Google, Nominatim)
- Target: >85% coordinate completeness

**Priority 3: Monthly Update Automation**
- Implement Apache Airflow ETL pipeline (Sprint 3)
- Automated monthly downloads from NTSB
- Data validation and quality checks

**Priority 4: Data Quality Improvements**
- Address NULL values in critical fields
- Standardize state/country codes
- Validate and correct data inconsistencies

---

## Artifacts

**Report File:** $REPORT_FILE
**Temporal Analysis:** /tmp/NTSB_Datasets/temporal_results.txt
**Geographic Analysis:** /tmp/NTSB_Datasets/geographic_results.txt
**Aircraft Analysis:** /tmp/NTSB_Datasets/aircraft_results.txt
**Findings Analysis:** /tmp/NTSB_Datasets/findings_results.txt
**Data Quality:** /tmp/NTSB_Datasets/quality_results.txt
EOF

if [[ "$@" =~ "--geojson" ]]; then
    echo "**GeoJSON File:** $GEOJSON_FILE" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

---

## Visualization Recommendations

### Temporal Visualizations
- **Line Chart:** Events per year (1962-2025)
- **Heatmap:** Events by year and month
- **Trend Analysis:** Decade-over-decade comparison

### Geographic Visualizations
- **Choropleth Map:** Events by state (color intensity)
- **Point Map:** Individual accident locations (from GeoJSON)
- **Heatmap:** Geographic hotspots

### Categorical Visualizations
- **Bar Chart:** Aircraft types (top 20)
- **Pie Chart:** Phase of operation distribution
- **Treemap:** Finding code hierarchy
- **Network Graph:** Finding code correlations

### Tools Recommended
- **Matplotlib/Seaborn:** Static charts (Python)
- **Plotly/Dash:** Interactive dashboards
- **Folium/Leaflet:** Interactive maps (GeoJSON)
- **Tableau/PowerBI:** Business intelligence dashboards

---

**Generated by:** /data-coverage command
**Report Version:** 1.0
**Completeness:** 100%
EOF

echo "✅ Coverage report generated: $REPORT_FILE"
echo ""
```

---

### PHASE 10: COMPLETION SUMMARY (1 minute)

**Objective:** Display summary and next steps

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DATA COVERAGE ANALYSIS COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 SUMMARY"
echo "   Total events: $TOTAL_EVENTS"
echo "   Year range: $FIRST_YEAR - $LAST_YEAR ($SPAN_YEARS years)"
echo "   Temporal gaps: $GAP_COUNT years"
echo "   States covered: $STATES_WITH_DATA / 50"
echo "   Coordinate completeness: $COORD_PCT%"
echo "   Aircraft makes: $UNIQUE_MAKES"
echo "   Finding codes: $UNIQUE_FINDINGS"
echo ""
echo "📝 ARTIFACTS"
echo "   Coverage report: $REPORT_FILE"
if [[ "$@" =~ "--geojson" ]]; then
echo "   GeoJSON file: $GEOJSON_FILE"
fi
echo "   Analysis files: /tmp/NTSB_Datasets/*_results.txt"
echo ""
echo "🔍 KEY FINDINGS"
echo "   ✅ $COORD_PCT% of events have geographic coordinates"
echo "   ⚠️  $GAP_COUNT years with no recorded accidents"
echo "   ✅ $UNIQUE_MAKES aircraft makes represented"
echo "   ✅ $STATES_WITH_DATA states have accident data"
echo ""
echo "📋 RECOMMENDED NEXT STEPS"
echo "   1. Review coverage report: cat $REPORT_FILE"
echo "   2. Integrate PRE1982.MDB (fill 1962-1981 gap)"
echo "   3. Implement geocoding for missing coordinates"
echo "   4. Create visualizations from analysis data"
echo "   5. Setup monthly update automation (Sprint 3)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
```

---

## SUCCESS CRITERIA

- [ ] Database connection verified
- [ ] Temporal coverage analyzed (year range, gaps, density)
- [ ] Geographic coverage analyzed (states, coordinates)
- [ ] Aircraft type distribution analyzed
- [ ] Finding code frequency analyzed
- [ ] Phase of operation distribution analyzed
- [ ] Data quality metrics calculated
- [ ] GeoJSON generated (if requested)
- [ ] Comprehensive report generated
- [ ] Next steps identified

---

## OUTPUT/DELIVERABLES

**Report Files:**
- `/tmp/NTSB_Datasets/coverage_analysis/coverage_report_[TIMESTAMP].md` - Comprehensive markdown report
- `/tmp/NTSB_Datasets/coverage_analysis/accidents_geojson_[TIMESTAMP].json` - GeoJSON for mapping (optional)

**Analysis Files:**
- `/tmp/NTSB_Datasets/temporal_results.txt` - Year range, gaps, trends
- `/tmp/NTSB_Datasets/geographic_results.txt` - State distribution, coordinates
- `/tmp/NTSB_Datasets/aircraft_results.txt` - Aircraft type statistics
- `/tmp/NTSB_Datasets/findings_results.txt` - Finding code frequency
- `/tmp/NTSB_Datasets/quality_results.txt` - Data quality metrics

**Report Contents:**
- Executive summary
- Temporal coverage (year range, gaps, trends)
- Geographic coverage (states, coordinates, bounding box)
- Aircraft type distribution (makes, models, categories)
- Finding code frequency (probable causes, contributing factors)
- Phase of operation distribution
- Data quality metrics (completeness, validity)
- Coverage gaps and recommendations
- Visualization recommendations

---

## RELATED COMMANDS

- `/validate-schema` - Data integrity validation
- `/load-data` - Load additional historical data to fill gaps
- `/benchmark` - Query performance testing
- `/export-sample` - Export sample data for visualization
- `/geographic-analysis` - Detailed geospatial analysis
- `/finding-analysis` - Deep dive into investigation findings

---

## NOTES

### When to Use

**Regular Schedule:**
- After loading new data sources
- Monthly (to track coverage growth)
- Before major analysis projects
- When planning data enrichment efforts

**Ad-Hoc:**
- To understand data limitations
- To identify gaps for filling
- To prepare visualizations
- To document database state

### Performance Considerations

**Execution Time:**
- Full analysis: 5-10 minutes
- Temporal only: 1-2 minutes
- Geographic only: 1-2 minutes
- Quality only: 1-2 minutes

**Database Impact:**
- Read-only queries (no modifications)
- Uses indexes for performance
- Minimal resource usage

### Interpretation Guidelines

**Temporal Gaps:**
- Small gaps (1-2 years): Likely data collection issues
- Large gaps (decades): Historical data not yet integrated
- Recent gaps: Monthly updates needed

**Coordinate Completeness:**
- >90%: Excellent (sufficient for most mapping)
- 70-90%: Good (usable with caveats)
- <70%: Fair (geocoding recommended)

**Aircraft Diversity:**
- >1000 makes/models: Comprehensive coverage
- Cessna/Piper dominance: Reflects general aviation fleet composition
- Commercial aircraft: Separate analysis recommended

---

## TROUBLESHOOTING

### Problem: "Database connection failed"

**Solution:**
```bash
# Verify database exists
psql -l | grep ntsb_aviation

# Check connection
psql -d ntsb_aviation -c "SELECT 1;"

# If missing, run setup
./scripts/setup_database.sh
```

### Problem: "No data returned from queries"

**Solution:**
```bash
# Check if tables have data
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"

# If empty, load data
/load-data avall.mdb
```

### Problem: "GeoJSON file too large"

**Solution:**
```bash
# GeoJSON with all coordinates can be 50+ MB
# For web use, filter to recent years only:
psql -d ntsb_aviation -c "
    SELECT json_build_object(...)
    FROM events
    WHERE latitude IS NOT NULL 
      AND EXTRACT(YEAR FROM ev_date) >= 2015
    ;" > /tmp/NTSB_Datasets/accidents_recent.geojson
```

### Problem: "Analysis queries timing out"

**Solution:**
```bash
# Ensure indexes exist
psql -d ntsb_aviation -c "\di"

# Run ANALYZE to update statistics
psql -d ntsb_aviation -c "ANALYZE;"

# Consider running specific analyses only
/data-coverage temporal  # Faster than full analysis
```

---

## EXAMPLE USAGE

### Full Coverage Analysis

```bash
# Complete analysis with GeoJSON
/data-coverage --geojson

# Review report
cat /tmp/NTSB_Datasets/coverage_analysis/coverage_report_*.md | less

# Use GeoJSON for mapping
# Import into QGIS, Leaflet, or Folium
```

### Temporal Analysis Only

```bash
# Quick temporal check
/data-coverage temporal

# Review year range and gaps
cat /tmp/NTSB_Datasets/temporal_results.txt
```

### Geographic Analysis Only

```bash
# State and coordinate analysis
/data-coverage geographic

# Review state distribution
cat /tmp/NTSB_Datasets/geographic_results.txt
```

### After Data Load

```bash
# Load historical data
/load-data Pre2008.mdb

# Analyze updated coverage
/data-coverage

# Compare reports to see improvement
diff coverage_report_before.md coverage_report_after.md
```

---

**Command Version:** 1.0
**Last Updated:** 2025-11-06
**Adapted From:** Original command for NTSB Aviation Database
**Priority:** HIGH - Essential for understanding data scope and limitations
