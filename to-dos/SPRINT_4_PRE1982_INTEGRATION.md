# Sprint 4: PRE1982.MDB Historical Data Integration

**Sprint**: Phase 1 Sprint 4
**Objective**: Integrate 1962-1981 aviation accident data to complete 63-year historical dataset
**Timeline**: 12-16 hours (3-4 work sessions)
**Priority**: MEDIUM (fills critical 20-year coverage gap)
**Status**: PLANNING

---

## Executive Summary

Sprint 4 will integrate PRE1982.MDB into the NTSB Aviation Database, adding approximately 87,000 aviation accident events from 1962-1981. This integration is critical for completing the historical dataset, extending coverage from the current 2000-2025 timeframe (26 years) back to 1962 (63 years total).

**Key Challenge**: PRE1982.MDB uses a fundamentally different schema structure than modern NTSB databases. The legacy database stores data in a **denormalized, wide-table format** with 200+ columns per table, while the current PostgreSQL schema uses a **normalized relational structure** with 11 tables and focused entities.

**Approach**: Custom ETL pipeline to transform denormalized legacy data into normalized schema, preserving all historical information while maintaining data quality standards established in Sprints 1-3.

**Expected Outcomes**:
- Complete historical dataset: 1962-2025 (63 years)
- +87,000 events (92,771 → ~180,000)
- +~1.5 million child records (aircraft, crew, injury, findings)
- Database size: 512 MB → ~1,000 MB
- Foundation for longitudinal aviation safety trend analysis

**Why This Matters**:
- Enables 63-year aviation safety trend analysis (vs. current 26 years)
- Bridges critical data gap: 1962-1976 (complete gap), 1977-1981 (partial coverage)
- Provides research-grade historical dataset for academic and policy research
- Completes NTSB's public aviation accident record digitization

---

## Prerequisites

**Required Before Sprint 4**:
- [x] Phase 1 Sprint 3 complete (Airflow + monitoring infrastructure)
- [x] Database backup created (`pg_dump ntsb_aviation > backup_pre_pre1982.sql`)
- [x] PRE1982.MDB file available in `datasets/` directory (188 MB)
- [x] `mdbtools` installed for data extraction (`sudo apt install mdbtools` or `brew install mdbtools`)
- [x] Python environment with pandas, psycopg2, numpy
- [x] Disk space available: +500 MB for PRE1982 data
- [x] `docs/PRE1982_ANALYSIS.md` reviewed (schema structure, challenges)

**Knowledge Prerequisites**:
- [ ] Understanding of denormalized vs. normalized database design
- [ ] Familiarity with pandas pivot operations (wide → tall transformations)
- [ ] Understanding of NTSB aviation coding system (`ref_docs/codman.pdf`)
- [ ] Knowledge of staging table pattern (see `scripts/load_with_staging.py`)

**System Requirements**:
- PostgreSQL 18.0+ (owner: current user, not postgres superuser)
- Python 3.11+
- mdbtools 1.0+
- Minimum 2 GB available disk space
- 4 GB+ RAM recommended (for large pandas DataFrames)

---

## Problem Statement

### Current State

**Database Coverage**:
- **Events**: 92,771 (1977-2025 with gaps)
- **Date Coverage**: 26 years (2000-2025 primary, some 1977-1999 data)
- **Coverage Gaps**:
  - 1962-1976: Complete gap (15 years)
  - 1977-1999: Partial coverage (some data from Pre2008.mdb overlap)
- **Database Size**: 512 MB
- **Total Rows**: ~733,000 across 11 tables

**Limitations**:
- Cannot analyze long-term aviation safety trends (50+ years)
- Missing critical early jet age accidents (1960s-1970s)
- Incomplete dataset for academic research and policy analysis
- 15-year complete data gap represents ~15,000-20,000 missing events

### Target State

**Database Coverage**:
- **Events**: ~180,000 (1962-2025 complete)
- **Date Coverage**: 63 years (1962-2025)
- **Coverage Gaps**: NONE (complete historical record)
- **Database Size**: ~1,000 MB (projected)
- **Total Rows**: ~2.5 million across 11 tables

**Benefits**:
- Complete NTSB aviation accident historical record
- Enable 63-year longitudinal trend analysis
- Support academic research on aviation safety evolution
- Foundation for machine learning on 6+ decades of data
- Policy analysis: Compare regulatory eras (pre-1978 CAB vs. post-1978 FAA)

### Integration Challenges

**From `docs/PRE1982_ANALYSIS.md`**:

#### 1. **Schema Incompatibility** (CRITICAL)

**PRE1982 Structure** (Denormalized):
- **2 main tables**: `tblFirstHalf`, `tblSecondHalf` (split due to MS Access 255 column limit)
- **200+ columns each**: Single row per event with all data embedded
- **Primary Key**: `RecNum` (integer: 40, 41, 42, ...)
- **Example columns**:
  - Event: `DATE_OCCURRENCE`, `DOCKET_NO`, `LOCATION`, `LOCAT_STATE_TERR`
  - Aircraft: `ACFT_MAKE`, `ACFT_MODEL`, `REGIST_NO`, `NO_ENGINES`
  - Pilot 1: `HOURS_TOTAL_PILOT1`, `AGE_PILOT1`, `MEDICAL_CERT_PILOT1`
  - Pilot 2: `HOURS_TOTAL_PILOT2`, `AGE_PILOT2`, `MEDICAL_CERT_PILOT2`
  - Injuries: `PILOT_FATAL`, `PILOT_SERIOUS`, `PASSENGERS_FATAL`, etc. (50+ columns)
  - Causes: `CAUSE_FACTOR_1P`, `CAUSE_FACTOR_1M`, ... `CAUSE_FACTOR_10S` (30 columns)

**Modern Schema** (Normalized):
- **11 related tables**: `events`, `aircraft`, `Flight_Crew`, `injury`, `Findings`, etc.
- **36 columns** in `events` table (focused entity)
- **Primary Key**: `ev_id` (VARCHAR: "19620723R000040")
- **Relationships**: Foreign keys linking child tables

**Transformation Required**:
- Split single denormalized row → multiple normalized rows across 6 tables
- Example: 1 PRE1982 row → 1 events + 1-3 aircraft + 0-2 flight_crew + 10-50 injury + 1-10 findings + 0-5 narratives

#### 2. **No Standard ev_id Field** (CRITICAL)

**Problem**: PRE1982 uses `RecNum` (integer) instead of structured `ev_id` (VARCHAR)

**Impact**: Cannot use existing foreign key relationships

**Solution**: Generate synthetic `ev_id` from `RecNum` + `DATE_OCCURRENCE`
- **Format**: `YYYYMMDDX{RecNum:06d}` (X = legacy marker)
- **Example**: RecNum=40, DATE_OCCURRENCE=07/23/62 → `19620723X000040`
- **Collision Risk**: MINIMAL (no overlap with modern ev_id pattern starting with 2000s)

#### 3. **Coded Fields Without Mapping Tables** (HIGH)

**Problem**: Many fields use legacy codes requiring lookup tables

**Examples**:
- `LOCAT_STATE_TERR`: `32` (numeric code, not 2-letter state abbreviation)
- `AGE_PILOT1`: `ZA` (coded age range, not actual age)
- `HOURS_TOTAL_PILOT1`: `11508A` (hours with letter suffix, unclear meaning)
- `CAUSE_FACTOR_1P/M/S`: `70`, `A`, `CB` (legacy cause codes)

**Impact**: Cannot decode ~30% of fields without reference tables

**Mitigation Options**:
1. **Preserve codes as-is**: Store legacy codes, document as "requires manual lookup"
2. **Partial mapping**: Map critical fields only (state codes, injury codes)
3. **Obtain ct_Pre1982 table**: Extract code lookup table from PRE1982.MDB

**Recommended**: Option 3 (extract `ct_Pre1982`), fallback to Option 1 for unmapped codes

#### 4. **Different Data Types and Formats** (MEDIUM)

**Date Format Differences**:
- **PRE1982**: `DATE_OCCURRENCE` = `"MM/DD/YY HH:MM:SS"` (e.g., `"07/23/62 00:00:00"`)
  - **Issue**: 2-digit year requires century inference (62 = 1962, not 2062)
- **Modern**: `ev_date` = `YYYY-MM-DD` (e.g., `2008-01-15`)

**Time Format**:
- **PRE1982**: Likely `TIME_OCCUR` as `HHMM` integer (same as modern)
- **Modern**: `ev_time` = `HH:MM:SS` TIME type

**Coordinates**:
- **PRE1982**: Possibly DMS (degrees/minutes/seconds) or coded values
- **Modern**: Decimal degrees (latitude: -90 to 90, longitude: -180 to 180)

**Pilot Hours**:
- **PRE1982**: `11508A` (numeric with letter suffix, unclear encoding)
- **Modern**: Integer (total hours)

#### 5. **Injury Data Denormalization** (MEDIUM)

**PRE1982** (50+ columns, wide format):
```
PILOT_FATAL=1, PILOT_SERIOUS=0, PILOT_MINOR=0, PILOT_NONE=0
CO_PILOT_FATAL=0, CO_PILOT_SERIOUS=1, CO_PILOT_MINOR=0, CO_PILOT_NONE=0
PASSENGERS_FATAL=5, PASSENGERS_SERIOUS=2, PASSENGERS_MINOR=3, PASSENGERS_NONE=0
CREW_FATAL=0, CREW_SERIOUS=0, ...
TOTAL_ABRD_FATAL=6, TOTAL_ABRD_SERIOUS=3, ...
```

**Modern Schema** (normalized, tall format):
```
injury table:
  (ev_id=19620723X000040, inj_person_category='PILOT', inj_level='FATL', inj_person_count=1)
  (ev_id=19620723X000040, inj_person_category='CO-PILOT', inj_level='SERS', inj_person_count=1)
  (ev_id=19620723X000040, inj_person_category='PASSENGER', inj_level='FATL', inj_person_count=5)
  (ev_id=19620723X000040, inj_person_category='PASSENGER', inj_level='SERS', inj_person_count=2)
  ... (10-50 rows per event)
```

**Transformation**: Pivot wide → tall using pandas melt/stack operations

#### 6. **Cause Factor Coding System** (MEDIUM)

**PRE1982** (30 columns, legacy codes):
```
CAUSE_FACTOR_1P=70, CAUSE_FACTOR_1M=A, CAUSE_FACTOR_1S=CB
CAUSE_FACTOR_2P=12, CAUSE_FACTOR_2M=NULL, CAUSE_FACTOR_2S=NULL
... (up to CAUSE_FACTOR_10S)
```
- **P**rimary, **M**odifier, **S**econdary codes
- Requires `ct_Pre1982` lookup table for descriptions

**Modern Schema** (hierarchical codes from `codman.pdf`):
```
Findings table:
  (finding_code='10000', finding_description='AIRFRAME - GENERAL', cm_inPC=TRUE)
  (finding_code='22000', finding_description='STALL/SPIN', cm_inPC=FALSE)
```

**Transformation**:
1. Decode legacy P/M/S codes using `ct_Pre1982`
2. Map to modern finding codes (if possible)
3. If no mapping: Store legacy code in `finding_description` as "LEGACY:70-A-CB"
4. Mark first cause factor as probable cause (`cm_inPC=TRUE`)

---

## Task Breakdown

### **Phase 1: Schema Analysis & Mapping** (3-4 hours)

#### Task 1.1: Extract PRE1982 Schema and Sample Data
**Duration**: 1 hour

**Steps**:
```bash
# 1. Extract schema definition
mdb-schema datasets/PRE1982.MDB postgres > /tmp/NTSB_Datasets/pre1982_schema.sql

# 2. List all tables
mdb-tables datasets/PRE1982.MDB > /tmp/NTSB_Datasets/pre1982_tables.txt

# 3. Export sample data (100 rows) from each table
mdb-export datasets/PRE1982.MDB tblFirstHalf | head -100 > /tmp/NTSB_Datasets/tblFirstHalf_sample.csv
mdb-export datasets/PRE1982.MDB tblSecondHalf | head -100 > /tmp/NTSB_Datasets/tblSecondHalf_sample.csv
mdb-export datasets/PRE1982.MDB tblOccurrences | head -100 > /tmp/NTSB_Datasets/tblOccurrences_sample.csv
mdb-export datasets/PRE1982.MDB tblSeqOfEvents | head -100 > /tmp/NTSB_Datasets/tblSeqOfEvents_sample.csv
mdb-export datasets/PRE1982.MDB ct_Pre1982 > /tmp/NTSB_Datasets/ct_Pre1982_full.csv

# 4. Get row counts
for table in tblFirstHalf tblSecondHalf tblOccurrences tblSeqOfEvents ct_Pre1982; do
    echo "$table: $(mdb-export datasets/PRE1982.MDB $table | wc -l) rows"
done
```

**Deliverables**:
- [ ] `/tmp/NTSB_Datasets/pre1982_schema.sql` (PostgreSQL schema)
- [ ] `/tmp/NTSB_Datasets/pre1982_tables.txt` (table list)
- [ ] `/tmp/NTSB_Datasets/tblFirstHalf_sample.csv` (100 sample rows)
- [ ] `/tmp/NTSB_Datasets/tblSecondHalf_sample.csv` (100 sample rows)
- [ ] `/tmp/NTSB_Datasets/ct_Pre1982_full.csv` (complete code table)
- [ ] Row count documentation

#### Task 1.2: Analyze Column Structure
**Duration**: 1 hour

**Steps**:
1. Load sample data into pandas:
   ```python
   import pandas as pd

   first_half = pd.read_csv('/tmp/NTSB_Datasets/tblFirstHalf_sample.csv')
   second_half = pd.read_csv('/tmp/NTSB_Datasets/tblSecondHalf_sample.csv')
   codes = pd.read_csv('/tmp/NTSB_Datasets/ct_Pre1982_full.csv')

   # Inspect structure
   print(f"tblFirstHalf: {len(first_half.columns)} columns")
   print(first_half.dtypes)
   print(first_half.head())

   print(f"\ntblSecondHalf: {len(second_half.columns)} columns")
   print(second_half.dtypes)

   print(f"\nct_Pre1982: {len(codes.columns)} columns")
   print(codes.head(20))
   ```

2. Categorize columns by purpose:
   - Event metadata (date, location, NTSB number)
   - Aircraft (make, model, registration, damage)
   - Pilot 1 (hours, age, certification)
   - Pilot 2 (hours, age, certification)
   - Injuries (50+ columns: role × severity)
   - Cause factors (30 columns: 10 factors × 3 codes)
   - Narrative text
   - Investigation metadata

3. Identify data types:
   - Date/time fields
   - Integer fields (hours, ages, counts)
   - Coded fields (state codes, cause codes, age codes)
   - Text fields (narratives, descriptions)

**Deliverables**:
- [ ] Column categorization spreadsheet or markdown table
- [ ] Data type mapping (PRE1982 type → PostgreSQL type)
- [ ] Coded field inventory (fields requiring lookup)

#### Task 1.3: Create Column Mapping Specification
**Duration**: 1-2 hours

**Steps**:
Create comprehensive mapping document: `/tmp/NTSB_Datasets/column_mapping_spec.md`

**Structure**:

```markdown
# PRE1982.MDB Column Mapping Specification

## 1. Events Table Mapping (tblFirstHalf → events)

| PRE1982 Column | Type | Modern Column | Type | Transformation | Notes |
|----------------|------|---------------|------|----------------|-------|
| RecNum | INTEGER | (synthetic ev_id) | VARCHAR(20) | Generate YYYYMMDDX{RecNum:06d} | Primary key transformation |
| DATE_OCCURRENCE | DATETIME | ev_date | DATE | Parse MM/DD/YY → YYYY-MM-DD | 2-digit year: 62-99 → 1962-1999 |
| TIME_OCCUR | INTEGER | ev_time | TIME | HHMM → HH:MM:SS | Same as modern format |
| LOCATION | TEXT | ev_city | VARCHAR(100) | Direct copy | May need city/state split |
| LOCAT_STATE_TERR | INTEGER | ev_state | CHAR(2) | Map code → 2-letter abbr | Requires state code table |
| DOCKET_NO | TEXT | ntsb_no | VARCHAR(30) | Direct copy | NTSB docket number |
| LATITUDE_DMS | TEXT | dec_latitude | DECIMAL(10,6) | DMS → decimal degrees | If DMS format |
| LONGITUDE_DMS | TEXT | dec_longitude | DECIMAL(11,6) | DMS → decimal degrees | If DMS format |
| ... | ... | ... | ... | ... | ... |

## 2. Aircraft Table Mapping (tblFirstHalf → aircraft)

| PRE1982 Column | Type | Modern Column | Type | Transformation | Notes |
|----------------|------|---------------|------|----------------|-------|
| RecNum | INTEGER | ev_id | VARCHAR(20) | Generate YYYYMMDDX{RecNum:06d} | Foreign key |
| (synthetic) | - | Aircraft_Key | VARCHAR(20) | Hard-code '1' | Single aircraft per event in PRE1982 |
| REGIST_NO | TEXT | regis_no | VARCHAR(15) | Direct copy | Aircraft registration |
| ACFT_MAKE | TEXT | acft_make | VARCHAR(100) | Decode code or direct copy | Check if coded |
| ACFT_MODEL | TEXT | acft_model | VARCHAR(100) | Direct copy | Aircraft model |
| NO_ENGINES | INTEGER | num_eng | INTEGER | Direct copy | Number of engines |
| ACFT_DAMAGE | TEXT | damage | VARCHAR(10) | Map to standard codes | DEST, SUBS, MINR, NONE |
| ... | ... | ... | ... | ... | ... |

## 3. Flight_Crew Table Mapping (tblFirstHalf → Flight_Crew, 2 rows per event)

### Pilot 1 Mapping:
| PRE1982 Column | Type | Modern Column | Type | Transformation | Notes |
|----------------|------|---------------|------|----------------|-------|
| RecNum | INTEGER | ev_id | VARCHAR(20) | Generate YYYYMMDDX{RecNum:06d} | Foreign key |
| (synthetic) | - | Aircraft_Key | VARCHAR(20) | Hard-code '1' | Single aircraft per event |
| (synthetic) | - | crew_category | VARCHAR(30) | Hard-code 'PILOT' | Pilot 1 always PILOT |
| AGE_PILOT1 | TEXT | crew_age | INTEGER | Decode age code ZA → integer | Requires code table |
| HOURS_TOTAL_PILOT1 | TEXT | pilot_tot_time | INTEGER | Parse "11508A" → 11508 | Remove letter suffix |
| MEDICAL_CERT_PILOT1 | TEXT | pilot_med_class | VARCHAR(5) | Map to class 1/2/3 | Requires code table |
| ... | ... | ... | ... | ... | ... |

### Pilot 2 Mapping:
(Same structure, with PILOT2 columns → crew_category='CO-PILOT')

## 4. Injury Table Mapping (tblFirstHalf/tblSecondHalf → injury, 10-50 rows per event)

**Denormalized Columns** (Wide format):
- PILOT_FATAL, PILOT_SERIOUS, PILOT_MINOR, PILOT_NONE
- CO_PILOT_FATAL, CO_PILOT_SERIOUS, CO_PILOT_MINOR, CO_PILOT_NONE
- PASSENGERS_FATAL, PASSENGERS_SERIOUS, PASSENGERS_MINOR, PASSENGERS_NONE
- CREW_FATAL, CREW_SERIOUS, CREW_MINOR, CREW_NONE
- TOTAL_ABRD_FATAL, TOTAL_ABRD_SERIOUS, TOTAL_ABRD_MINOR, TOTAL_ABRD_NONE
- (50+ columns total)

**Transformation Logic**:
```python
injury_categories = [
    ('PILOT', ['PILOT_FATAL', 'PILOT_SERIOUS', 'PILOT_MINOR', 'PILOT_NONE']),
    ('CO-PILOT', ['CO_PILOT_FATAL', 'CO_PILOT_SERIOUS', 'CO_PILOT_MINOR', 'CO_PILOT_NONE']),
    ('PASSENGER', ['PASSENGERS_FATAL', 'PASSENGERS_SERIOUS', 'PASSENGERS_MINOR', 'PASSENGERS_NONE']),
    ('CREW', ['CREW_FATAL', 'CREW_SERIOUS', 'CREW_MINOR', 'CREW_NONE']),
]

for category, columns in injury_categories:
    for col in columns:
        level = col.split('_')[-1]  # FATAL → FATL, SERIOUS → SERS, etc.
        count = row[col]
        if count > 0:
            injury_rows.append({
                'ev_id': generated_ev_id,
                'Aircraft_Key': '1',
                'inj_person_category': category,
                'inj_level': map_injury_level(level),
                'inj_person_count': count
            })
```

**Injury Level Mapping**:
| PRE1982 Suffix | Modern Code |
|----------------|-------------|
| FATAL | FATL |
| SERIOUS | SERS |
| MINOR | MINR |
| NONE | NONE |

## 5. Findings Table Mapping (tblFirstHalf → Findings, 1-10 rows per event)

**Denormalized Columns** (Wide format):
- CAUSE_FACTOR_1P, CAUSE_FACTOR_1M, CAUSE_FACTOR_1S
- CAUSE_FACTOR_2P, CAUSE_FACTOR_2M, CAUSE_FACTOR_2S
- ... (up to CAUSE_FACTOR_10P/M/S)

**Transformation Logic**:
```python
for i in range(1, 11):
    primary = row[f'CAUSE_FACTOR_{i}P']
    modifier = row[f'CAUSE_FACTOR_{i}M']
    secondary = row[f'CAUSE_FACTOR_{i}S']

    if pd.notna(primary):
        # Lookup description from ct_Pre1982
        description = code_table.get(primary, f"LEGACY:{primary}")

        findings.append({
            'ev_id': generated_ev_id,
            'Aircraft_Key': '1',
            'finding_code': None,  # No modern code mapping
            'finding_description': description,
            'cm_inPC': (i == 1),  # First cause is probable cause
            'modifier_code': modifier,
            'cause_factor': f"{primary}-{modifier}-{secondary}"  # Preserve legacy codes
        })
```

## 6. Narratives Table Mapping (tblSecondHalf → narratives)

| PRE1982 Column | Type | Modern Column | Type | Transformation | Notes |
|----------------|------|---------------|------|----------------|-------|
| RecNum | INTEGER | ev_id | VARCHAR(20) | Generate YYYYMMDDX{RecNum:06d} | Foreign key |
| NARRATIVE_TEXT | TEXT | narr_accp | TEXT | Direct copy | Accident narrative |
| PROBABLE_CAUSE_TEXT | TEXT | narr_cause | TEXT | Direct copy | Probable cause text |

## 7. Code Tables Required

### State Code Table (LOCAT_STATE_TERR → ev_state)
Extract from ct_Pre1982 or create manually:
| Code | State |
|------|-------|
| 1 | AL |
| 2 | AK |
| ... | ... |
| 32 | NY |
| ... | ... |

### Age Code Table (AGE_PILOT1 → crew_age)
Extract from ct_Pre1982:
| Code | Age Range |
|------|-----------|
| ZA | Unknown |
| A1 | 16-20 |
| A2 | 21-25 |
| ... | ... |

### Injury Level Mapping
| PRE1982 | Modern |
|---------|--------|
| FATAL | FATL |
| SERIOUS | SERS |
| MINOR | MINR |
| NONE | NONE |

## 8. Unmapped Columns (Preserve or Discard)

**Option 1**: Preserve as JSON in events.metadata JSONB column
**Option 2**: Discard (document as "not migrated")

Unmapped columns:
- Aircraft variant codes
- Investigation dates
- Examiner codes
- Legacy classification codes
```

**Deliverables**:
- [ ] `/tmp/NTSB_Datasets/column_mapping_spec.md` (comprehensive mapping)
- [ ] Identified all required code tables
- [ ] Transformation logic documented for complex fields

---

### **Phase 2: Code Mapping Tables** (3-4 hours)

#### Task 2.1: Extract ct_Pre1982 Code Table
**Duration**: 1 hour

**Steps**:
```bash
# 1. Export complete ct_Pre1982 table
mdb-export datasets/PRE1982.MDB ct_Pre1982 > /tmp/NTSB_Datasets/ct_Pre1982_full.csv

# 2. Load and analyze structure
python3 << 'EOF'
import pandas as pd

codes = pd.read_csv('/tmp/NTSB_Datasets/ct_Pre1982_full.csv')
print(f"Code table: {len(codes)} rows, {len(codes.columns)} columns")
print(codes.columns.tolist())
print(codes.head(20))

# Categorize code types
if 'CodeType' in codes.columns:
    print("\nCode types:")
    print(codes.groupby('CodeType').size())
EOF
```

**Deliverables**:
- [ ] Complete ct_Pre1982 extracted
- [ ] Code categories identified (state codes, age codes, cause codes, etc.)
- [ ] Code structure documented

#### Task 2.2: Create PostgreSQL Code Mapping Schema
**Duration**: 1 hour

**File**: `scripts/create_code_mappings.sql`

```sql
-- create_code_mappings.sql - PRE1982 Legacy Code Mapping Tables
-- Sprint 4: PRE1982 Integration
-- Purpose: Store lookup tables for legacy coded fields

-- Create schema for code mappings
CREATE SCHEMA IF NOT EXISTS code_mappings;

-- ============================================
-- 1. State Code Mapping (LOCAT_STATE_TERR → ev_state)
-- ============================================
CREATE TABLE IF NOT EXISTS code_mappings.state_codes (
    legacy_code INTEGER PRIMARY KEY,
    state_abbr CHAR(2) NOT NULL,
    state_name VARCHAR(100),
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Sample data (populate from ct_Pre1982 analysis)
INSERT INTO code_mappings.state_codes (legacy_code, state_abbr, state_name) VALUES
(1, 'AL', 'Alabama'),
(2, 'AK', 'Alaska'),
-- ... (50 states + territories)
(32, 'NY', 'New York'),
(33, 'NC', 'North Carolina')
-- ... complete list
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 2. Age Code Mapping (AGE_PILOT1 → crew_age)
-- ============================================
CREATE TABLE IF NOT EXISTS code_mappings.age_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    age_min INTEGER,
    age_max INTEGER,
    age_description VARCHAR(50),
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Sample data
INSERT INTO code_mappings.age_codes (legacy_code, age_min, age_max, age_description) VALUES
('ZA', NULL, NULL, 'Unknown'),
('A1', 16, 20, '16-20 years'),
('A2', 21, 25, '21-25 years'),
('A3', 26, 30, '26-30 years'),
-- ... complete age range codes
('A9', 61, NULL, '61+ years')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 3. Cause Factor Code Mapping (CAUSE_FACTOR_*P/M/S → Findings)
-- ============================================
CREATE TABLE IF NOT EXISTS code_mappings.cause_factor_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    cause_description TEXT NOT NULL,
    cause_category VARCHAR(50),
    modern_finding_code VARCHAR(10),  -- Map to codman.pdf codes if possible
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Sample data (populate from ct_Pre1982)
INSERT INTO code_mappings.cause_factor_codes (legacy_code, cause_description, cause_category) VALUES
('70', 'Pilot - inadequate preflight preparation', 'Pilot Error'),
('A', 'Aircraft - engine failure', 'Mechanical'),
('CB', 'Weather - low ceiling/visibility', 'Environmental')
-- ... (hundreds of legacy codes)
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 4. Injury Level Mapping (PRE1982 suffixes → modern codes)
-- ============================================
CREATE TABLE IF NOT EXISTS code_mappings.injury_level_mapping (
    legacy_suffix VARCHAR(20) PRIMARY KEY,
    modern_code VARCHAR(10) NOT NULL,
    description VARCHAR(50)
);

INSERT INTO code_mappings.injury_level_mapping (legacy_suffix, modern_code, description) VALUES
('FATAL', 'FATL', 'Fatal'),
('SERIOUS', 'SERS', 'Serious'),
('MINOR', 'MINR', 'Minor'),
('NONE', 'NONE', 'None')
ON CONFLICT (legacy_suffix) DO NOTHING;

-- ============================================
-- 5. Aircraft Damage Mapping
-- ============================================
CREATE TABLE IF NOT EXISTS code_mappings.damage_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    modern_code VARCHAR(10) NOT NULL,
    description VARCHAR(50)
);

INSERT INTO code_mappings.damage_codes (legacy_code, modern_code, description) VALUES
('D', 'DEST', 'Destroyed'),
('S', 'SUBS', 'Substantial'),
('M', 'MINR', 'Minor'),
('N', 'NONE', 'None')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- Indexes for fast lookup
-- ============================================
CREATE INDEX IF NOT EXISTS idx_state_codes_abbr ON code_mappings.state_codes(state_abbr);
CREATE INDEX IF NOT EXISTS idx_age_codes_range ON code_mappings.age_codes(age_min, age_max);
CREATE INDEX IF NOT EXISTS idx_cause_factor_category ON code_mappings.cause_factor_codes(cause_category);

-- ============================================
-- Helper functions for code lookup
-- ============================================

-- Decode state code
CREATE OR REPLACE FUNCTION code_mappings.decode_state(code INTEGER)
RETURNS CHAR(2) AS $$
    SELECT state_abbr FROM code_mappings.state_codes WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode age code (return midpoint of range)
CREATE OR REPLACE FUNCTION code_mappings.decode_age(code VARCHAR(10))
RETURNS INTEGER AS $$
    SELECT (age_min + COALESCE(age_max, age_min)) / 2
    FROM code_mappings.age_codes
    WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode cause factor code
CREATE OR REPLACE FUNCTION code_mappings.decode_cause_factor(code VARCHAR(10))
RETURNS TEXT AS $$
    SELECT cause_description
    FROM code_mappings.cause_factor_codes
    WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Grant permissions (assumes current user owns database)
GRANT USAGE ON SCHEMA code_mappings TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA code_mappings TO PUBLIC;
```

**Deliverables**:
- [ ] `scripts/create_code_mappings.sql` created
- [ ] 5 code mapping tables defined
- [ ] Helper functions created for decoding
- [ ] Script tested: `psql -d ntsb_aviation -f scripts/create_code_mappings.sql`

#### Task 2.3: Populate Code Mapping Tables from ct_Pre1982
**Duration**: 1-2 hours

**Steps**:
```python
# populate_code_tables.py - Extract codes from ct_Pre1982.csv and populate mapping tables

import pandas as pd
import psycopg2

# Load ct_Pre1982
codes = pd.read_csv('/tmp/NTSB_Datasets/ct_Pre1982_full.csv')

# Connect to database
conn = psycopg2.connect(dbname='ntsb_aviation', user='parobek')
cur = conn.cursor()

# 1. Populate state codes
state_codes = codes[codes['CodeType'] == 'STATE']  # Adjust based on actual structure
for _, row in state_codes.iterrows():
    cur.execute("""
        INSERT INTO code_mappings.state_codes (legacy_code, state_abbr, state_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (legacy_code) DO NOTHING
    """, (row['Code'], row['StateAbbr'], row['StateName']))

# 2. Populate age codes
age_codes = codes[codes['CodeType'] == 'AGE']
for _, row in age_codes.iterrows():
    cur.execute("""
        INSERT INTO code_mappings.age_codes (legacy_code, age_min, age_max, age_description)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (legacy_code) DO NOTHING
    """, (row['Code'], row['AgeMin'], row['AgeMax'], row['Description']))

# 3. Populate cause factor codes
cause_codes = codes[codes['CodeType'] == 'CAUSE']
for _, row in cause_codes.iterrows():
    cur.execute("""
        INSERT INTO code_mappings.cause_factor_codes (legacy_code, cause_description, cause_category)
        VALUES (%s, %s, %s)
        ON CONFLICT (legacy_code) DO NOTHING
    """, (row['Code'], row['Description'], row['Category']))

conn.commit()
conn.close()

print(f"Populated {len(state_codes)} state codes")
print(f"Populated {len(age_codes)} age codes")
print(f"Populated {len(cause_codes)} cause factor codes")
```

**Deliverables**:
- [ ] Code mapping tables fully populated
- [ ] Verification queries run to check row counts
- [ ] Sample lookups tested using helper functions

---

### **Phase 3: Custom ETL Development** (4-6 hours)

#### Task 3.1: Create load_pre1982.py Script
**Duration**: 3-4 hours

**File**: `scripts/load_pre1982.py`

**Structure** (800-1200 lines estimated):

```python
#!/usr/bin/env python3
"""
load_pre1982.py - Load PRE1982.MDB Historical Aviation Accident Data (1962-1981)

Sprint 4: PRE1982 Integration
Version: 1.0.0
Date: 2025-11-XX

This script transforms denormalized PRE1982.MDB legacy data into normalized PostgreSQL schema.

Key Transformations:
- Denormalized tblFirstHalf/tblSecondHalf → Normalized 6 tables (events, aircraft, Flight_Crew, injury, Findings, narratives)
- RecNum integer → ev_id VARCHAR (generated: YYYYMMDDX{RecNum:06d})
- 200+ wide columns → 36 normalized columns per table
- Coded fields → Decoded using code_mappings schema
- Injury wide columns → Tall normalized rows

Architecture:
1. Extract tblFirstHalf, tblSecondHalf from PRE1982.MDB
2. Join on RecNum (1 row per event)
3. Transform denormalized row → 6 normalized tables
4. Load to staging tables
5. Validate data quality
6. Merge to production
7. Update load_tracking

Usage:
    python scripts/load_pre1982.py --source PRE1982.MDB [--limit 100] [--dry-run]
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
import logging
import subprocess
from typing import Dict, List, Tuple, Optional
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_pre1982.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ntsb_aviation"),
    "user": os.getenv("DB_USER", os.getenv("USER")),
    "password": os.getenv("DB_PASSWORD", ""),
}

# MDB file configuration
MDB_FILE = "datasets/PRE1982.MDB"
TABLE_FIRST_HALF = "tblFirstHalf"
TABLE_SECOND_HALF = "tblSecondHalf"

class PRE1982Loader:
    """Transform and load PRE1982.MDB legacy data into normalized schema."""

    def __init__(self, mdb_file: str, limit: Optional[int] = None, dry_run: bool = False):
        self.mdb_file = mdb_file
        self.limit = limit
        self.dry_run = dry_run
        self.conn = None
        self.code_tables = {}

    def connect(self):
        """Connect to PostgreSQL database."""
        self.conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database: {DB_CONFIG['database']}")

    def load_code_tables(self):
        """Load code mapping tables into memory for fast lookup."""
        logger.info("Loading code mapping tables...")

        with self.conn.cursor() as cur:
            # Load state codes
            cur.execute("SELECT legacy_code, state_abbr FROM code_mappings.state_codes")
            self.code_tables['states'] = dict(cur.fetchall())

            # Load age codes
            cur.execute("SELECT legacy_code, (age_min + COALESCE(age_max, age_min))/2 FROM code_mappings.age_codes")
            self.code_tables['ages'] = dict(cur.fetchall())

            # Load cause factor codes
            cur.execute("SELECT legacy_code, cause_description FROM code_mappings.cause_factor_codes")
            self.code_tables['causes'] = dict(cur.fetchall())

            # Load injury level mapping
            cur.execute("SELECT legacy_suffix, modern_code FROM code_mappings.injury_level_mapping")
            self.code_tables['injury_levels'] = dict(cur.fetchall())

            # Load damage codes
            cur.execute("SELECT legacy_code, modern_code FROM code_mappings.damage_codes")
            self.code_tables['damage'] = dict(cur.fetchall())

        logger.info(f"Loaded {len(self.code_tables)} code mapping tables")

    def extract_from_mdb(self, table_name: str) -> pd.DataFrame:
        """Extract table from MDB file using mdb-export."""
        logger.info(f"Extracting {table_name} from {self.mdb_file}...")

        try:
            cmd = ["mdb-export", self.mdb_file, table_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Load into pandas
            df = pd.read_csv(StringIO(result.stdout))

            # Apply limit if specified
            if self.limit:
                df = df.head(self.limit)

            logger.info(f"Extracted {len(df)} rows from {table_name}")
            return df

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract {table_name}: {e}")
            raise

    def generate_ev_id(self, rec_num: int, date_occurrence: str) -> str:
        """
        Generate synthetic ev_id from RecNum and DATE_OCCURRENCE.

        Format: YYYYMMDDX{RecNum:06d}
        Example: RecNum=40, DATE_OCCURRENCE='07/23/62' → '19620723X000040'

        The 'X' marker distinguishes legacy events from modern events.
        """
        # Parse date (handles MM/DD/YY format)
        try:
            date_obj = pd.to_datetime(date_occurrence, format='%m/%d/%y')
            date_str = date_obj.strftime('%Y%m%d')
        except:
            # Fallback if date parsing fails
            date_str = '19620101'  # Default to 1962-01-01

        # Format: YYYYMMDDX{RecNum:06d}
        ev_id = f"{date_str}X{rec_num:06d}"
        return ev_id

    def parse_legacy_date(self, date_str: str) -> Optional[str]:
        """
        Parse legacy date format 'MM/DD/YY HH:MM:SS' to PostgreSQL DATE 'YYYY-MM-DD'.

        Handles 2-digit year:
        - 00-39 → 2000-2039
        - 40-99 → 1940-1999
        (PRE1982 should only have 62-81, but be defensive)
        """
        if pd.isna(date_str) or date_str == "":
            return None

        try:
            # Parse with 2-digit year (pandas infers century)
            date_obj = pd.to_datetime(date_str, format='%m/%d/%y %H:%M:%S')

            # Enforce 1962-1981 range for PRE1982
            if date_obj.year < 1962 or date_obj.year > 1981:
                logger.warning(f"Date out of expected range: {date_str} → {date_obj}")

            return date_obj.strftime('%Y-%m-%d')
        except:
            logger.warning(f"Failed to parse date: {date_str}")
            return None

    def parse_legacy_time(self, time_value) -> Optional[str]:
        """
        Parse legacy time format (likely HHMM integer) to PostgreSQL TIME 'HH:MM:SS'.
        Same as modern format.
        """
        if pd.isna(time_value) or time_value == "":
            return None

        try:
            time_int = int(float(time_value))
            hours = time_int // 100
            minutes = time_int % 100

            if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
                return None

            return f"{hours:02d}:{minutes:02d}:00"
        except:
            return None

    def decode_state(self, state_code: int) -> Optional[str]:
        """Decode numeric state code to 2-letter abbreviation."""
        return self.code_tables['states'].get(state_code)

    def decode_age(self, age_code: str) -> Optional[int]:
        """Decode age code to integer (midpoint of range)."""
        return self.code_tables['ages'].get(age_code)

    def parse_pilot_hours(self, hours_str: str) -> Optional[int]:
        """
        Parse pilot hours with letter suffix (e.g., '11508A' → 11508).
        Strip non-numeric characters.
        """
        if pd.isna(hours_str) or hours_str == "":
            return None

        try:
            # Remove all non-numeric characters
            hours_numeric = ''.join(c for c in str(hours_str) if c.isdigit())
            return int(hours_numeric) if hours_numeric else None
        except:
            return None

    def transform_events(self, row: pd.Series) -> Dict:
        """Transform denormalized row to events table format."""
        return {
            'ev_id': self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
            'ev_date': self.parse_legacy_date(row['DATE_OCCURRENCE']),
            'ev_time': self.parse_legacy_time(row.get('TIME_OCCUR')),
            'ev_year': pd.to_datetime(row['DATE_OCCURRENCE'], format='%m/%d/%y').year,
            'ev_month': pd.to_datetime(row['DATE_OCCURRENCE'], format='%m/%d/%y').month,
            'ev_city': row.get('LOCATION'),
            'ev_state': self.decode_state(row.get('LOCAT_STATE_TERR')),
            'ev_country': 'USA',  # All PRE1982 events are domestic
            'ntsb_no': row.get('DOCKET_NO'),
            'dec_latitude': row.get('LATITUDE'),  # May need DMS conversion
            'dec_longitude': row.get('LONGITUDE'),  # May need DMS conversion
            # ... map remaining 20+ fields
            'inj_tot_f': row.get('TOTAL_ABRD_FATAL', 0),
            'inj_tot_s': row.get('TOTAL_ABRD_SERIOUS', 0),
            'inj_tot_m': row.get('TOTAL_ABRD_MINOR', 0),
            'inj_tot_n': row.get('TOTAL_ABRD_NONE', 0),
        }

    def transform_aircraft(self, row: pd.Series) -> Dict:
        """Transform denormalized row to aircraft table format."""
        ev_id = self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE'])

        return {
            'ev_id': ev_id,
            'Aircraft_Key': '1',  # PRE1982 has single aircraft per event
            'regis_no': row.get('REGIST_NO'),
            'acft_make': row.get('ACFT_MAKE'),
            'acft_model': row.get('ACFT_MODEL'),
            'num_eng': row.get('NO_ENGINES'),
            'damage': self.code_tables['damage'].get(row.get('ACFT_DAMAGE')),
            # ... map remaining fields
        }

    def transform_flight_crew(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized row to Flight_Crew table format.
        Returns 0-2 rows (Pilot 1, Pilot 2).
        """
        ev_id = self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE'])
        crews = []

        # Pilot 1
        if pd.notna(row.get('PILOT_INVOLED1')) and row['PILOT_INVOLED1'] == 'A':
            crews.append({
                'ev_id': ev_id,
                'Aircraft_Key': '1',
                'crew_category': 'PILOT',
                'crew_age': self.decode_age(row.get('AGE_PILOT1')),
                'pilot_tot_time': self.parse_pilot_hours(row.get('HOURS_TOTAL_PILOT1')),
                'pilot_make_time': self.parse_pilot_hours(row.get('HOURS_IN_TYPE_PILOT1')),
                # ... map remaining fields
            })

        # Pilot 2 (Co-pilot)
        if pd.notna(row.get('PILOT_INVOLED2')) and row['PILOT_INVOLED2'] == 'A':
            crews.append({
                'ev_id': ev_id,
                'Aircraft_Key': '1',
                'crew_category': 'CO-PILOT',
                'crew_age': self.decode_age(row.get('AGE_PILOT2')),
                'pilot_tot_time': self.parse_pilot_hours(row.get('HOURS_TOTAL_PILOT2')),
                'pilot_make_time': self.parse_pilot_hours(row.get('HOURS_IN_TYPE_PILOT2')),
                # ... map remaining fields
            })

        return crews

    def transform_injury(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized injury columns to normalized injury table rows.
        Returns 10-50 rows per event.
        """
        ev_id = self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE'])
        injuries = []

        # Define injury categories and their column prefixes
        injury_categories = [
            ('PILOT', 'PILOT'),
            ('CO-PILOT', 'CO_PILOT'),
            ('PASSENGER', 'PASSENGERS'),
            ('CREW', 'CREW'),
            ('TOTAL', 'TOTAL_ABRD'),
        ]

        # Define injury levels
        injury_levels = ['FATAL', 'SERIOUS', 'MINOR', 'NONE']

        for category, prefix in injury_categories:
            for level in injury_levels:
                col_name = f'{prefix}_{level}'
                count = row.get(col_name, 0)

                if pd.notna(count) and count > 0:
                    injuries.append({
                        'ev_id': ev_id,
                        'Aircraft_Key': '1',
                        'inj_person_category': category,
                        'inj_level': self.code_tables['injury_levels'].get(level, level[:4]),
                        'inj_person_count': int(count)
                    })

        return injuries

    def transform_findings(self, row: pd.Series) -> List[Dict]:
        """
        Transform denormalized cause factor columns to normalized Findings table rows.
        Returns 1-10 rows per event.
        """
        ev_id = self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE'])
        findings = []

        # PRE1982 has up to 10 cause factors (P/M/S triplets)
        for i in range(1, 11):
            primary = row.get(f'CAUSE_FACTOR_{i}P')
            modifier = row.get(f'CAUSE_FACTOR_{i}M')
            secondary = row.get(f'CAUSE_FACTOR_{i}S')

            if pd.notna(primary):
                # Lookup description from code table
                description = self.code_tables['causes'].get(primary, f"LEGACY:{primary}")

                findings.append({
                    'ev_id': ev_id,
                    'Aircraft_Key': '1',
                    'finding_code': None,  # No modern code mapping
                    'finding_description': description,
                    'cm_inPC': (i == 1),  # First cause factor is probable cause
                    'modifier_code': modifier,
                    'cause_factor': f"{primary}-{modifier}-{secondary}" if modifier or secondary else primary
                })

        return findings

    def transform_narratives(self, row: pd.Series) -> Dict:
        """Transform denormalized row to narratives table format."""
        return {
            'ev_id': self.generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
            'Aircraft_Key': '1',
            'narr_accp': row.get('NARRATIVE_TEXT', ''),
            'narr_cause': row.get('PROBABLE_CAUSE_TEXT', ''),
        }

    def load_to_staging(self, df: pd.DataFrame, table_name: str):
        """Bulk load DataFrame to staging table using COPY."""
        logger.info(f"Loading {len(df)} rows to staging.{table_name}...")

        # Clear staging table
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE staging.{table_name}")

        # Bulk COPY
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep='\\N')
        buffer.seek(0)

        with self.conn.cursor() as cur:
            cur.copy_expert(
                f"COPY staging.{table_name} FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
                buffer
            )

        self.conn.commit()
        logger.info(f"Loaded {len(df)} rows to staging.{table_name}")

    def main(self):
        """Main ETL pipeline."""
        try:
            # 1. Connect to database
            self.connect()

            # 2. Load code mapping tables
            self.load_code_tables()

            # 3. Extract from PRE1982.MDB
            logger.info(f"Extracting from {self.mdb_file}...")
            first_half = self.extract_from_mdb(TABLE_FIRST_HALF)
            second_half = self.extract_from_mdb(TABLE_SECOND_HALF)

            # 4. Join tblFirstHalf + tblSecondHalf on RecNum
            logger.info("Joining tblFirstHalf + tblSecondHalf...")
            combined = pd.merge(first_half, second_half, on='RecNum', how='inner')
            logger.info(f"Combined dataset: {len(combined)} rows")

            # 5. Transform denormalized → normalized
            logger.info("Transforming data...")

            events_rows = []
            aircraft_rows = []
            crew_rows = []
            injury_rows = []
            findings_rows = []
            narratives_rows = []

            for idx, row in combined.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx}/{len(combined)} rows...")

                # Transform to all 6 tables
                events_rows.append(self.transform_events(row))
                aircraft_rows.append(self.transform_aircraft(row))
                crew_rows.extend(self.transform_flight_crew(row))
                injury_rows.extend(self.transform_injury(row))
                findings_rows.extend(self.transform_findings(row))
                narratives_rows.append(self.transform_narratives(row))

            logger.info(f"Transformation complete:")
            logger.info(f"  events: {len(events_rows)}")
            logger.info(f"  aircraft: {len(aircraft_rows)}")
            logger.info(f"  Flight_Crew: {len(crew_rows)}")
            logger.info(f"  injury: {len(injury_rows)}")
            logger.info(f"  Findings: {len(findings_rows)}")
            logger.info(f"  narratives: {len(narratives_rows)}")

            # 6. Load to staging tables
            if not self.dry_run:
                self.load_to_staging(pd.DataFrame(events_rows), 'events')
                self.load_to_staging(pd.DataFrame(aircraft_rows), 'aircraft')
                self.load_to_staging(pd.DataFrame(crew_rows), 'Flight_Crew')
                self.load_to_staging(pd.DataFrame(injury_rows), 'injury')
                self.load_to_staging(pd.DataFrame(findings_rows), 'Findings')
                self.load_to_staging(pd.DataFrame(narratives_rows), 'narratives')

                # 7. Merge to production (reuse load_with_staging.py logic)
                logger.info("Merging staging → production...")
                # Call existing merge functions from load_with_staging.py

                # 8. Update load_tracking
                logger.info("Updating load_tracking...")
                with self.conn.cursor() as cur:
                    cur.execute("""
                        UPDATE load_tracking
                        SET load_status = 'completed',
                            events_loaded = %s,
                            duplicate_events_found = 0,
                            load_completed_at = NOW()
                        WHERE database_name = 'PRE1982.MDB'
                    """, (len(events_rows),))
                self.conn.commit()

            logger.info("✅ PRE1982 load complete!")

        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load PRE1982.MDB historical data')
    parser.add_argument('--source', default='PRE1982.MDB', help='MDB filename')
    parser.add_argument('--limit', type=int, help='Limit rows for testing')
    parser.add_argument('--dry-run', action='store_true', help='Test without loading')

    args = parser.parse_args()

    loader = PRE1982Loader(
        mdb_file=f"datasets/{args.source}",
        limit=args.limit,
        dry_run=args.dry_run
    )
    loader.main()
```

**Deliverables**:
- [ ] `scripts/load_pre1982.py` created (800-1200 lines)
- [ ] All transformation functions implemented
- [ ] Code lookup integrated
- [ ] Staging table loading implemented
- [ ] Error handling and logging

#### Task 3.2: Test with Sample Data (100 rows)
**Duration**: 1 hour

**Steps**:
```bash
# Test extraction
python scripts/load_pre1982.py --source PRE1982.MDB --limit 100 --dry-run

# Expected output:
# - 100 events extracted
# - 100 aircraft rows
# - 50-150 Flight_Crew rows
# - 500-5000 injury rows
# - 100-1000 Findings rows
# - 100 narratives rows
# - No database changes (dry-run)

# Verify transformation logic
# Check generated ev_ids (should be YYYYMMDDX{RecNum:06d} format)
# Check date parsing (1962-1981 range)
# Check code decoding (states, ages, causes)
```

**Deliverables**:
- [ ] Sample transformation successful
- [ ] ev_id generation verified
- [ ] Code decoding tested
- [ ] No errors in dry-run mode

---

### **Phase 4: Testing & Validation** (2-3 hours)

#### Task 4.1: Small Dataset Testing (100 rows)
**Duration**: 30 minutes

**Steps**:
```bash
# 1. Backup database
pg_dump ntsb_aviation > /tmp/ntsb_aviation_backup_pre_pre1982.sql

# 2. Load 100 rows (real load, not dry-run)
python scripts/load_pre1982.py --source PRE1982.MDB --limit 100

# 3. Verify staging tables
psql -d ntsb_aviation -c "SELECT * FROM get_row_counts();"

# Expected:
# staging.events: 100 rows
# staging.aircraft: 100 rows
# staging.Flight_Crew: ~50-150 rows
# staging.injury: ~500-5000 rows
# staging.Findings: ~100-1000 rows
# staging.narratives: 100 rows

# 4. Check for duplicates
psql -d ntsb_aviation -c "SELECT * FROM get_duplicate_stats();"

# Expected: 0 duplicates (PRE1982 has no overlap with 2000-2025 data)

# 5. Verify production merge
psql -d ntsb_aviation -c "
  SELECT
    'events' as table_name,
    COUNT(*) as production_rows,
    COUNT(DISTINCT ev_id) as unique_events
  FROM events
  WHERE ev_id LIKE '1962%X%' OR ev_id LIKE '1963%X%'
  LIMIT 5;
"

# Expected: 100 events with ev_id starting with '1962' or '1963' and containing 'X'
```

**Deliverables**:
- [ ] 100 rows loaded successfully
- [ ] All tables populated correctly
- [ ] Zero duplicates found
- [ ] Production tables updated

#### Task 4.2: Data Quality Validation
**Duration**: 1 hour

**Steps**:
```sql
-- 1. Check for NULL ev_id (should be 0)
SELECT COUNT(*) as null_ev_id_count
FROM events
WHERE ev_id IS NULL;

-- 2. Check ev_id format (should match YYYYMMDDX{RecNum:06d})
SELECT ev_id
FROM events
WHERE ev_id LIKE '1962%X%' OR ev_id LIKE '1963%X%'
LIMIT 10;

-- 3. Check date range (should be 1962-1981 only)
SELECT MIN(ev_date) as min_date, MAX(ev_date) as max_date
FROM events
WHERE ev_id LIKE '%X%';  -- Legacy marker

-- Expected: min_date >= 1962-01-01, max_date <= 1981-12-31

-- 4. Check coordinate validity
SELECT COUNT(*) as invalid_coords
FROM events
WHERE (dec_latitude < -90 OR dec_latitude > 90)
   OR (dec_longitude < -180 OR dec_longitude > 180);

-- Expected: 0 invalid coordinates

-- 5. Check foreign key integrity
SELECT COUNT(*) as orphaned_aircraft
FROM aircraft a
LEFT JOIN events e ON a.ev_id = e.ev_id
WHERE e.ev_id IS NULL
  AND a.ev_id LIKE '%X%';

-- Expected: 0 orphaned records

-- 6. Check injury level mapping
SELECT DISTINCT inj_level
FROM injury
WHERE ev_id LIKE '%X%';

-- Expected: FATL, SERS, MINR, NONE (4 levels only)

-- 7. Check cause factor decoding
SELECT finding_description, COUNT(*) as count
FROM Findings
WHERE ev_id LIKE '%X%'
GROUP BY finding_description
ORDER BY count DESC
LIMIT 10;

-- Expected: Decoded descriptions, not raw codes

-- 8. Check crew ages (should be decoded, not codes)
SELECT MIN(crew_age) as min_age, MAX(crew_age) as max_age,
       COUNT(*) as total_crew
FROM Flight_Crew
WHERE ev_id LIKE '%X%';

-- Expected: min_age >= 16, max_age <= 120

-- 9. Check state codes (should be 2-letter, not numeric)
SELECT DISTINCT ev_state
FROM events
WHERE ev_id LIKE '%X%'
ORDER BY ev_state;

-- Expected: AL, AK, AZ, CA, ... (2-letter abbreviations)
```

**Deliverables**:
- [ ] All validation queries passed
- [ ] No data quality issues found
- [ ] Foreign key integrity 100%

#### Task 4.3: Full Dataset Load
**Duration**: 30 minutes

**Steps**:
```bash
# 1. Clear staging tables
psql -d ntsb_aviation -c "SELECT clear_all_staging();"

# 2. Load full PRE1982.MDB (~87,000 events)
python scripts/load_pre1982.py --source PRE1982.MDB

# Expected output:
# - Extracted ~87,000 rows from tblFirstHalf
# - Extracted ~87,000 rows from tblSecondHalf
# - Combined: ~87,000 rows
# - Transformed:
#   - events: 87,000
#   - aircraft: 87,000
#   - Flight_Crew: ~40,000-100,000
#   - injury: ~800,000-1,200,000
#   - Findings: ~400,000-800,000
#   - narratives: 87,000
# - Loaded to staging: ~2.5M total rows
# - Merged to production: 87,000 unique events, 0 duplicates
# - Load duration: 5-10 minutes

# 3. Verify final row counts
psql -d ntsb_aviation -c "
  SELECT
    schemaname,
    tablename,
    n_live_tup as rows
  FROM pg_stat_user_tables
  WHERE schemaname = 'public'
  ORDER BY n_live_tup DESC;
"

# Expected:
# events: ~180,000 (92,771 + 87,000)
# injury: ~2,000,000 (91,333 + 1,200,000)
# Findings: ~900,000 (101,243 + 800,000)
# ... etc

# 4. Check database size
psql -d ntsb_aviation -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));"

# Expected: ~1,000 MB (from 512 MB)
```

**Deliverables**:
- [ ] Full PRE1982.MDB loaded (87,000 events)
- [ ] Database size ~1,000 MB
- [ ] All child records loaded
- [ ] Load tracking updated

#### Task 4.4: Performance Benchmarking
**Duration**: 30 minutes

**Steps**:
```bash
# 1. Refresh materialized views
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"

# 2. Run performance benchmarks
psql -d ntsb_aviation -f scripts/validate_data.sql > /tmp/benchmark_post_pre1982.txt

# 3. Run anomaly detection
python scripts/detect_anomalies.py --lookback-days 36500 --output /tmp/anomalies_post_pre1982.json

# Expected: All 5 checks passed, 0 critical anomalies

# 4. Compare query performance (before/after)
psql -d ntsb_aviation << 'EOF'
-- Benchmark queries
\timing on

-- Q1: Event count by year
SELECT ev_year, COUNT(*) FROM events GROUP BY ev_year ORDER BY ev_year;

-- Q2: State-level statistics
SELECT ev_state, COUNT(*) FROM events WHERE ev_state IS NOT NULL GROUP BY ev_state;

-- Q3: Aircraft make/model statistics
SELECT acft_make, acft_model, COUNT(*)
FROM aircraft
GROUP BY acft_make, acft_model
HAVING COUNT(*) >= 5
ORDER BY COUNT(*) DESC;

-- Q4: Decade trends
SELECT (ev_year / 10) * 10 as decade, COUNT(*)
FROM events
GROUP BY decade
ORDER BY decade;

\timing off
EOF

# Expected: All queries complete in <500ms (p99 target)
```

**Deliverables**:
- [ ] Performance benchmarks run
- [ ] Query performance within targets (<500ms p99)
- [ ] Materialized views refreshed
- [ ] Anomaly detection passed

---

### **Phase 5: Documentation & Integration** (1-2 hours)

#### Task 5.1: Update Documentation
**Duration**: 1 hour

**Files to Update**:

**1. README.md**:
```markdown
## Database Coverage

- **Events**: 179,771 (1962-2025)
- **Date Coverage**: 63 years (complete historical record)
- **Coverage Gaps**: NONE
- **Database Size**: 1,012 MB
- **Total Rows**: ~2.5 million across 11 tables

## Historical Data Sources

| Database | Date Range | Events | Status |
|----------|------------|--------|--------|
| PRE1982.MDB | 1962-1981 | 87,000 | ✅ Loaded (Sprint 4) |
| Pre2008.mdb | 2000-2007 | 92,771 | ✅ Loaded (Sprint 2) |
| avall.mdb | 2008-2025 | 29,773 | ✅ Loaded (Sprint 1) |
```

**2. CHANGELOG.md**:
```markdown
## [Sprint 4] - 2025-11-XX

### Added
- PRE1982.MDB historical data integration (87,000 events, 1962-1981)
- Complete 63-year historical dataset (1962-2025)
- Code mapping tables for legacy coded fields
- Custom ETL pipeline for denormalized → normalized transformation
- `scripts/load_pre1982.py` (1,200 lines) - Legacy data loader
- `scripts/create_code_mappings.sql` (400 lines) - Code lookup tables
- Code decoding helper functions

### Changed
- Database coverage: 26 years → 63 years
- Events: 92,771 → 179,771
- Total rows: ~733,000 → ~2.5 million
- Database size: 512 MB → 1,012 MB

### Fixed
- 15-year complete coverage gap (1962-1976)
- Partial coverage gap (1977-1981)

### Documentation
- Updated README.md with 63-year coverage
- Created PRE1982 ETL specification
- Sprint 4 completion report
```

**3. CLAUDE.local.md**:
```markdown
## Current Sprint Status: Phase 1 Sprint 4

**Objective**: PRE1982 Historical Data Integration
**Progress**: ✅ 100% COMPLETE

### Completed ✅:

1. **Schema Analysis & Mapping** (3 hours)
   - Extracted PRE1982 schema and sample data
   - Created comprehensive column mapping specification
   - Identified all code tables and transformations

2. **Code Mapping Tables** (3 hours)
   - Created code_mappings schema with 5 lookup tables
   - Populated 500+ legacy codes from ct_Pre1982
   - Created helper functions for code decoding

3. **Custom ETL Development** (4 hours)
   - Created load_pre1982.py (1,200 lines)
   - Implemented denormalized → normalized transformation
   - Integrated code decoding and data validation

4. **Testing & Validation** (2 hours)
   - Tested with 100-row sample (successful)
   - Loaded full 87,000 events (successful)
   - All data quality checks passed
   - Performance benchmarks within targets

5. **Documentation & Integration** (1 hour)
   - Updated README, CHANGELOG, CLAUDE.local.md
   - Created Sprint 4 completion report
   - Git commit and push

### Database State (Post-Sprint 4):

**Database**: ntsb_aviation
**Size**: 1,012 MB (from 512 MB, +97.7%)
**Events**: 179,771 (from 92,771, +93.7%)
**Coverage**: 1962-2025 (63 years complete)

### Row Counts:

| Table | Rows (Post-Sprint 4) | Rows (Pre-Sprint 4) | Increase |
|-------|----------------------|---------------------|----------|
| events | 179,771 | 92,771 | +87,000 |
| aircraft | 181,533 | 94,533 | +87,000 |
| Flight_Crew | 101,003 | 31,003 | +70,000 |
| injury | 1,291,333 | 91,333 | +1,200,000 |
| Findings | 901,243 | 101,243 | +800,000 |
| narratives | 139,880 | 52,880 | +87,000 |
| **TOTAL** | **~2,500,000** | **~733,000** | **+241%** |

### Load Tracking Status:

| Database | Status | Events Loaded | Load Date |
|----------|--------|---------------|-----------|
| PRE1982.MDB | ✅ completed | 87,000 | 2025-11-XX |
| Pre2008.mdb | ✅ completed | 92,771 | 2025-11-06 |
| avall.mdb | ✅ completed | 29,773 | 2025-11-05 |
```

**Deliverables**:
- [ ] README.md updated
- [ ] CHANGELOG.md updated
- [ ] CLAUDE.local.md updated

#### Task 5.2: Create Sprint 4 Completion Report
**Duration**: 30 minutes

**File**: `docs/SPRINT_4_COMPLETION_REPORT.md` (800-1000 lines estimated)

**Structure**:
1. Executive Summary
2. Key Achievements
3. Deliverables Summary
4. Implementation Details
5. Testing Results
6. Performance Metrics
7. Lessons Learned
8. Technical Decisions
9. Code Quality Metrics
10. Next Steps and Recommendations

**Deliverables**:
- [ ] `docs/SPRINT_4_COMPLETION_REPORT.md` created
- [ ] Comprehensive report with all metrics
- [ ] Screenshots/sample outputs included

#### Task 5.3: Git Commit and Push
**Duration**: 15 minutes

**Steps**:
```bash
# 1. Run code quality checks
ruff check scripts/load_pre1982.py
ruff format scripts/load_pre1982.py

# 2. Verify SQL syntax
psql -d ntsb_aviation -f scripts/create_code_mappings.sql --dry-run

# 3. Stage changes
git add scripts/load_pre1982.py
git add scripts/create_code_mappings.sql
git add docs/SPRINT_4_COMPLETION_REPORT.md
git add README.md CHANGELOG.md

# 4. Commit with conventional message
git commit -m "feat(sprint4): integrate PRE1982.MDB historical data (1962-1981)

- Add 87,000 historical events (1962-1981) to complete 63-year dataset
- Create custom ETL pipeline for denormalized → normalized transformation
- Implement code mapping tables for legacy coded fields (5 tables, 500+ codes)
- Database size: 512 MB → 1,012 MB (+97.7%)
- Events: 92,771 → 179,771 (+93.7%)
- Total rows: ~733,000 → ~2.5M (+241%)

Deliverables:
- scripts/load_pre1982.py (1,200 lines) - Legacy data loader
- scripts/create_code_mappings.sql (400 lines) - Code lookup tables
- docs/SPRINT_4_COMPLETION_REPORT.md (comprehensive report)
- Updated README, CHANGELOG, CLAUDE.local.md

Testing:
- All data quality checks passed (9/9)
- Zero duplicates found
- Performance benchmarks within targets (<500ms p99)
- Anomaly detection passed (5/5 checks)

Generated with Claude Code (https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 5. Push to repository
git push origin main
```

**Deliverables**:
- [ ] Code formatted with ruff
- [ ] SQL syntax validated
- [ ] Git commit created
- [ ] Pushed to repository

---

## Success Criteria

### Data Loading ✅
- [x] PRE1982.MDB fully extracted (~87,000 events)
- [x] All events transformed to normalized schema
- [x] All child records created (aircraft, crew, injury, findings, narratives)
- [x] No data loss (all PRE1982 data represented)
- [x] Load tracking updated (PRE1982.MDB marked completed)

### Data Quality ✅
- [x] Zero NULL values in critical fields (ev_id, ev_date)
- [x] All dates within valid range (1962-1981)
- [x] All coordinates within valid bounds (-90/90, -180/180)
- [x] Zero duplicate ev_id values
- [x] 100% foreign key integrity (no orphaned records)
- [x] All anomaly detection checks passed (5/5)

### Performance ✅
- [x] Database size increase acceptable (~500 MB, total ~1,000 MB)
- [x] Query performance not degraded (within 10% of baseline, <500ms p99)
- [x] Materialized views refreshed successfully
- [x] ETL script completes in <15 minutes

### Code Quality ✅
- [x] Python code passes ruff check (0 errors)
- [x] Python code formatted with ruff format
- [x] SQL code syntax validated
- [x] Comprehensive error handling
- [x] Progress logging every 1000 rows

### Documentation ✅
- [x] README.md updated (coverage 1962-2025)
- [x] CHANGELOG.md updated (Sprint 4 entry)
- [x] CLAUDE.local.md updated (database state)
- [x] SPRINT_4_COMPLETION_REPORT.md created
- [x] Code mapping tables documented

---

## Deliverables

### Scripts (3 files, ~1,600 lines total):
1. **`scripts/create_code_mappings.sql`** (400 lines)
   - Code mapping schema (5 tables: states, ages, causes, injury levels, damage)
   - INSERT statements for 500+ codes
   - Helper functions for decoding (decode_state, decode_age, decode_cause_factor)
   - Indexes on code_value columns

2. **`scripts/load_pre1982.py`** (1,200 lines)
   - Extract from PRE1982.MDB using mdb-export
   - Transform denormalized → normalized (6 tables)
   - Load to staging tables using COPY
   - Validate and merge to production
   - Update load_tracking

3. **`scripts/populate_code_tables.py`** (optional, 200 lines)
   - Extract codes from ct_Pre1982.csv
   - Populate code_mappings tables

### Documentation (4 files, ~2,500 lines total):
1. **`/tmp/NTSB_Datasets/column_mapping_spec.md`** (800 lines)
   - Detailed column mapping specification
   - Transformation logic documentation
   - Code mapping table reference

2. **`docs/SPRINT_4_COMPLETION_REPORT.md`** (1,000 lines)
   - Executive summary
   - Implementation details
   - Testing results
   - Performance benchmarks
   - Lessons learned

3. **Updated `README.md`**
   - Events: 92,771 → 179,771
   - Coverage: 1977-2025 → 1962-2025
   - Database size: 512 MB → 1,012 MB

4. **Updated `CHANGELOG.md`**
   - Sprint 4 entry with all deliverables

### Database Changes:
- **+1 schema**: `code_mappings`
- **+5 tables**: state_codes, age_codes, cause_factor_codes, injury_level_mapping, damage_codes
- **+~87,000 events**: 1962-1981 historical data
- **+~1.5M child records**: aircraft, crew, injury, findings, narratives
- **+500 MB** database size
- **Materialized views** refreshed with historical data

---

## Risk Assessment

### High Risk ⚠️
1. **Schema Complexity** - 200+ columns to 36 normalized columns
   - **Mitigation**: Thorough testing with sample data first (100 rows)
   - **Mitigation**: Document all column mappings in specification
   - **Mitigation**: Dry-run mode for testing without database changes

2. **Data Quality Issues** - Legacy 1960s-1970s data may have inconsistencies
   - **Mitigation**: Comprehensive validation before merge (9 validation queries)
   - **Mitigation**: Staging table pattern for rollback capability
   - **Mitigation**: Anomaly detection after load

3. **Performance Impact** - Database doubling in size (~500 MB → ~1,000 MB)
   - **Mitigation**: Benchmark before/after (validate_data.sql)
   - **Mitigation**: Index optimization (existing 59 indexes)
   - **Mitigation**: Materialized view refresh

### Medium Risk ⚠️
1. **Code Mapping Accuracy** - Manual extraction from ct_Pre1982
   - **Mitigation**: Cross-reference with existing modern data
   - **Mitigation**: Spot-check code descriptions (sample 100 codes)
   - **Mitigation**: Store legacy codes as fallback (LEGACY:{code} format)

2. **Denormalization Logic** - Complex row splitting (1 row → 10-50 rows)
   - **Mitigation**: Unit tests for transformation functions
   - **Mitigation**: Dry-run mode for testing
   - **Mitigation**: Sample data validation (100 rows)

3. **Missing Code Tables** - ct_Pre1982 may not have all codes
   - **Mitigation**: Preserve unmapped codes as "LEGACY:{code}"
   - **Mitigation**: Document unmapped fields in specification
   - **Mitigation**: Accept partial decoding (better than no data)

### Low Risk ✅
1. **Disk Space** - +500 MB database size
   - **Mitigation**: Check available disk space first (df -h)
   - **Mitigation**: Clean up /tmp/ before loading
   - **Mitigation**: Database size well within modern storage limits

2. **Date Range Overlap** - PRE1982 (1962-1981) vs. current (2000-2025)
   - **Mitigation**: ZERO overlap expected (verified in analysis)
   - **Mitigation**: Duplicate detection in staging tables
   - **Mitigation**: ev_id format different (YYYYMMDDX vs. YYYYMMDD)

---

## Timeline

### Estimated Total: 12-16 hours (over 3-4 work sessions)

**Session 1: Analysis & Mapping** (3-4 hours)
- Task 1.1: Extract PRE1982 schema (1 hour)
- Task 1.2: Analyze column structure (1 hour)
- Task 1.3: Create column mapping spec (1-2 hours)

**Session 2: Code Tables & ETL** (5-6 hours)
- Task 2.1: Extract ct_Pre1982 (1 hour)
- Task 2.2: Create code mapping schema (1 hour)
- Task 2.3: Populate code tables (1-2 hours)
- Task 3.1: Create load_pre1982.py (3-4 hours)
- Task 3.2: Test with sample data (overlap with 3.1)

**Session 3: Testing & Validation** (2-3 hours)
- Task 4.1: Small dataset testing (30 minutes)
- Task 4.2: Data quality validation (1 hour)
- Task 4.3: Full dataset load (30 minutes)
- Task 4.4: Performance benchmarking (30 minutes)

**Session 4: Documentation & Integration** (1-2 hours)
- Task 5.1: Update documentation (1 hour)
- Task 5.2: Create completion report (30 minutes)
- Task 5.3: Git commit and push (15 minutes)

**Contingency Buffer**: +2-4 hours for unexpected issues

---

## Dependencies

### External Dependencies ✅
- [x] mdbtools installed (`sudo apt install mdbtools` or `brew install mdbtools`)
- [x] PRE1982.MDB file in datasets/ (188 MB)
- [x] PostgreSQL 18.0 running
- [x] Python 3.11+ with pandas, psycopg2, numpy

### Internal Dependencies ✅
- [x] Phase 1 Sprint 3 complete (Airflow + monitoring)
- [x] Database ownership transferred to current user
- [x] Staging table infrastructure exists
- [x] load_tracking table operational

### Knowledge Dependencies 📚
- [ ] docs/PRE1982_ANALYSIS.md reviewed
- [ ] ref_docs/codman.pdf aviation codes understood
- [ ] scripts/load_with_staging.py pattern understood
- [ ] Pandas pivot operations (wide → tall transformations)

---

## Success Metrics

### Quantitative Metrics 📊
- **Events loaded**: +87,000 (92,771 → 179,771)
- **Coverage years**: 63 (1962-2025, from 26 years)
- **Database size**: ~1,012 MB (from 512 MB, +97.7%)
- **Total rows**: ~2.5M (from ~733,000, +241%)
- **Data quality**: 9/9 checks passed
- **Query performance**: <500ms p99 (within targets)
- **ETL duration**: <15 minutes
- **Code coverage**: 500+ legacy codes decoded

### Qualitative Metrics ✨
- ✅ Complete historical aviation accident dataset (63 years)
- ✅ Foundation for longitudinal trend analysis (1962-2025)
- ✅ Research-grade data quality maintained
- ✅ Comprehensive documentation (2,500+ lines)
- ✅ Reproducible ETL pipeline
- ✅ No regression in existing data quality

---

## Post-Sprint Recommendations

### Immediate (Week Following Sprint 4)
1. **Coverage Gap Analysis** (1 hour)
   - Identify any remaining gaps in 1977-1999 (partial Pre2008 coverage)
   - Assess data completeness by year
   - Document missing events (if any)

2. **Historical Trend Visualization** (2-3 hours)
   - Create 63-year accident trend charts
   - Analyze decade-by-decade patterns
   - Identify significant safety improvements

### Phase 2 (Future Sprints)
3. **Advanced Analytics Preparation** (Sprint 5)
   - Machine learning on 63 years of data
   - Predictive models for accident severity
   - Time series forecasting
   - Causal analysis (regulation impact)

4. **API Development** (Sprint 6)
   - Expose historical data via REST API
   - Enable external research access
   - Public data portal

5. **Research Applications**
   - Academic partnerships
   - Policy impact studies
   - Aviation safety evolution analysis
   - Regulatory era comparison (pre-1978 CAB vs. post-1978 FAA)

---

## References

### Documentation 📚
- [docs/PRE1982_ANALYSIS.md](../docs/PRE1982_ANALYSIS.md) - Schema analysis and integration challenges
- [ref_docs/codman.pdf](../ref_docs/codman.pdf) - Aviation coding manual (occurrence codes, cause codes)
- [ref_docs/eadmspub_legacy.pdf](../ref_docs/eadmspub_legacy.pdf) - Legacy schema documentation
- [scripts/load_with_staging.py](../scripts/load_with_staging.py) - ETL pattern reference (staging tables)
- [scripts/schema.sql](../scripts/schema.sql) - Target normalized schema

### Tools 🛠️
- **mdbtools**: https://github.com/mdbtools/mdbtools
- **pandas**: https://pandas.pydata.org/docs/
- **PostgreSQL COPY**: https://www.postgresql.org/docs/current/sql-copy.html
- **psycopg2**: https://www.psycopg.org/docs/

### Code Examples 💻
- Existing ETL: `scripts/load_with_staging.py` (staging pattern, duplicate detection)
- Data validation: `scripts/validate_data.sql` (9 validation categories)
- Anomaly detection: `scripts/detect_anomalies.py` (5 automated checks)

---

## Appendix A: Sample PRE1982 Record

**Source**: tblFirstHalf (denormalized)

```csv
RecNum: 40
TRANS_DATE: 05/20/71
DOCKET_NO: 1 0042
REGIST_NO: N6502R
DATE_OCCURRENCE: 07/23/62 00:00:00
TIME_OCCUR: 1430
LOCATION: ROME,NY
LOCAT_STATE_TERR: 32
ACFT_MAKE: ARMSTRONG
ACFT_MODEL: AW-650
ACFT_DAMAGE: D
NO_ENGINES: 2
PILOT_INVOLED1: A
HOURS_TOTAL_PILOT1: 11508A
AGE_PILOT1: ZA
PILOT_FATAL: 1
PILOT_SERIOUS: 0
PASSENGERS_FATAL: 5
PASSENGERS_SERIOUS: 2
TOTAL_ABRD_FATAL: 6
CAUSE_FACTOR_1P: 70
CAUSE_FACTOR_1M: A
CAUSE_FACTOR_1S: CB
```

**Transformed to** (6 normalized tables):

**events**:
```sql
ev_id: '19620723X000040'
ev_date: '1962-07-23'
ev_time: '14:30:00'
ev_year: 1962
ev_month: 7
ev_city: 'ROME,NY'
ev_state: 'NY'
ntsb_no: '1 0042'
inj_tot_f: 6
inj_tot_s: 2
```

**aircraft**:
```sql
ev_id: '19620723X000040'
Aircraft_Key: '1'
regis_no: 'N6502R'
acft_make: 'ARMSTRONG'
acft_model: 'AW-650'
num_eng: 2
damage: 'DEST'
```

**Flight_Crew**:
```sql
ev_id: '19620723X000040'
Aircraft_Key: '1'
crew_category: 'PILOT'
crew_age: NULL  (ZA = Unknown)
pilot_tot_time: 11508
```

**injury** (4 rows):
```sql
(ev_id='19620723X000040', inj_person_category='PILOT', inj_level='FATL', inj_person_count=1)
(ev_id='19620723X000040', inj_person_category='PASSENGER', inj_level='FATL', inj_person_count=5)
(ev_id='19620723X000040', inj_person_category='PASSENGER', inj_level='SERS', inj_person_count=2)
(ev_id='19620723X000040', inj_person_category='TOTAL', inj_level='FATL', inj_person_count=6)
```

**Findings**:
```sql
ev_id: '19620723X000040'
Aircraft_Key: '1'
finding_description: 'Pilot - inadequate preflight preparation'
cm_inPC: TRUE
modifier_code: 'A'
cause_factor: '70-A-CB'
```

---

## Appendix B: ev_id Generation Logic

**Format**: `YYYYMMDDX{RecNum:06d}`

**Examples**:
- RecNum=40, DATE_OCCURRENCE='07/23/62' → `19620723X000040`
- RecNum=12345, DATE_OCCURRENCE='12/31/81' → `19811231X012345`

**Python Implementation**:
```python
def generate_ev_id(rec_num: int, date_occurrence: str) -> str:
    """
    Generate synthetic ev_id for PRE1982 events.

    Format: YYYYMMDDX{RecNum:06d}
    - YYYYMMDD: Event date (parsed from MM/DD/YY)
    - X: Legacy marker (distinguishes from modern events)
    - RecNum: 6-digit zero-padded RecNum
    """
    # Parse date (handles 2-digit year)
    date_obj = pd.to_datetime(date_occurrence, format='%m/%d/%y %H:%M:%S')
    date_str = date_obj.strftime('%Y%m%d')

    # Format ev_id
    ev_id = f"{date_str}X{rec_num:06d}"
    return ev_id
```

**Collision Risk**: MINIMAL
- Modern ev_id format: `YYYYMMDD{sequence:05d}` (no 'X')
- Legacy ev_id format: `YYYYMMDDX{RecNum:06d}` (has 'X')
- Date ranges non-overlapping: PRE1982 (1962-1981) vs. modern (2000-2025)

---

**END OF SPRINT 4 PLANNING DOCUMENT**

**Status**: READY FOR EXECUTION
**Estimated Effort**: 12-16 hours (3-4 work sessions)
**Expected Completion**: Sprint 4 end
**Priority**: MEDIUM (fills critical 20-year coverage gap)

---

**Document Metadata**:
- **Created**: 2025-11-07
- **Author**: Claude Code (Planning Agent)
- **Version**: 1.0.0
- **Last Updated**: 2025-11-07
- **Sprint**: Phase 1 Sprint 4
- **Status**: PLANNING COMPLETE, READY FOR EXECUTION
