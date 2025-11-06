# PRE1982.MDB Structure Analysis Report

**Analysis Date**: November 6, 2025
**Database File**: datasets/PRE1982.MDB (181 MB)
**Purpose**: Assess compatibility with current NTSB schema and determine integration approach

---

## Executive Summary

PRE1982.MDB contains **20 years of aviation accident data (1962-1981)** but uses a **completely different schema** than the modern NTSB database structure. The database stores events in a **denormalized, wide-table format** rather than the normalized relational structure used in avall.mdb and Pre2008.mdb.

### Key Findings

| Attribute | Value | Notes |
|-----------|-------|-------|
| **Date Range** | 1962-01-01 to 1981-12-31 | ✅ Exactly 20 years as expected |
| **Total Records** | ~87,000 events | Estimated from tblFirstHalf |
| **Schema Compatibility** | ❌ **Incompatible** | Completely different structure |
| **Integration Effort** | **HIGH** (8-16 hours) | Requires custom ETL mapping |
| **Data Quality** | ⚠️ **Legacy Format** | Fixed-width coded fields, different taxonomy |
| **Priority** | **MEDIUM** | Fills 1962-1981 gap, but requires significant transformation |

### Recommendation

**Do NOT use the existing load_with_staging.py loader** for PRE1982.MDB. Instead:

1. **Phase 1 Sprint 2**: Document structure, defer integration to Sprint 3
2. **Sprint 3 (Future)**: Create custom ETL pipeline to transform PRE1982 → modern schema
3. **Alternative**: Accept 2000-2025 coverage (26 years) as sufficient for most analyses

---

## Database Structure

### Tables

PRE1982.MDB contains **5 tables** vs **11 tables** in modern schema:

| Table | Rows | Description | Modern Equivalent |
|-------|------|-------------|-------------------|
| **tblFirstHalf** | ~87,000 | Main event data (Part 1) | `events` + `aircraft` + `Flight_Crew` |
| **tblSecondHalf** | ~87,000 | Main event data (Part 2) | `injury` + cause factors |
| **tblOccurrences** | Unknown | Occurrence codes | `Occurrences` |
| **tblSeqOfEvents** | Unknown | Event sequences | `seq_of_events` |
| **ct_Pre1982** | Unknown | Code table lookup | Not directly mapped |

### Schema Comparison

#### Modern Schema (avall.mdb, Pre2008.mdb)
- **Normalized**: 11 related tables with foreign keys
- **Primary Key**: `ev_id` (VARCHAR) - Example: "20080101X00001"
- **Relationships**: `events` → `aircraft` → `engines`, `Flight_Crew`, etc.
- **Columns**: 40-60 columns per table, focused entities

#### Legacy Schema (PRE1982.MDB)
- **Denormalized**: Wide tables with 200+ columns each
- **Primary Key**: `RecNum` (INTEGER) - Sequential ID
- **Relationships**: Minimal, mostly flat structure
- **Columns**: 200+ columns in tblFirstHalf/tblSecondHalf (split for MS Access field limits)

---

## Key Differences

### 1. Column Count and Organization

**tblFirstHalf** (200+ columns):
- Event metadata: `DOCKET_NO`, `DATE_OCCURRENCE`, `TRANS_DATE`
- Aircraft: `ACFT_MAKE`, `ACFT_MODEL`, `REGIST_NO`, `NO_ENGINES`, `TYPE_CRAFT`
- Location: `LOCATION`, `LOCAT_STATE_TERR`, `LOCAT_NEAR`
- Pilot 1: `HOURS_TOTAL_PILOT1`, `AGE_PILOT1`, `MEDICAL_CERT_PILOT1`
- Pilot 2: `HOURS_TOTAL_PILOT2`, `AGE_PILOT2`, `MEDICAL_CERT_PILOT2`
- Cause factors: `CAUSE_FACTOR_1P`, `CAUSE_FACTOR_1M`, `CAUSE_FACTOR_1S` ... `CAUSE_FACTOR_10P/M/S`
- Injuries: `PILOT_FATAL`, `PASSENGERS_FATAL`, `TOTAL_ABRD_FATAL`, etc.

**tblSecondHalf** (likely another 200+ columns):
- Additional cause factors
- Detailed injury breakdowns
- Investigation metadata
- Narrative text (possibly)

### 2. Date Format

**PRE1982**:
- `DATE_OCCURRENCE`: Timestamp format `"MM/DD/YY HH:MM:SS"` (e.g., `"07/23/62 00:00:00"`)
- **Issue**: 2-digit year requires parsing (62 = 1962, not 2062)

**Modern Schema**:
- `ev_date`: DATE format `YYYY-MM-DD` (e.g., `2008-01-15`)
- Clean, unambiguous format

### 3. Primary Key Structure

**PRE1982**:
- `RecNum`: Simple integer (40, 41, 42, ...)
- No standardized event ID

**Modern Schema**:
- `ev_id`: Structured VARCHAR (`20080101X00001`)
- Format: `YYYYMMDD` + `X` + `00001` (sequential)
- **Challenge**: Must generate synthetic `ev_id` for PRE1982 events

### 4. Injury Data Structure

**PRE1982** (Denormalized):
- Separate columns for each role/severity combination:
  - `PILOT_FATAL`, `PILOT_SERIOUS`, `PILOT_MINOR`, `PILOT_NONE`
  - `CO_PILOT_FATAL`, `CO_PILOT_SERIOUS`, ...
  - `PASSENGERS_FATAL`, `PASSENGERS_SERIOUS`, ...
  - `TOTAL_ABRD_FATAL`, `TOTAL_ABRD_SERIOUS`, ...

**Modern Schema** (Normalized):
- `injury` table with rows:
  - `inj_person_category` (PILOT, CO-PILOT, PASSENGER)
  - `inj_level` (FATL, SERS, MINR, NONE)
  - `inj_person_count` (integer)

**Transformation Required**: Pivot wide columns → tall normalized rows

### 5. Cause Factor Coding

**PRE1982** (Legacy Codes):
- 30 cause factor columns: `CAUSE_FACTOR_1P`, `CAUSE_FACTOR_1M`, `CAUSE_FACTOR_1S` ... `CAUSE_FACTOR_10S`
- **P**rimary, **M**odifier, **S**econdary codes
- Coded values (e.g., "70", "A", "CB")
- Requires lookup table (`ct_Pre1982`)

**Modern Schema** (Hierarchical Codes):
- `Findings` table with:
  - `finding_code` (5-9 digit hierarchical code from `codman.pdf`)
  - `finding_description` (text description)
  - `cm_inPC` (boolean: in probable cause?)
  - `modifier_code` (optional modifier)

**Transformation Required**: Map legacy codes → modern finding codes

### 6. Aircraft Make/Model

**PRE1982**:
- Separate fields: `ACFT_MAKE`, `ACFT_MODEL`, `ACFT_MAKE_CODE`, `ACFT_MODEL_CODE`
- Abbreviated text (e.g., "ARMSTRONG", "AW-650")
- Codes (e.g., "175", "01")

**Modern Schema**:
- `acft_make` (full manufacturer name)
- `acft_model` (full model designation)
- `acft_series` (variant)

**Challenge**: Code-to-name mapping may require external reference tables

---

## Data Quality Assessment

### Sample Record Analysis

```csv
RecNum: 40
TRANS_DATE: 05/20/71
DOCKET_NO: 1 0042
REGIST_NO: N6502R
DATE_OCCURRENCE: 07/23/62 00:00:00
LOCATION: ROME,NY
ACFT_MAKE: ARMSTRONG
ACFT_MODEL: AW-650
LOCAT_STATE_TERR: 32 (code for NY?)
PILOT_INVOLED1: A (Active)
HOURS_TOTAL_PILOT1: 11508A (format unclear)
AGE_PILOT1: ZA (coded value)
```

### Observations

1. **Coded Values**: Many fields use single-letter or numeric codes requiring lookup tables
2. **Ambiguous Formats**: Pilot hours encoded as strings with letter suffixes
3. **Missing Standards**: State codes are numeric, not standard 2-letter abbreviations
4. **Data Entry Era**: Manual data entry from 1960s-1980s paper forms
5. **Completeness**: Likely higher NULL/unknown rates due to data collection methods of the era

---

## Integration Challenges

### Critical Blockers

1. **No ev_id Field**
   - **Impact**: Cannot use existing foreign key relationships
   - **Solution**: Generate synthetic `ev_id` from `RecNum` and `DATE_OCCURRENCE`
   - **Format**: `19620723R000040` (date + R + RecNum)

2. **Denormalized Structure**
   - **Impact**: Cannot bulk load into normalized tables
   - **Solution**: Write custom transformation logic to pivot data

3. **Code Tables Missing**
   - **Impact**: Cannot decode many legacy coded fields
   - **Solution**: Obtain or reverse-engineer `ct_Pre1982` lookup table

4. **Different Taxonomy**
   - **Impact**: Cause factors use different coding system than modern `codman.pdf`
   - **Solution**: Create mapping table or document as "legacy codes"

### Medium Challenges

5. **Date Parsing**: 2-digit years require century inference
6. **Injury Data Pivoting**: 50+ injury columns → normalized `injury` table rows
7. **Pilot Data**: Two pilots hard-coded as columns vs. flexible `Flight_Crew` table
8. **Aircraft Keys**: Must generate composite keys for `aircraft` table

---

## Proposed ETL Architecture

### Phase 1: Extraction (1-2 hours)
```bash
# Export all tables to CSV
mdb-export datasets/PRE1982.MDB tblFirstHalf > data/pre1982/tblFirstHalf.csv
mdb-export datasets/PRE1982.MDB tblSecondHalf > data/pre1982/tblSecondHalf.csv
mdb-export datasets/PRE1982.MDB tblOccurrences > data/pre1982/tblOccurrences.csv
mdb-export datasets/PRE1982.MDB tblSeqOfEvents > data/pre1982/tblSeqOfEvents.csv
mdb-export datasets/PRE1982.MDB ct_Pre1982 > data/pre1982/ct_Pre1982.csv
```

### Phase 2: Transformation (6-10 hours)

Create `scripts/transform_pre1982.py`:

```python
# Pseudo-code transformation logic

def transform_events(row):
    """Transform tblFirstHalf → events table"""
    return {
        'ev_id': generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
        'ev_date': parse_legacy_date(row['DATE_OCCURRENCE']),
        'ev_time': parse_legacy_time(row['TIME_OCCUR']),
        'ev_city': row['LOCATION'],
        'ev_state': decode_state(row['LOCAT_STATE_TERR']),
        'ntsb_no': row['DOCKET_NO'],
        # ... map 40+ fields
    }

def transform_aircraft(row):
    """Transform tblFirstHalf → aircraft table"""
    return {
        'ev_id': generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
        'Aircraft_Key': '1',  # Single aircraft per event in PRE1982
        'regis_no': row['REGIST_NO'],
        'acft_make': decode_make(row['ACFT_MAKE_CODE']),
        'acft_model': row['ACFT_MODEL'],
        'num_eng': row['NO_ENGINES'],
        # ... map 20+ fields
    }

def transform_flight_crew(row):
    """Transform tblFirstHalf → Flight_Crew table (2 rows per event)"""
    crews = []
    for pilot_num in [1, 2]:
        if row[f'PILOT_INVOLED{pilot_num}'] == 'A':
            crews.append({
                'ev_id': generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
                'Aircraft_Key': '1',
                'crew_category': 'PILOT' if pilot_num == 1 else 'CO-PILOT',
                'crew_age': decode_age(row[f'AGE_PILOT{pilot_num}']),
                'pilot_tot_time': parse_hours(row[f'HOURS_TOTAL_PILOT{pilot_num}']),
                'pilot_make_time': parse_hours(row[f'HOURS_IN_TYPE_PILOT{pilot_num}']),
                # ... map 10+ fields
            })
    return crews

def transform_injury(row):
    """Transform tblFirstHalf/tblSecondHalf → injury table (10+ rows per event)"""
    injuries = []
    injury_categories = [
        ('PILOT', 'PILOT'),
        ('CO_PILOT', 'CO-PILOT'),
        ('PASSENGERS', 'PASSENGER'),
        # ... 10+ categories
    ]
    for field_prefix, category in injury_categories:
        for level, level_code in [('FATL', 'FATAL'), ('SERS', 'SERIOUS'),
                                   ('MINR', 'MINOR'), ('NONE', 'NONE')]:
            count = row[f'{field_prefix}_{level_code}']
            if count > 0:
                injuries.append({
                    'ev_id': generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
                    'Aircraft_Key': '1',
                    'inj_person_category': category,
                    'inj_level': level,
                    'inj_person_count': count
                })
    return injuries

def transform_findings(row):
    """Transform tblFirstHalf → Findings table (10+ rows per event)"""
    findings = []
    for i in range(1, 11):
        primary = row[f'CAUSE_FACTOR_{i}P']
        modifier = row[f'CAUSE_FACTOR_{i}M']
        secondary = row[f'CAUSE_FACTOR_{i}S']

        if primary:
            findings.append({
                'ev_id': generate_ev_id(row['RecNum'], row['DATE_OCCURRENCE']),
                'Aircraft_Key': '1',
                'finding_code': map_legacy_code(primary),
                'modifier_code': modifier,
                'cm_inPC': (i == 1),  # First cause factor is probable cause
                # ... map description from ct_Pre1982
            })
    return findings
```

### Phase 3: Loading (2-3 hours)

Use `load_with_staging.py` with transformed data:
1. Load transformed CSVs → staging tables
2. Identify duplicates (should be ZERO, PRE1982 is 1962-1981, no overlap)
3. Merge into production tables
4. Validate row counts and foreign keys

---

## Integration Decision Matrix

| Scenario | Recommendation | Effort | Timeline |
|----------|----------------|--------|----------|
| **Need complete 1962-2025 coverage** | Proceed with custom ETL | HIGH (8-16 hrs) | Sprint 3 |
| **2000-2025 coverage sufficient** | Defer PRE1982 integration | N/A | No action |
| **Research on 1962-1981 accidents** | Extract to separate database | MEDIUM (4-6 hrs) | Sprint 3 |
| **Urgent analysis needed** | Use PRE1982 directly in MS Access | LOW (1 hr) | Immediate |

---

## Estimated Row Counts After Integration

### Current State (Post-Sprint 2)
- **Events**: 92,771 (2000-2025)
- **Total Rows**: ~1.38 million
- **Date Coverage**: 26 years

### Projected State (Post-PRE1982 Integration)
- **Events**: ~179,771 (1962-2025)
- **Total Rows**: ~3.5 million (estimated)
- **Date Coverage**: 64 years

**Calculation**:
- PRE1982 tblFirstHalf: ~87,000 events
- Estimated child records: ~1.5M (aircraft, crew, injury, findings)
- Database size: ~1.2 GB (projected)

---

## Risks and Mitigations

### High Risk
- ⚠️ **Code Mappings**: Legacy codes may not map cleanly to modern taxonomy
  - **Mitigation**: Store legacy codes as-is in `finding_description`, document discrepancies

### Medium Risk
- ⚠️ **Data Quality**: 1960s-1970s data entry may have high error rates
  - **Mitigation**: Implement aggressive data validation, flag suspect records

### Low Risk
- ✅ **Date Range Overlap**: PRE1982 (1962-1981) has ZERO overlap with current data
  - **Mitigation**: None needed, clean separation

---

## Recommendations

### Immediate (Sprint 2)
1. ✅ **Document PRE1982 structure** (this report)
2. ✅ **Mark as 'pending' in load_tracking table**
3. ✅ **Defer integration to Sprint 3 or later**

### Future (Sprint 3+)
4. ⏳ Extract `ct_Pre1982` code table for reference
5. ⏳ Create field mapping spreadsheet (PRE1982 → modern schema)
6. ⏳ Develop `scripts/transform_pre1982.py` custom ETL
7. ⏳ Test transformation with sample of 1,000 records
8. ⏳ Validate transformed data quality
9. ⏳ Load full PRE1982 dataset using staging tables
10. ⏳ Update CHANGELOG and README with 1962-2025 coverage

### Alternative Approach
- **Separate Database**: Load PRE1982 into `ntsb_aviation_legacy` database
  - Preserves original structure
  - Allows research on 1962-1981 without schema conflicts
  - Join queries across databases when needed

---

## Conclusion

PRE1982.MDB **is NOT compatible** with the current staging table loader and requires a **custom ETL pipeline** (8-16 hour effort). The data covers the expected 1962-1981 date range with ~87,000 events but uses a fundamentally different schema.

**Recommended Action**: Accept current 2000-2025 coverage (92,771 events) for Sprint 2 completion. Defer PRE1982 integration to Sprint 3 or future work based on research priorities.

**Alternative**: If 1962-1981 data is critical, prioritize custom ETL development in Sprint 3 with estimated 1-2 week timeline for complete integration.

---

**Report Status**: ✅ COMPLETE
**Next Steps**: Mark PRE1982 as out-of-scope for Sprint 2, proceed with query optimization
**Author**: Claude Code (Staging Table Analysis)
**Date**: November 6, 2025
