# DATA DICTIONARY

Complete schema reference for NTSB Aviation Accident Database. Based on official NTSB documentation (`ref_docs/eadmspub.pdf`) and database release notes.

## Table of Contents

- [Overview](#overview)
- [Primary Tables](#primary-tables)
- [events - Master Accident Records](#events---master-accident-records)
- [aircraft - Aircraft Details](#aircraft---aircraft-details)
- [Flight_Crew - Crew Information](#flight_crew---crew-information)
- [injury - Injury Statistics](#injury---injury-statistics)
- [Findings - Investigation Results](#findings---investigation-results)
- [Occurrences - Event Classification](#occurrences---event-classification)
- [seq_of_events - Timeline Data](#seq_of_events---timeline-data)
- [engines - Powerplant Details](#engines---powerplant-details)
- [narratives - Accident Descriptions](#narratives---accident-descriptions)
- [NTSB_Admin - Administrative Metadata](#ntsb_admin---administrative-metadata)
- [Key Relationships](#key-relationships)
- [Data Types Reference](#data-types-reference)

## Overview

The NTSB Aviation Accident Database uses a relational structure with `ev_id` (Event ID) as the primary linking key across most tables. The database follows a star schema pattern with `events` as the fact table and dimension tables containing related details.

### Database Evolution

**Release 3.0 (March 1, 2024)**:
- Added `cm_inPC` field to Findings table (TRUE/FALSE for probable cause citation)
- Deprecated `cause_factor` field (retained for pre-October 2020 cases)

**Legacy Schema**: Pre-2008 databases use older schema documented in `ref_docs/eadmspub_legacy.pdf`

## Primary Tables

| Table | Primary Key | Foreign Keys | Row Type | Description |
|-------|-------------|--------------|----------|-------------|
| events | ev_id | - | One per accident | Master accident/incident records |
| aircraft | Aircraft_Key | ev_id | Many per accident | Aircraft involved in events |
| Flight_Crew | crew_no | ev_id, Aircraft_Key | Many per aircraft | Crew member details |
| injury | - | ev_id, Aircraft_Key | Many per event | Injury and fatality records |
| Findings | - | ev_id, Aircraft_Key | Many per event | Investigation findings |
| Occurrences | - | ev_id, Aircraft_Key | Many per event | Occurrence classifications |
| seq_of_events | - | ev_id, Aircraft_Key | Many per event | Event sequence timeline |
| engines | eng_no | ev_id, Aircraft_Key | Many per aircraft | Engine specifications |
| narratives | - | ev_id | One per event | Narrative accident descriptions |
| NTSB_Admin | ev_id | - | One per event | Administrative metadata |

## events - Master Accident Records

Core table containing one record per aviation accident or incident.

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| **ev_id** | String(20) | **Primary Key** - Unique event identifier | "20220101001234" |
| ev_date | Date | Date of accident occurrence | "2022-01-15" |
| ev_time | Time | Local time of accident | "14:30" |
| ev_year | Integer | Year extracted from ev_date | 2022 |
| ev_month | Integer | Month (1-12) | 1 |
| ev_dow | String | Day of week | "Saturday" |

### Location Fields

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| ev_city | String(50) | City name | May be NULL for remote locations |
| ev_state | String(2) | US state abbreviation | "CA", "TX", etc. |
| ev_country | String(3) | Country code | "USA", "CAN", "MEX" |
| ev_site_zipcode | String(10) | ZIP code of accident site | |
| latitude | String(10) | Latitude in DMS format | "043594N" |
| longitude | String(10) | Longitude in DMS format | "0883325W" |
| **dec_latitude** | Decimal | Latitude in decimal degrees | 43.98 |
| **dec_longitude** | Decimal | Longitude in decimal degrees | -88.55 |

**Note**: Use `dec_latitude` and `dec_longitude` for geospatial analysis, not the DMS format fields.

### Classification Fields

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| ev_type | String(10) | Accident/Incident type | "ACC" (Accident), "INC" (Incident) |
| ev_highest_injury | String(10) | Highest injury severity | "FATL", "SERS", "MINR", "NONE" |
| ev_nr_apt_id | String(10) | Nearest airport identifier | ICAO/FAA codes |
| ev_nr_apt_loc | String(50) | Nearest airport location | City/State |
| ev_nr_apt_dist | Decimal | Distance to nearest airport (miles) | |

### Injury Totals

| Field | Type | Description |
|-------|------|-------------|
| inj_tot_f | Integer | Total fatalities |
| inj_tot_s | Integer | Total serious injuries |
| inj_tot_m | Integer | Total minor injuries |
| inj_tot_n | Integer | Total uninjured |

**Note**: NULL values indicate unreported data. Use COALESCE(inj_tot_f, 0) in queries.

### Weather Conditions

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| wx_cond_basic | String(10) | Basic weather condition | "VMC", "IMC", "UNK" |
| wx_temp | Integer | Temperature (Fahrenheit) | |
| wx_wind_dir | Integer | Wind direction (degrees) | 0-360 |
| wx_wind_speed | Integer | Wind speed (knots) | |
| wx_vis | Decimal | Visibility (statute miles) | |

### Flight Information

| Field | Type | Description |
|-------|------|-------------|
| flight_plan_filed | String(10) | Flight plan status | "NONE", "VFR", "IFR", "COMP" |
| flight_activity | String(50) | Type of flight activity | "Personal", "Instructional", etc. |
| flight_phase | String(50) | Phase of flight | "Cruise", "Approach", "Landing" |

### Investigation Details

| Field | Type | Description |
|-------|------|-------------|
| ntsb_no | String(20) | NTSB accident number | Official investigation ID |
| report_status | String(10) | Report status | "PREL" (Preliminary), "FINL" (Final) |
| probable_cause | Text | Probable cause statement | Long text field |

## aircraft - Aircraft Details

Contains information about each aircraft involved in an accident (one accident may involve multiple aircraft).

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| **Aircraft_Key** | String(20) | **Primary Key** | Unique aircraft identifier |
| **ev_id** | String(20) | **Foreign Key** → events.ev_id | Links to accident record |
| acft_serial_number | String(30) | Aircraft serial number | |
| regis_no | String(10) | Aircraft registration (N-number) | "N12345" |

### Aircraft Type

| Field | Type | Description |
|-------|------|-------------|
| acft_make | String(50) | Manufacturer | "Cessna", "Boeing" |
| acft_model | String(50) | Model designation | "172", "737-800" |
| acft_series | String(20) | Series/variant | "N", "MAX" |
| acft_category | String(20) | Aircraft category | "Airplane", "Helicopter", "Glider" |
| acft_type_code | String(10) | Type certification | |

### Operation Details

| Field | Type | Description |
|-------|------|-------------|
| far_part | String(10) | FAR part under which operated | "91", "121", "135" |
| oper_country | String(3) | Country of operator | "USA" |
| owner_city | String(50) | Owner city | |
| owner_state | String(2) | Owner state | |

### Damage Assessment

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| damage | String(10) | Aircraft damage level | "DEST" (Destroyed), "SUBS" (Substantial), "MINR" (Minor), "NONE" |

### Certification

| Field | Type | Description |
|-------|------|-------------|
| cert_max_gr_wt | Integer | Certified max gross weight (lbs) | |
| num_eng | Integer | Number of engines | |
| fixed_retractable | String(10) | Landing gear type | "RETR", "FIXD" |

## Flight_Crew - Crew Information

Crew member details including certifications and medical information.

| Field | Type | Description |
|-------|------|-------------|
| **crew_no** | Integer | **Primary Key** - Crew member sequence |
| **ev_id** | String(20) | **Foreign Key** → events.ev_id |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key |
| crew_category | String(20) | Crew role | "PLT" (Pilot), "COPL" (Co-pilot), "FLI" (Flight Instructor) |
| crew_age | Integer | Age at time of accident | |
| crew_sex | String(1) | Gender | "M", "F" |
| crew_seat | String(20) | Seat position | "Left", "Right" |

### Certifications

| Field | Type | Description |
|-------|------|-------------|
| pilot_cert | String(50) | Pilot certificate | "ATP", "Commercial", "Private", "Student" |
| pilot_rat | String(100) | Pilot ratings | "Airplane Multi-engine Land, Instrument" |
| pilot_med_class | String(5) | Medical certificate class | "1", "2", "3" |
| pilot_med_date | Date | Medical certificate issue date | |

### Experience

| Field | Type | Description |
|-------|------|-------------|
| pilot_tot_time | Integer | Total flight hours | |
| pilot_make_time | Integer | Hours in make/model | |
| pilot_90_days | Integer | Hours in last 90 days | |
| pilot_30_days | Integer | Hours in last 30 days | |
| pilot_24_hrs | Integer | Hours in last 24 hours | |

## injury - Injury Statistics

Detailed injury information for crew and passengers.

| Field | Type | Description |
|-------|------|-------------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key |
| inj_person_category | String(20) | Person category | "PLT", "PAX", "CREW", "GRND" |
| inj_level | String(10) | Injury severity | "FATL", "SERS", "MINR", "NONE" |
| inj_person_count | Integer | Number of persons | |

## Findings - Investigation Results

Investigation findings and causal factors.

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id | |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key | |
| finding_code | String(10) | Finding code | See Aviation Coding Lexicon (10000-93300) |
| finding_description | String(255) | Finding text | Human-readable description |
| **cm_inPC** | Boolean | **In Probable Cause** | TRUE if cited in probable cause (Release 3.0+) |
| cause_factor | String(10) | **Deprecated** - Legacy cause/factor flag | "C" (Cause), "F" (Factor) - pre-Oct 2020 only |
| modifier_code | String(10) | Modifier | Additional context code |

**Important**: Use `cm_inPC = TRUE` to filter probable cause findings in modern data (Oct 2020+).

## Occurrences - Event Classification

Specific occurrence types during accidents.

| Field | Type | Description | Code Range |
|-------|------|-------------|------------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id | |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key | |
| occurrence_code | String(10) | Occurrence type code | 100-430 |
| occurrence_description | String(255) | Occurrence text | "ENGINE FAILURE", "COLLISION WITH TERRAIN" |
| phase_code | String(10) | Phase of flight code | 500-610 |
| phase_description | String(100) | Phase text | "Cruise", "Approach", "Landing" |

**Reference**: See `AVIATION_CODING_LEXICON.md` for complete code definitions.

## seq_of_events - Timeline Data

Sequential events leading to and during the accident.

| Field | Type | Description |
|-------|------|-------------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key |
| seq_event_no | Integer | Sequence number |
| occurrence_code | String(10) | Event code |
| phase_of_flight | String(50) | Phase when event occurred |
| altitude | Integer | Altitude (feet MSL) |
| defining_event | Boolean | TRUE if defining event |

## engines - Powerplant Details

Engine specifications and failure information.

| Field | Type | Description |
|-------|------|-------------|
| **eng_no** | Integer | **Primary Key** - Engine sequence |
| **ev_id** | String(20) | **Foreign Key** → events.ev_id |
| **Aircraft_Key** | String(20) | **Foreign Key** → aircraft.Aircraft_Key |
| eng_make | String(50) | Engine manufacturer |
| eng_model | String(50) | Engine model |
| eng_type | String(20) | Engine type | "REC" (Reciprocating), "TURB" (Turbine) |
| eng_hp_or_lbs | Integer | Horsepower or thrust (lbs) |
| eng_carb_injection | String(10) | Fuel system | "CARB", "INJ" |

## narratives - Accident Descriptions

Unstructured narrative text describing the accident.

| Field | Type | Description | NLP Applications |
|-------|------|-------------|------------------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id | |
| narr_accp | Text | Accident description | Topic modeling, entity extraction |
| narr_cause | Text | Cause/contributing factors | Causal inference, classification |
| narr_rectification | Text | Corrective actions | Recommendation mining |

**NLP Techniques**:
- Named Entity Recognition (aircraft types, locations, weather)
- Topic Modeling (LDA, LSA)
- Sentiment Analysis
- Causal Relation Extraction
- BERT-based classification

## NTSB_Admin - Administrative Metadata

Administrative and investigation tracking data.

| Field | Type | Description |
|-------|------|-------------|
| **ev_id** | String(20) | **Foreign Key** → events.ev_id |
| ntsb_docket | String(50) | Docket number |
| invest_start_date | Date | Investigation start date |
| report_date | Date | Report publication date |
| invest_in_charge | String(100) | Lead investigator |

## Key Relationships

### Entity Relationship Diagram

```
events (1) ──< (M) aircraft
   │                  │
   │                  ├──< Flight_Crew
   │                  ├──< injury
   │                  ├──< Findings
   │                  ├──< Occurrences
   │                  ├──< seq_of_events
   │                  └──< engines
   │
   ├──< narratives (1:1)
   └──< NTSB_Admin (1:1)
```

### Join Patterns

**Common Joins**:

```sql
-- Events with aircraft details
SELECT e.*, a.*
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id;

-- Events with probable cause findings
SELECT e.ev_id, e.ev_date, f.finding_description
FROM events e
JOIN Findings f ON e.ev_id = f.ev_id
WHERE f.cm_inPC = TRUE;  -- Release 3.0+

-- Events with crew information
SELECT e.*, fc.pilot_cert, fc.pilot_tot_time
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
JOIN Flight_Crew fc ON a.Aircraft_Key = fc.Aircraft_Key
WHERE fc.crew_category = 'PLT';
```

## Data Types Reference

### String Conventions
- **Fixed-length codes**: Right-padded with spaces
- **Variable text**: May contain special characters, requires sanitization
- **NULL handling**: Empty strings ("") vs NULL - both possible

### Numeric Conventions
- **Integers**: Use COALESCE for NULL values in aggregations
- **Decimals**: Precision varies by field (2-6 decimal places)
- **Negative values**: Some fields (temperatures, longitudes) can be negative

### Date/Time Conventions
- **Dates**: ISO format (YYYY-MM-DD) in modern exports
- **Times**: 24-hour format, local time at accident location
- **Timezone**: Not stored; assume local time

### Boolean Conventions (Release 3.0+)
- **TRUE/FALSE**: Used in cm_inPC field
- **Legacy**: Older fields use "Y"/"N" or "1"/"0"

## Data Quality Notes

### Common Issues
1. **Missing Coordinates**: ~20% of pre-1990 records lack dec_latitude/dec_longitude
2. **Incomplete Narratives**: Preliminary reports have abbreviated narratives
3. **NULL Injury Counts**: Some records missing injury data entirely
4. **Date Format Inconsistencies**: Legacy databases use different date formats
5. **Code Deprecation**: Older coding systems evolved over 60 years

### Validation Techniques
- **Geospatial**: Check dec_latitude range [-90, 90], dec_longitude range [-180, 180]
- **Date**: Use `TRY_CAST` for robust date parsing, validate with BETWEEN checks
- **Codes**: Cross-reference against `AVIATION_CODING_LEXICON.md`
- **Regex**: `ev_date ~ '^\d{4}-\d{2}-\d{2}$'` for date format validation

---

**References**:
- Official Schema: `ref_docs/eadmspub.pdf`
- Legacy Schema: `ref_docs/eadmspub_legacy.pdf`
- Release Notes: `ref_docs/MDB_Release_Notes.pdf`
- Coding Manual: `ref_docs/codman.pdf`

**Last Updated**: January 2025 (Database Release 3.0)
