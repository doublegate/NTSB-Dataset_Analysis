# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **NTSB Aviation Accident Database** repository - a data repository containing aviation accident investigation records from the National Transportation Safety Board. This repository contains **databases and documentation only**, not application code.

## Database Files

Three Microsoft Access database files (.mdb) in `datasets/` covering different time periods:

- **datasets/avall.mdb** (537MB): Aviation accident data from **2008 to present**, updated monthly
- **datasets/Pre2008.mdb** (893MB): Aviation accident data from **1982 to 2007**
- **datasets/PRE1982.MDB** (188MB): Aviation accident data from **1962 to 1981**

## Database Schema

Core relational database structure (documented in `ref_docs/eadmspub.pdf`):

### Primary Tables
- **events**: Master table for accident events, keyed by `ev_id`
- **aircraft**: Aircraft involved in accidents, keyed by `Aircraft_Key`
- **Flight_Crew**: Flight crew information
- **injury**: Injury details for crew and passengers
- **Findings**: Investigation findings and probable causes
- **Occurrences**: Specific occurrence events during accidents
- **seq_of_events**: Sequence of events leading to accidents
- **Events_Sequence**: Event ordering and relationships
- **engines**: Engine details for involved aircraft
- **narratives**: Textual accident narratives and descriptions
- **NTSB_Admin**: Administrative metadata

### Key Relationships
- `ev_id` links events across most tables
- `Aircraft_Key` identifies specific aircraft within events
- Foreign key relationships documented in entity relationship diagrams in `ref_docs/eadmspub.pdf`

## Coding System

The aviation coding manual (`ref_docs/codman.pdf`, Revised 12/98) defines a hierarchical coding system:

### Occurrences (100-430)
Event types: ABRUPT MANEUVER, ENGINE FAILURE, MIDAIR COLLISION, FUEL EXHAUSTION, etc.

### Phase of Operation (500-610)
Flight phases: STANDING, TAXI, TAKEOFF, CRUISE, APPROACH, LANDING, MANEUVERING, etc.

### Section IA: Aircraft/Equipment Subjects (10000-21104)
Hierarchical codes for aircraft components:
- **10000-11700**: Airframe (wings, fuselage, landing gear, flight controls)
- **12000-13500**: Systems (hydraulic, electrical, environmental, fuel)
- **14000-17710**: Powerplant (engines, propellers, turbines, exhaust)

### Section IB: Performance/Operations (22000-25000)
- **22000-23318**: Performance subjects (stall, altitude, airspeed, weather)
- **24000-24700**: Operations (pilot technique, procedures, planning)
- **25000**: ATC and maintenance

### Section II: Direct Underlying Causes (30000-84200)
Detailed cause codes organized by aircraft component and failure mode

### Section III: Indirect Underlying Causes (90000-93300)
Contributing factors: design, maintenance, organizational, regulatory

## Database Schema Changes

Recent changes documented in `ref_docs/MDB_Release_Notes.pdf`:

### Release 3.0 (March 1, 2024)
- **Added**: `cm_inPC` field to Findings table (TRUE/FALSE indicating if finding is cited in probable cause)
- **Deprecated**: `cause_factor` field (retained for pre-October 2020 cases for historical compatibility)

## Working with the Data

### Accessing Database Files
Use Microsoft Access or compatible tools (mdbtools on Linux/Mac):

```bash
# List tables (mdbtools)
mdb-tables datasets/avall.mdb

# Export table to CSV
mdb-export datasets/avall.mdb events > events.csv

# Query database
mdb-sql datasets/avall.mdb
```

### Common Queries
- Link `events` → `aircraft` via `ev_id`
- Link `aircraft` → `engines` via `Aircraft_Key`
- Join `events` → `Findings` to get probable causes
- Join `events` → `Occurrences` for occurrence codes
- Use `seq_of_events` + `Events_Sequence` for event ordering

### Interpreting Codes
1. Consult `ref_docs/codman.pdf` for code definitions
2. Codes are hierarchical (e.g., 12000-12999 are all system-related)
3. Phase codes (500-610) indicate when in flight the occurrence happened
4. Occurrence codes (100-430) describe what happened
5. Section II codes (30000+) describe why it happened
6. Section III codes (90000+) describe contributing factors

## Database Coverage

- **Current data** (datasets/avall.mdb): Updated monthly from NTSB
- **Historical data**: datasets/Pre2008.mdb and datasets/PRE1982.MDB are static snapshots
- **Schema differences**: Legacy schema documented in `ref_docs/eadmspub_legacy.pdf`

## Analysis Environment Setup

### Quick Start
```fish
# Install all recommended tools
./setup.fish

# Extract all tables from a database
./scripts/extract_all_tables.fish datasets/avall.mdb

# Activate Python environment
source .venv/bin/activate.fish

# Start Jupyter for interactive analysis
jupyter lab

# Or run example analysis
python examples/quick_analysis.py
```

### Comprehensive Tool Documentation
See **TOOLS_AND_UTILITIES.md** for complete list of recommended tools including:
- Database access (mdbtools, DBeaver, DuckDB, SQLite)
- Python data science (pandas, polars, numpy, scipy, scikit-learn)
- Visualization (matplotlib, seaborn, plotly, streamlit, dash)
- Geospatial analysis (geopandas, folium)
- Text analysis (nltk, spacy, wordcloud)
- Rust tools (polars-cli, xsv, qsv, datafusion)
- CLI tools (csvkit, jq, ripgrep, fd, fzf)

### Performance Tips
- Use **Polars** instead of pandas for 10x+ speedup on large datasets
- Use **DuckDB** for fast SQL analytics directly on CSV files
- Convert to **Parquet** format for 5-10x better compression
- Use **Dask** for datasets larger than RAM
