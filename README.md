# NTSB Aviation Accident Database Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Database Size](https://img.shields.io/badge/Database%20Size-1.6%20GB-blue.svg)](https://github.com/doublegate/NTSB-Dataset_Analysis)
[![Last Commit](https://img.shields.io/github/last-commit/doublegate/NTSB-Dataset_Analysis?label=Last%20Commit)](https://github.com/doublegate/NTSB-Dataset_Analysis/commits/main)
[![Data Source: NTSB](https://img.shields.io/badge/Data-NTSB-blue.svg)](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)
[![Fish Shell](https://img.shields.io/badge/Shell-Fish-green.svg)](https://fishshell.com/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

Complete archive and analysis toolkit for National Transportation Safety Board (NTSB) aviation accident investigation data from 1962 to present. This repository provides comprehensive tools for extracting, querying, analyzing, and visualizing aviation accident data across 60+ years of aviation history.

## Project Status

**Version**: 1.0.1
**Status**: Production-ready
**Last Updated**: November 2025

This repository is fully functional and production-ready with:
- Three comprehensive databases (1962-present, 1.6GB total)
- Complete extraction and analysis toolkit
- Automated setup scripts for CachyOS/Arch Linux
- Comprehensive documentation and examples
- Active maintenance and monthly data updates (avall.mdb)
- Production-ready Python examples with robust error handling

**Repository Topics**: aviation, ntsb, accident-analysis, aviation-safety, data-analysis, python, fish-shell, mdb-database, duckdb, jupyter-notebook

## Recent Improvements

### Error Handling & Data Validation (v1.0.1)

All Python example scripts now include comprehensive error handling and data validation, making them production-ready for real-world NTSB data analysis. Scripts gracefully handle data quality issues and provide clear user feedback.

**Key Fixes**:

1. **Seasonal Analysis Date Parsing** - Fixed crash on invalid date formats
   - **Issue**: Script crashed with "Conversion Error: Could not convert string '/0' to INT32"
   - **Solution**: Implemented TRY_CAST, regex validation, and BETWEEN checks
   - **Result**: Analysis continues gracefully with warning message instead of crashing

2. **Geospatial Coordinate Columns** - Fixed coordinate mapping bug
   - **Issue**: Script found 0 events because it queried DMS format columns (latitude/longitude)
   - **Root Cause**: NTSB database stores coordinates in two formats:
     - DMS format: latitude/longitude (e.g., "043594N", "0883325W")
     - Decimal degrees: dec_latitude/dec_longitude (e.g., 43.98, -88.55)
   - **Solution**: Changed to use dec_latitude/dec_longitude columns
   - **Result**: Now successfully loads 7,903 events and creates 3 interactive maps

3. **Database-Prefixed Filenames** - Fixed CSV file references
   - **Issue**: Scripts looked for generic filenames (events.csv)
   - **Solution**: Updated to use database-prefixed names (avall-events.csv)
   - **Result**: Scripts work out-of-the-box after running extraction scripts

**Error Handling Improvements**:
- Input validation and parameter checking (year ranges, coordinate bounds)
- Defensive SQL queries with TRY_CAST, COALESCE, TRIM, LENGTH validation
- Comprehensive try-except blocks with meaningful error messages
- Empty dataset detection and warning messages
- Graceful degradation (continue analysis when one part fails)
- Formatted output with thousand separators

## Testing Results

All example scripts have been tested and verified:

### quick_analysis.py
- ✅ Analyzes 100 recent events (2023-2024)
- ✅ Reports 250 fatalities, 48 serious injuries
- ✅ Identifies 21 fatal and 79 non-fatal accidents
- ✅ Handles NULL values in injury fields

### advanced_analysis.py
- ✅ Processes 29,773 total events
- ✅ Completes 5 core analyses successfully
- ✅ Handles invalid dates gracefully in seasonal analysis
- ✅ Generates summary report with 9,510 records (2020+)
- ✅ Top aircraft: Cessna 172 (643), Boeing 737 (616)

### geospatial_analysis.py
- ✅ Loads 7,903 events with valid coordinates (2020+)
- ✅ Creates 3 interactive maps (accident map, heatmap, fatal accidents)
- ✅ Maps 1,389 fatal accidents
- ✅ Regional analysis: West (9,442), South (8,142), Midwest (4,339)
- ✅ Uses decimal coordinate columns (dec_latitude/dec_longitude)

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Database Structure](#database-structure)
- [NTSB Coding System](#ntsb-coding-system)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Example Queries](#example-queries)
- [Recommended Tools](#recommended-tools)
- [Use Cases](#use-cases)
- [Data Updates](#data-updates)
- [Contributing](#contributing)
- [License](#license)
- [Resources](#resources)

## Features

- **Three comprehensive databases** spanning 1962-present with 60+ years of accident data
- **Automated extraction scripts** for converting MDB databases to CSV/SQLite formats
- **Fast SQL query tools** using DuckDB for rapid data analytics
- **Python analysis examples** with pandas, polars, and visualization libraries
- **Geospatial analysis** capabilities with mapping and hotspot identification
- **Text mining tools** for analyzing accident narratives and investigation reports
- **Jupyter notebooks** for interactive data exploration
- **Complete documentation** including database schema, coding manual references, and installation guides
- **Fish shell scripts** for streamlined workflow automation
- **Cross-format support** (MDB, CSV, SQLite, Parquet) for flexible analysis

## 📊 Datasets

This repository contains three comprehensive Microsoft Access databases:

| Database | Time Period | Size | Records |
|----------|-------------|------|---------|
| `datasets/avall.mdb` | 2008 - Present | 537 MB | Updated monthly |
| `datasets/Pre2008.mdb` | 1982 - 2007 | 893 MB | Static snapshot |
| `datasets/PRE1982.MDB` | 1962 - 1981 | 188 MB | Static snapshot |

## Database Structure

### Core Tables

The databases follow a relational structure with the following primary tables:

- **events** - Master accident/incident records (keyed by `ev_id`)
- **aircraft** - Aircraft involved in each event
- **Flight_Crew** - Pilot and crew information
- **injury** - Injury details and fatality counts
- **Findings** - Investigation findings and probable causes
- **Occurrences** - Specific events during accidents
- **seq_of_events** - Timeline of events leading to accidents
- **engines** - Engine specifications and failures
- **narratives** - Detailed textual descriptions

### Key Relationships

- `ev_id` - Primary key linking events across most tables
- `Aircraft_Key` - Identifies specific aircraft within events
- Foreign key relationships documented in `ref_docs/eadmspub.pdf`

For complete schema documentation with entity relationship diagrams, see `ref_docs/eadmspub.pdf`.

## NTSB Coding System

NTSB uses a hierarchical coding system to classify accidents (see `ref_docs/codman.pdf`):

- **100-430**: Occurrence types (engine failure, midair collision, fuel exhaustion, etc.)
- **500-610**: Phase of operation (taxi, takeoff, cruise, approach, landing, etc.)
- **10000-21104**: Aircraft/equipment subjects (airframe, systems, powerplant)
- **22000-25000**: Performance, operations, ATC, maintenance
- **30000-84200**: Direct underlying causes
- **90000-93300**: Indirect contributing factors

## Quick Start

Get started analyzing NTSB data in under 5 minutes:

### 1. Clone the Repository

```fish
git clone https://github.com/doublegate/NTSB-Dataset_Analysis.git
cd NTSB-Dataset_Analysis
```

### 2. Automated Setup

```fish
# Install all required tools (mdbtools, Python, Rust tools, etc.)
./setup.fish

# Activate Python environment
source .venv/bin/activate.fish
```

### 3. Extract Data from Databases

```fish
# Extract all tables with database-prefixed naming
./scripts/extract_all_tables.fish datasets/avall.mdb
```

### 4. Start Analyzing

All example scripts are production-ready and work out-of-the-box:

```fish
# Quick analysis (100 recent events)
.venv/bin/python examples/quick_analysis.py

# Comprehensive analysis (5 analyses, summary report)
.venv/bin/python examples/advanced_analysis.py

# Interactive maps (requires folium)
.venv/bin/python examples/geospatial_analysis.py

# Or launch Jupyter for interactive work
jupyter lab
# Open examples/starter_notebook.ipynb
```

## Installation

### Prerequisites

- CachyOS/Arch Linux (or compatible)
- Fish shell
- Python 3.11+
- ~5GB free disk space

### Detailed Installation

For complete installation instructions including troubleshooting, see [INSTALLATION.md](INSTALLATION.md).

Quick automated install:
```fish
./setup.fish
```

This installs:
- System packages (sqlite, python, gdal)
- AUR packages (mdbtools, duckdb)
- Python packages (pandas, jupyter, plotly, etc.)
- Rust tools (xsv, polars-cli)
- NLP models for text analysis

## Documentation

Comprehensive documentation is available:

- **[INSTALLATION.md](INSTALLATION.md)** - Complete setup guide for CachyOS/Arch Linux
- **[QUICKSTART.md](QUICKSTART.md)** - Essential commands and common workflows
- **[TOOLS_AND_UTILITIES.md](TOOLS_AND_UTILITIES.md)** - Comprehensive tool catalog
- **[CLAUDE.md](CLAUDE.md)** - Repository structure and database schema reference
- **[scripts/README.md](scripts/README.md)** - Detailed Fish shell script documentation
- **[examples/README.md](examples/README.md)** - Python analysis examples guide
- **ref_docs/** - Official NTSB schema documentation and coding manuals
  - `eadmspub.pdf` - Database schema and entity relationships
  - `codman.pdf` - Aviation coding manual
  - `MDB_Release_Notes.pdf` - Database release notes

## Example Queries

### SQL Queries (using DuckDB)
```sql
-- Recent fatal accidents
SELECT ev_id, ev_date, ev_type, ev_state, inj_tot_f
FROM 'data/events.csv'
WHERE inj_tot_f > 0 AND ev_year >= 2020
ORDER BY ev_date DESC;

-- Most common occurrence types
SELECT occurrence_code, COUNT(*) as count
FROM 'data/Occurrences.csv'
GROUP BY occurrence_code
ORDER BY count DESC
LIMIT 10;
```

### Python Queries (using pandas)
```python
import pandas as pd

# Load events
events = pd.read_csv('data/events.csv')

# Filter by year and location
recent = events[
    (events['ev_year'] >= 2020) &
    (events['ev_state'] == 'CA')
]

# Summary statistics
print(recent['inj_tot_f'].sum())  # Total fatalities
```

### Fish Shell (using csvkit)
```fish
# Quick statistics
csvstat data/events.csv

# Filter and count
csvgrep -c ev_state -m "CA" data/events.csv | wc -l

# Convert to JSON
csvjson data/events.csv > events.json
```

## Recommended Tools

### Database Access
- **mdbtools** - Extract data from .mdb files
- **DBeaver** - Universal database GUI
- **DuckDB** - Fast SQL analytics on CSV/Parquet
- **SQLite** - Convert MDB for easier querying

### Data Analysis
- **Python**: pandas, polars, numpy, scipy, scikit-learn
- **Rust**: polars-cli, datafusion, xsv, qsv
- **R**: dplyr, ggplot2, sf (geospatial)

### Visualization
- **Static**: matplotlib, seaborn, ggplot2
- **Interactive**: plotly, altair
- **Dashboards**: streamlit, dash, shiny

### Geospatial Analysis
- **geopandas** - Geospatial dataframes in Python
- **folium** - Interactive maps
- **QGIS** - Desktop GIS application

See **TOOLS_AND_UTILITIES.md** for complete installation and usage instructions.

## Use Cases

### Safety Analysis
- Identify common accident causes and trends
- Analyze seasonal/geographic patterns
- Study specific aircraft types or operations

### Machine Learning
- Predict accident severity from incident factors
- Classify accidents by probable cause
- Time series forecasting of accident rates

### Geospatial Analysis
- Map accident hotspots
- Analyze proximity to airports/airspace
- Visualize flight paths and trajectories

### Text Mining
- Extract insights from accident narratives
- Identify recurring themes in investigation reports
- Sentiment analysis of witness statements

### Statistical Research
- Survival analysis and injury severity models
- Causal inference in accident investigation
- Bayesian analysis of contributing factors

## Data Updates

- **avall.mdb**: Updated monthly by NTSB
- **Pre2008.mdb**: Static (archived data)
- **PRE1982.MDB**: Static (archived data)

Download latest data from [NTSB Aviation Accident Database](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Ways to contribute:
- Add analysis scripts and Jupyter notebooks
- Improve data cleaning/preprocessing utilities
- Create visualization dashboards
- Enhance documentation
- Report bugs and suggest features
- Share your research using this dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The NTSB aviation accident data itself is in the public domain as U.S. government data. The scripts, documentation, and tooling in this repository are provided under the MIT License.

## Acknowledgments

- **National Transportation Safety Board (NTSB)** for maintaining and providing comprehensive aviation accident investigation data
- **mdbtools** project for enabling MDB file access on Linux/Unix systems
- **DuckDB** project for fast analytical SQL queries
- All contributors who have improved this repository

## Resources

- [NTSB Aviation Accident Database](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)
- [NTSB Coding Manual](https://www.ntsb.gov/_layouts/ntsb.aviation/codeman.pdf)
- [FAA Aircraft Registry](https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/)
- [Aviation Safety Network](https://aviation-safety.net/)
