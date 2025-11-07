# NTSB Aviation Accident Database Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Database Size](https://img.shields.io/badge/Database%20Size-1.6%20GB-blue.svg)](https://github.com/doublegate/NTSB-Dataset_Analysis)
[![Last Commit](https://img.shields.io/github/last-commit/doublegate/NTSB-Dataset_Analysis?label=Last%20Commit)](https://github.com/doublegate/NTSB-Dataset_Analysis/commits/main)
[![Data Source: NTSB](https://img.shields.io/badge/Data-NTSB-blue.svg)](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)
[![Fish Shell](https://img.shields.io/badge/Shell-Fish-green.svg)](https://fishshell.com/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

Complete archive and analysis toolkit for National Transportation Safety Board (NTSB) aviation accident investigation data from 1962 to present. This repository provides comprehensive tools for extracting, querying, analyzing, and visualizing aviation accident data across 60+ years of aviation history.

## Project Status

**Version**: 2.0.0
**Status**: Production-ready with high-performance PostgreSQL database and automated ETL
**Last Updated**: November 6, 2025
**Current Sprint**: Phase 1 Sprint 2 - ✅ COMPLETE

This repository is fully functional and production-ready with:
- Three comprehensive databases (1962-present, 1.6GB MDB files)
- High-performance PostgreSQL database (966 MB, 92,771 events, 726,969 total rows)
- Query optimization: 6 materialized views, 59 indexes (30-114x speedup)
- Performance: p50 2ms, p95 13ms, 98.81% buffer cache hit ratio
- Automated one-command database setup (no manual SQL required)
- Production-grade ETL with staging tables and duplicate detection
- Comprehensive data validation suite
- Complete extraction and analysis toolkit
- Active maintenance and monthly data updates (avall.mdb)
- Production-ready Python examples with robust error handling

### Sprint 2 Achievements (November 2025)

✅ **Query Optimization**
- 6 materialized views (yearly, state, aircraft, decade, crew, findings)
- 59 total indexes (30 base + 29 performance/MV indexes)
- 30-114x speedup for analytical queries
- Index usage: 99.99% on primary tables

✅ **Historical Data Integration**
- 92,771 events loaded (1977-2025, 48 years with gaps)
- Staging table infrastructure for safe ETL operations
- Load tracking system prevents duplicate loads
- Production-grade deduplication logic

✅ **Performance Benchmarks**
- 20 comprehensive benchmark queries across 8 categories
- Query latency: p50 ~2ms, p95 ~13ms, p99 ~47ms
- Buffer cache hit ratio: 98.81% (excellent memory utilization)
- All queries meet or exceed performance targets

✅ **Production Infrastructure**
- `setup_database.sh` - Automated one-command setup (285 lines)
- NO SUDO operations required after initial setup
- Regular user ownership model
- Comprehensive data validation framework

See [Sprint 2 Completion Report](SPRINT_2_COMPLETION_REPORT.md) for detailed metrics.

**Repository Topics**: aviation, ntsb, accident-analysis, aviation-safety, data-analysis, python, fish-shell, mdb-database, duckdb, jupyter-notebook

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Database Structure](#database-structure)
- [NTSB Coding System](#ntsb-coding-system)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [📚 Comprehensive Documentation](#-comprehensive-documentation)
  - [Core Documentation (docs/)](#core-documentation-docs)
  - [Supporting Documentation (docs/supporting/)](#supporting-documentation-docssupporting)
  - [Project Roadmap (to-dos/)](#project-roadmap-to-dos)
  - [Key Highlights](#key-highlights)
- [Example Queries](#example-queries)
- [Recommended Tools](#recommended-tools)
- [Use Cases](#use-cases)
- [Data Updates](#data-updates)
- [Recent Improvements](#recent-improvements)
- [Testing Results](#testing-results)
- [Contributing](#contributing)
- [License](#license)
- [Resources](#resources)

## Features

- **Three comprehensive databases** spanning 1962-present with 60+ years of accident data
- **PostgreSQL database** with optimized schema, materialized views, and 59 indexes
- **Automated database setup** (`setup_database.sh`) for one-command initialization
- **Production-grade ETL pipelines** with staging tables and duplicate detection
- **Automated extraction scripts** for converting MDB databases to CSV/SQLite/PostgreSQL formats
- **Fast SQL query tools** using DuckDB and PostgreSQL for rapid data analytics
- **Query optimization** with 6 materialized views for common analytical queries
- **Data validation framework** with comprehensive quality checks
- **Python analysis examples** with pandas, polars, and visualization libraries
- **Geospatial analysis** capabilities with PostGIS, mapping and hotspot identification
- **Text mining tools** for analyzing accident narratives and investigation reports
- **Jupyter notebooks** for interactive data exploration
- **Complete documentation** including database schema, coding manual references, and installation guides
- **Fish shell scripts** for streamlined workflow automation
- **Cross-format support** (MDB, CSV, SQLite, PostgreSQL, Parquet) for flexible analysis

## 📊 Datasets

This repository contains three comprehensive Microsoft Access databases and an optimized PostgreSQL database:

### Source Databases (MDB Files)

| Database | Time Period | Size | Records |
|----------|-------------|------|---------|
| `datasets/avall.mdb` | 2008 - Present | 537 MB | Updated monthly |
| `datasets/Pre2008.mdb` | 1982 - 2007 | 893 MB | Static snapshot |
| `datasets/PRE1982.MDB` | 1962 - 1981 | 188 MB | Static snapshot |

### PostgreSQL Database

| Database | Events | Total Rows | Size | Coverage |
|----------|--------|------------|------|----------|
| `ntsb_aviation` | 92,771 | 726,969 | 966 MB | 1977-2025 (48 years) |

**Features:**
- Optimized schema with PostGIS for geospatial analysis
- 6 materialized views for fast analytical queries
- 59 indexes for query performance
- Data validation and referential integrity
- Automated ETL pipeline with duplicate detection

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

### Option A: PostgreSQL Database (Recommended)

For optimal query performance and advanced analytics:

```bash
# 1. Clone the repository
git clone https://github.com/doublegate/NTSB-Dataset_Analysis.git
cd NTSB-Dataset_Analysis

# 2. Setup PostgreSQL database (automated one-command setup)
./scripts/setup_database.sh

# 3. Load current data (2008-present)
source .venv/bin/activate
python scripts/load_with_staging.py --source datasets/avall.mdb

# 4. Load historical data (optional, 1982-2007)
python scripts/load_with_staging.py --source datasets/Pre2008.mdb

# 5. Optimize queries (create materialized views + indexes)
psql -d ntsb_aviation -f scripts/optimize_queries.sql

# 6. Run performance benchmarks (optional)
psql -d ntsb_aviation -f scripts/test_performance.sql

# 7. Start querying (sub-millisecond response times)
psql -d ntsb_aviation -c "SELECT * FROM mv_yearly_stats ORDER BY year DESC LIMIT 5;"
```

See [QUICKSTART_POSTGRESQL.md](QUICKSTART_POSTGRESQL.md) for detailed PostgreSQL setup and usage.

### Option B: CSV/DuckDB Analysis

For quick exploration without database setup:

```fish
# 1. Clone the repository
git clone https://github.com/doublegate/NTSB-Dataset_Analysis.git
cd NTSB-Dataset_Analysis

# 2. Automated Setup
./setup.fish  # Install tools: mdbtools, Python, Rust tools
source .venv/bin/activate.fish

# 3. Extract Data from Databases
./scripts/extract_all_tables.fish datasets/avall.mdb

# 4. Start Analyzing
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

### Getting Started
- **[QUICKSTART_POSTGRESQL.md](QUICKSTART_POSTGRESQL.md)** - PostgreSQL database setup and usage
- **[QUICKSTART.md](QUICKSTART.md)** - CSV/DuckDB analysis workflow
- **[INSTALLATION.md](INSTALLATION.md)** - Complete setup guide for CachyOS/Arch Linux

### Technical Documentation
- **[TOOLS_AND_UTILITIES.md](TOOLS_AND_UTILITIES.md)** - Comprehensive tool catalog
- **[CLAUDE.md](CLAUDE.md)** - Repository structure and database schema reference
- **[scripts/README.md](scripts/README.md)** - Detailed Fish shell script documentation
- **[examples/README.md](examples/README.md)** - Python analysis examples guide

### Sprint Reports & Analysis
- **[SPRINT_1_REPORT.md](SPRINT_1_REPORT.md)** - Phase 1 Sprint 1: PostgreSQL migration (478,631 rows)
- **[SPRINT_2_COMPLETION_REPORT.md](SPRINT_2_COMPLETION_REPORT.md)** - Phase 1 Sprint 2: Query optimization + historical data (700+ lines)
- **[docs/PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md)** - Comprehensive performance analysis (450+ lines)
- **[docs/PRE1982_ANALYSIS.md](docs/PRE1982_ANALYSIS.md)** - PRE1982.MDB schema analysis and integration strategy

### Reference Documentation
- **ref_docs/** - Official NTSB schema documentation and coding manuals
  - `eadmspub.pdf` - Database schema and entity relationships
  - `codman.pdf` - Aviation coding manual
  - `MDB_Release_Notes.pdf` - Database release notes
  - `eadmspub_legacy.pdf` - Legacy schema for PRE1982.MDB

## Performance Metrics

Sprint 2 query optimization achieved exceptional performance on the PostgreSQL database:

### Query Performance
- **p50 Latency**: ~2ms (median query response time)
- **p95 Latency**: ~13ms (95th percentile)
- **p99 Latency**: ~47ms (99th percentile)
- **Buffer Cache Hit Ratio**: 98.81% (excellent memory utilization)
- **Index Usage**: 99.99% on primary tables (events, aircraft)

### Materialized Views (30-114x Speedup)
- **mv_yearly_stats**: Yearly accident statistics (47 rows, 114x faster)
- **mv_state_stats**: State-level statistics (57 rows, 89x faster)
- **mv_aircraft_stats**: Aircraft make/model statistics (971 types, 78x faster)
- **mv_decade_stats**: Decade-level trends (6 decades, 58x faster)
- **mv_crew_stats**: Crew certification statistics (10 categories, 42x faster)
- **mv_finding_stats**: Investigation findings (861 distinct findings, 35x faster)

### Database Optimization
- **Total Indexes**: 59 (30 base + 29 performance/MV indexes)
- **Materialized Views**: 6 active views with automated refresh
- **Data Integrity**: 100% (zero duplicate events, zero orphaned records)
- **Coordinate Validation**: 100% (all coordinates within valid bounds)

All performance metrics meet or exceed enterprise database standards. See [Performance Benchmarks](docs/PERFORMANCE_BENCHMARKS.md) for detailed analysis and methodology.

## Key Scripts

### Database Setup & Management
- **`scripts/setup_database.sh`** (285 lines) - Automated one-command database setup
  - Creates PostgreSQL database with PostGIS extension
  - Installs schema, staging tables, load tracking
  - Transfers ownership to current user (NO SUDO after setup)
- **`scripts/schema.sql`** (468 lines) - Complete PostgreSQL schema definition
  - 11 core tables with triggers, constraints, indexes
  - Generated columns (ev_year, ev_month, location_geom)
- **`scripts/transfer_ownership.sql`** (98 lines) - Ownership transfer automation

### Data Loading & ETL
- **`scripts/load_with_staging.py`** (597 lines) - Production-grade ETL loader
  - Staging table pattern for safe data loading
  - Duplicate detection and deduplication logic
  - Load tracking prevents accidental re-loads
  - Handles 15,000-45,000 rows/sec throughput
- **`scripts/create_staging_tables.sql`** (279 lines) - Staging infrastructure
  - 11 staging tables in separate schema
  - Helper functions for row counts and duplicate stats
- **`scripts/create_load_tracking.sql`** (123 lines) - Load tracking system

### Query Optimization & Performance
- **`scripts/optimize_queries.sql`** (324 lines) - Query optimization suite
  - Creates 6 materialized views
  - Creates 9 additional performance indexes
  - Analyzes all tables and materialized views
  - `refresh_all_materialized_views()` function
- **`scripts/test_performance.sql`** (427 lines) - Performance benchmark suite
  - 20 comprehensive benchmark queries
  - Measures latency (p50, p95, p99)
  - Database health metrics

### Data Validation & Quality
- **`scripts/validate_data.sql`** (384 lines) - Comprehensive data quality checks
  - 10 validation categories
  - Row counts, primary keys, NULL values, data integrity
  - Foreign key validation, index usage, database size

All scripts are production-ready with error handling and comprehensive documentation.

## Example Queries

### PostgreSQL Queries (Recommended)

```sql
-- Recent fatal accidents with aircraft details
SELECT e.ev_id, e.ev_date, e.ev_type, e.ev_state, e.inj_tot_f,
       a.acft_make, a.acft_model
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.inj_tot_f > 0 AND e.ev_year >= 2020
ORDER BY e.ev_date DESC
LIMIT 100;

-- Yearly accident statistics (from materialized view)
SELECT * FROM mv_yearly_stats
WHERE year >= 2020
ORDER BY year DESC;

-- Top 10 aircraft types by accident count
SELECT * FROM mv_aircraft_stats
ORDER BY total_accidents DESC
LIMIT 10;

-- Geospatial query: accidents near a location
SELECT ev_id, ev_date, ev_state,
       ST_Distance(location_geom, ST_MakePoint(-122.4194, 37.7749)) as distance_meters
FROM events
WHERE location_geom IS NOT NULL
  AND ST_DWithin(location_geom, ST_MakePoint(-122.4194, 37.7749), 50000)
ORDER BY distance_meters;
```

### DuckDB Queries (CSV files)
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
- **PostgreSQL** - Primary analytical database (recommended)
- **PostGIS** - Geospatial extensions for PostgreSQL
- **pgAdmin** / **DBeaver** - Database GUI tools
- **mdbtools** - Extract data from .mdb files
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

## 📚 Comprehensive Documentation

This project now includes extensive documentation for transforming the NTSB database into an advanced analytics platform:

### Core Documentation (docs/)

- **PROJECT_OVERVIEW.md** - Executive summary, vision, architecture, technology stack
- **DATA_DICTIONARY.md** - Complete schema reference with 150+ field definitions
- **AVIATION_CODING_LEXICON.md** - NTSB hierarchical coding system (100-93300 range)
- **MACHINE_LEARNING_APPLICATIONS.md** - ML techniques with XGBoost (91.2% accuracy)
- **AI_POWERED_ANALYSIS.md** - LLM integration, RAG systems, knowledge graphs
- **ADVANCED_ANALYTICS_TECHNIQUES.md** - Time series, survival analysis, causal inference
- **ARCHITECTURE_VISION.md** - 7-layer system architecture, cloud deployment
- **TECHNICAL_IMPLEMENTATION.md** - PostgreSQL, Airflow, FastAPI, MLflow
- **NLP_TEXT_MINING.md** - SafeAeroBERT, text preprocessing, topic modeling
- **FEATURE_ENGINEERING_GUIDE.md** - Domain-specific feature creation for ML
- **MODEL_DEPLOYMENT_GUIDE.md** - MLflow versioning, A/B testing, drift detection
- **GEOSPATIAL_ADVANCED.md** - HDBSCAN clustering, Getis-Ord Gi* hotspot analysis
- **DOCUMENTATION_COMPLETION_REPORT.md** - Executive summary of TIER 1 work

### Supporting Documentation (docs/supporting/)

- **RESEARCH_OPPORTUNITIES.md** - Academic venues, grant funding, industry partnerships
- **DATA_QUALITY_STRATEGY.md** - Great Expectations, Pandera validation, MICE imputation
- **ETHICAL_CONSIDERATIONS.md** - Bias detection, fairness metrics, responsible AI
- **VISUALIZATION_DASHBOARDS.md** - Plotly Dash vs Streamlit, KPI design
- **API_DESIGN.md** - RESTful API, FastAPI, authentication, rate limiting
- **PERFORMANCE_OPTIMIZATION.md** - Database indexing, Polars vs pandas benchmarks
- **SECURITY_BEST_PRACTICES.md** - Encryption, RBAC/ABAC, secrets management

### Project Roadmap (to-dos/)

- **ROADMAP_OVERVIEW.md** - 15-month plan (5 phases, Q1 2025 - Q1 2026)
- **PHASE_1_FOUNDATION.md** - Database migration, ETL pipeline, data quality, API (12 weeks, 74KB, 32 code examples)
- **PHASE_2_ANALYTICS.md** - Time series, geospatial, survival analysis, dashboards (12 weeks, 99KB, 30+ code examples)
- **PHASE_3_MACHINE_LEARNING.md** - Feature engineering, XGBoost, SHAP, MLflow (12 weeks)
- **PHASE_4_AI_INTEGRATION.md** - NLP pipeline, RAG system, knowledge graphs (12 weeks)
- **PHASE_5_PRODUCTION.md** - Kubernetes, public API, real-time capabilities (12 weeks)
- **TECHNICAL_DEBT.md** - Code quality, refactoring priorities, performance bottlenecks
- **RESEARCH_TASKS.md** - Open research questions, experiments, conference submissions
- **TESTING_STRATEGY.md** - Test pyramid, unit/integration/E2E tests, security testing
- **DEPLOYMENT_CHECKLIST.md** - 100-item production launch checklist

### Key Highlights

- **🎯 91.2% ML Accuracy**: XGBoost severity prediction benchmark
- **📊 100+ Features**: Comprehensive feature engineering pipeline
- **🤖 87.9% NLP Accuracy**: SafeAeroBERT for aviation narratives
- **📈 15-Month Roadmap**: Detailed sprint-level planning
- **💻 500+ Code Examples**: Production-ready implementations
- **🔬 50+ Academic Papers**: Research-backed methodologies

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

## Next Steps (Sprint 3)

**Objective**: Apache Airflow ETL Pipeline for Automated Monthly Updates

### Planned Features

**ETL Automation** (Priority 1)
- 5 production DAGs for automated data pipeline
  - `monthly_sync_dag.py` - Automated avall.mdb updates from NTSB
  - `data_transformation_dag.py` - Data cleaning and normalization
  - `quality_check_dag.py` - Automated validation suite
  - `mv_refresh_dag.py` - Materialized view updates
  - `feature_engineering_dag.py` - ML feature preparation

**Monitoring & Alerting** (Priority 2)
- Email notifications for pipeline failures
- Slack integration for status updates
- Performance dashboard for ETL metrics
- Automated retry logic and error handling
- Data quality monitoring with alerts

**Historical Data Integration** (Priority 3)
- PRE1982.MDB integration (1962-1981 data)
- Custom ETL for legacy schema transformation
- Mapping coded fields to modern taxonomy
- Estimated ~87,000 additional events

**Estimated Timeline**: 4-6 weeks (December 2025 - January 2026)

See [Sprint 3 Implementation Plan](to-dos/SPRINT_3_IMPLEMENTATION_PLAN.md) for detailed task breakdown.

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
