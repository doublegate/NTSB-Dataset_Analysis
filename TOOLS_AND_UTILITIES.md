# Tools and Utilities for NTSB Database Analysis

Recommended tools for analyzing aviation accident data on CachyOS Linux with Fish shell.

## Database Access & Conversion

### Core Tools (Already Mentioned)
```fish
# mdbtools - Extract data from .mdb files (AUR package)
paru -S mdbtools
mdb-tables datasets/avall.mdb
mdb-export datasets/avall.mdb events > events.csv
```

### Conversion & Migration
```fish
# Convert MDB to SQLite (easier querying)
sudo pacman -S sqlite
mdb-sqlite datasets/avall.mdb avall.db

# DBeaver - Universal database GUI (supports MDB via JDBC)
paru -S dbeaver

# PostgreSQL - For robust relational queries
sudo pacman -S postgresql
# Use mdb-export | psql for migration
```

### Alternative Access Methods
```fish
# Python: pandas with sqlalchemy
pip install pandas sqlalchemy pandas-access

# Python: Direct MDB reading (limited support)
pip install pyodbc  # Requires unixODBC setup
```

## Python Data Science Stack

### Core Libraries
```fish
# Essential data analysis
pip install pandas numpy scipy

# Polars - Faster DataFrame library (Rust-based)
pip install polars

# Date/time handling for accident timestamps
pip install python-dateutil pytz
```

### Statistical Analysis
```fish
pip install statsmodels  # Time series, regression
pip install scikit-learn  # Machine learning
pip install pingouin  # Statistical tests
```

### Geospatial Analysis
```fish
# Aviation accidents have coordinates
pip install geopandas shapely folium
pip install geopy  # Geocoding and distance calculations
sudo pacman -S gdal  # Geospatial data abstraction library
```

### Text Analysis (Narratives)
```fish
# Natural language processing for accident narratives
pip install nltk spacy
pip install wordcloud  # Visualize common terms
pip install textblob  # Sentiment/classification

# Download spaCy English model
python -m spacy download en_core_web_sm
```

## Visualization

### Python Plotting
```fish
pip install matplotlib seaborn  # Static plots
pip install plotly  # Interactive visualizations
pip install altair  # Declarative statistical visualization
```

### Dashboards & Web Apps
```fish
pip install streamlit  # Rapid dashboard prototyping
pip install dash  # Production dashboards
pip install panel  # HoloViz dashboard framework
```

### Business Intelligence
```fish
# Metabase - Self-hosted BI (Docker)
docker pull metabase/metabase

# Apache Superset
paru -S apache-superset  # or pip install apache-superset

# Grafana - Excellent for time-series visualization
sudo pacman -S grafana
```

## Rust Data Tools

### DataFrames & Query Engines
```fish
# Polars - Lightning-fast DataFrame library
cargo install polars-cli

# DataFusion - SQL query engine for CSV/Parquet
cargo install datafusion-cli

# DuckDB - SQL OLAP database (AUR package)
paru -S duckdb
```

### CSV Processing
```fish
# xsv - Blazingly fast CSV toolkit (recommended, stable)
cargo install xsv

# qsv - Extended xsv with more features (full-featured version)
cargo install qsv --features feature_capable

# Or install qsvlite for a lighter version
cargo install qsv --features lite
```

**⚠️ Known Issue with qsv v9.1.0**: The current crates.io version has compilation errors. If installation fails, use one of these alternatives:

1. **Use xsv instead** (recommended - similar functionality, more stable)
2. **Install from git**: `cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'`
3. **Wait for qsv v9.1.1** - Fix is in progress

**qsv Variants**:
- `feature_capable` - Full-featured `qsv` binary with all capabilities
- `lite` - Lighter `qsvlite` binary with reduced features and faster compile time
- `datapusher_plus` - Specialized `qsvdp` binary for DataPusher+ integration

**Note**: The setup script installs the full-featured version for maximum functionality and handles errors gracefully.

## CLI Data Tools

### CSV/TSV Manipulation
```fish
# csvkit - CSV swiss army knife
pip install csvkit

# Usage examples:
csvstat events.csv  # Statistics
csvgrep -c ev_id -m "20140101" events.csv  # Filter
csvsql --query "SELECT * FROM events LIMIT 10" events.csv
```

### JSON/YAML Processing
```fish
sudo pacman -S jq yq  # JSON/YAML querying
paru -S dasel  # Unified selector for JSON/YAML/TOML/XML/CSV
```

### Modern Replacements
```fish
# bat - Better cat with syntax highlighting
sudo pacman -S bat

# ripgrep - Faster grep
sudo pacman -S ripgrep

# fd - Better find
sudo pacman -S fd

# fzf - Fuzzy finder
sudo pacman -S fzf
```

## Jupyter & Notebooks

### Interactive Development
```fish
pip install jupyter jupyterlab
pip install ipython

# Enhanced notebook features
pip install jupyter-contrib-nbextensions
pip install jupyterlab-git

# Start JupyterLab
jupyter lab
```

### Notebook Alternatives
```fish
# Marimo - Reactive Python notebooks
pip install marimo

# Quarto - Multi-language scientific publishing
paru -S quarto-cli
```

## High-Performance Processing

### Parallel & Distributed
```fish
# Dask - Parallel computing with pandas API
pip install dask[complete]

# Ray - Distributed computing framework
pip install ray[default]

# Modin - Parallel pandas (Ray/Dask backend)
pip install modin[all]
```

### Efficient File Formats
```fish
# Apache Arrow/Parquet - Columnar data format
pip install pyarrow fastparquet

# HDF5 - Hierarchical data format
pip install h5py tables
sudo pacman -S hdf5
```

## Database Query Engines

### DuckDB (Recommended)
```fish
# DuckDB - Fast analytical queries on CSV/Parquet (AUR package)
paru -S duckdb
pip install duckdb

# Direct SQL on CSV files:
duckdb -c "SELECT * FROM 'events.csv' WHERE year > 2020"
```

### SQLite
```fish
# Already mentioned, but excellent for local analysis
sudo pacman -S sqlite sqlite-analyzer

# Import CSV to SQLite
sqlite3 avall.db
.mode csv
.import events.csv events
```

## Reporting & Documentation

### Document Generation
```fish
# Quarto - Scientific publishing system
paru -S quarto-cli

# Pandoc - Universal document converter
sudo pacman -S pandoc

# Typst - Modern LaTeX alternative
cargo install typst-cli
```

### Automation
```fish
# Papermill - Parameterize and execute notebooks
pip install papermill

# Schedule with systemd timers or cron
sudo pacman -S cronie
```

## Aviation-Specific Tools

### Flight Data Analysis
```fish
# Traffic - Air traffic data analysis
pip install traffic

# FlightRadar24 API (if correlating with flight data)
pip install FlightRadarAPI
```

### Aviation Databases & APIs
```fish
# FAA aircraft registry
# ICAO aircraft database
# OpenSky Network API
```

## Recommended Development Workflow

### 1. Initial Setup
```fish
# Convert MDB to SQLite/CSV for faster access
mdb-export datasets/avall.mdb events > data/events.csv
mdb-export datasets/avall.mdb aircraft > data/aircraft.csv
mdb-export datasets/avall.mdb Findings > data/findings.csv

# Or convert entire database
for table in (mdb-tables datasets/avall.mdb)
    mdb-export datasets/avall.mdb $table > data/$table.csv
end
```

### 2. Exploratory Analysis
```fish
# Use DuckDB for fast SQL exploration
duckdb analysis.db

# Or use Jupyter with pandas/polars
jupyter lab
```

### 3. Production Pipeline
```fish
# Rust for ETL (performance)
# Python for analysis (ecosystem)
# DuckDB/SQLite for querying
# Streamlit/Dash for dashboards
```

## Package Installation Script

Create `setup.fish` for quick environment setup:

```fish
#!/usr/bin/env fish

# System packages
sudo pacman -S --needed sqlite postgresql python python-pip \
    gdal hdf5 jq yq bat ripgrep fd fzf cmake base-devel \
    gettext autoconf-archive txt2man

# AUR packages (if using paru)
paru -S --needed mdbtools duckdb dbeaver quarto-cli

# Python packages
pip install pandas polars numpy scipy statsmodels scikit-learn \
    matplotlib seaborn plotly altair \
    geopandas folium geopy \
    nltk spacy wordcloud \
    jupyter jupyterlab \
    streamlit dash \
    dask[complete] pyarrow \
    duckdb sqlalchemy \
    csvkit

# Rust tools (install separately for better control)
cargo install xsv
cargo install qsv --features feature_capable  # Full-featured version (may fail - see note above)
cargo install polars-cli
cargo install datafusion-cli

# Download NLP models
python -m spacy download en_core_web_sm
```

**Note**: If `qsv` installation fails due to compilation errors (known issue with v9.1.0), simply use `xsv` instead or install from git.

## Performance Considerations

### For Large Datasets
- **Polars** over pandas for 10x+ speedup
- **DuckDB** for in-process SQL analytics
- **Parquet** format for 5-10x compression vs CSV
- **Dask** for out-of-core computation (larger than RAM)
- **Arrow** for zero-copy data sharing between tools

### Memory Optimization
```python
# Read CSV in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Use categorical dtypes for coded fields
df['occurrence_code'] = df['occurrence_code'].astype('category')

# Polars lazy evaluation
df = pl.scan_csv('events.csv').filter(...).select(...).collect()
```

## Integration Examples

### Python + DuckDB
```python
import duckdb
import pandas as pd

# Query CSV directly with SQL
result = duckdb.query("""
    SELECT ev_id, ev_date, ev_type
    FROM 'data/events.csv'
    WHERE ev_year > 2020
""").to_df()
```

### Rust + Polars
```rust
use polars::prelude::*;

let df = CsvReader::from_path("data/events.csv")?
    .finish()?
    .lazy()
    .filter(col("ev_year").gt(2020))
    .collect()?;
```

### Fish Shell Automation
```fish
# Extract and analyze in one pipeline
mdb-export datasets/avall.mdb events | \
    csvgrep -c ev_year -m 2023 | \
    csvstat --mean --median
```

## Troubleshooting

### Failed qsv Installation (v9.1.0)

The current version of qsv (v9.1.0) on crates.io has known compilation errors. If you encounter build failures:

#### Quick Cleanup
```fish
# Run the provided cleanup script
./cleanup_qsv.fish
```

#### Manual Cleanup Commands
```fish
# 1. Uninstall qsv (if partially installed)
cargo uninstall qsv

# 2. Remove temporary build directories
rm -rf /tmp/cargo-install*

# 3. Clean qsv from cargo registry
find ~/.cargo/registry/cache -name "qsv-*.crate" -delete
find ~/.cargo/registry/src -type d -name "qsv-*" -exec rm -rf {} +
```

#### Deep Cleanup (All Unused Rust Dependencies)
```fish
# Install cargo-cache utility
cargo install cargo-cache

# Remove all unused dependencies (safe)
cargo cache --autoclean

# More aggressive cleanup
cargo cache --autoclean-expensive

# Check what will be removed first
cargo cache --dry-run
```

#### Alternatives to qsv
1. **Use xsv** (recommended) - Similar functionality, very stable
   ```fish
   cargo install xsv --locked
   # xsv usage is nearly identical to qsv for most operations
   ```

2. **Install qsv from git** (has fixes)
   ```fish
   cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'
   ```

3. **Wait for qsv v9.1.1** - Official fix in progress

### Cargo Cache Management

Monitor and manage Rust/Cargo cache size:

```fish
# Check cache size
du -sh ~/.cargo/registry
du -sh ~/.cargo/git

# List installed Rust tools
ls -lh ~/.cargo/bin

# Remove specific tool
cargo uninstall <tool_name>

# Clean build cache
cargo clean

# With cargo-cache installed:
cargo cache --info           # Show cache information
cargo cache --list-dirs      # List cache directories
cargo cache --gc             # Remove old cache entries
cargo cache --remove-dir git # Remove git cache only
```

### Tool Installation Verification

Verify which CSV tools are installed:

```fish
# Check for xsv
command -v xsv && xsv --version

# Check for qsv
command -v qsv && qsv --version

# Check for csvkit
command -v csvstat && csvstat --version

# Check for polars-cli
command -v polars && polars --version

# Check for datafusion-cli
command -v datafusion-cli && datafusion-cli --version
```
