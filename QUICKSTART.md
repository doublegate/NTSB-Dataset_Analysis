# NTSB Database - Quick Reference

Essential commands and workflows for analyzing NTSB aviation accident data.

## 🚀 Initial Setup (One Time)

```fish
# Run setup script to install all tools
./setup.fish

# Extract data from databases
./scripts/extract_all_tables.fish datasets/avall.mdb
./scripts/extract_all_tables.fish datasets/Pre2008.mdb
./scripts/extract_all_tables.fish datasets/PRE1982.MDB
```

## 📊 Common Data Operations

### Extract Data from MDB

```fish
# List all tables in a database
mdb-tables datasets/avall.mdb

# Export single table
mdb-export datasets/avall.mdb events > data/events.csv

# Export all tables (automated)
./scripts/extract_all_tables.fish datasets/avall.mdb
```

### Quick CSV Analysis

```fish
# View first 10 rows
head -n 10 data/events.csv

# Count rows
wc -l data/events.csv

# View column names
head -n 1 data/events.csv | tr ',' '\n'

# Quick statistics
csvstat data/events.csv

# Filter by state
csvgrep -c ev_state -m "CA" data/events.csv > data/ca_events.csv

# Search for term
csvgrep -c ev_city -m "Los Angeles" data/events.csv
```

### DuckDB Quick Queries

```fish
# Launch interactive DuckDB
duckdb

# Or one-liner queries
duckdb -c "SELECT COUNT(*) FROM 'data/events.csv'"
duckdb -c "SELECT ev_state, COUNT(*) as count FROM 'data/events.csv' GROUP BY ev_state ORDER BY count DESC LIMIT 10"
```

## 🐍 Python Analysis

### Launch Jupyter

```fish
# Activate environment
source .venv/bin/activate.fish

# Start Jupyter Lab
jupyter lab

# Or classic notebook
jupyter notebook
```

### Quick Python Script

```python
import pandas as pd
import duckdb

# Load data with DuckDB (fast)
df = duckdb.query("SELECT * FROM 'data/events.csv' WHERE ev_year >= 2020").to_df()

# Or with pandas
df = pd.read_csv('data/events.csv')

# Quick exploration
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())

# Filter and analyze
recent = df[df['ev_year'] >= 2020]
print(f"Accidents since 2020: {len(recent)}")
print(f"Total fatalities: {recent['inj_tot_f'].sum()}")

# Group by state
by_state = df.groupby('ev_state').size().sort_values(ascending=False)
print(by_state.head(10))
```

## 🦀 Rust Tools

### CSV Processing with xsv

```fish
# Count rows
xsv count data/events.csv

# Select specific columns
xsv select ev_id,ev_date,ev_state,ev_city data/events.csv | head

# Search
xsv search -s ev_state "CA" data/events.csv

# Statistics
xsv stats data/events.csv

# Frequency count
xsv frequency -s ev_state data/events.csv
```

### Polars CLI (if installed)

```fish
# Query with Polars
polars -c "df = pl.read_csv('data/events.csv'); df.filter(pl.col('ev_year') >= 2020)"
```

## 📈 Visualization Examples

### Python Quick Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Accidents by year
df.groupby('ev_year').size().plot(kind='line')
plt.title('Accidents by Year')
plt.show()

# Top 10 states
df['ev_state'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 States by Accident Count')
plt.show()

# Fatalities over time
df.groupby('ev_year')['inj_tot_f'].sum().plot(kind='bar')
plt.title('Total Fatalities by Year')
plt.show()
```

## 🔗 Common Table Joins

### SQL (DuckDB)

```sql
-- Events with aircraft details
SELECT e.ev_id, e.ev_date, e.ev_state, a.acft_make, a.acft_model
FROM 'data/events.csv' e
LEFT JOIN 'data/aircraft.csv' a ON e.ev_id = a.ev_id
WHERE e.ev_year >= 2020
LIMIT 100;

-- Events with findings (probable causes)
SELECT e.ev_id, e.ev_date, f.finding_description
FROM 'data/events.csv' e
LEFT JOIN 'data/Findings.csv' f ON e.ev_id = f.ev_id
WHERE e.inj_tot_f > 0;

-- Occurrences grouped
SELECT o.occurrence_code, COUNT(*) as count
FROM 'data/Occurrences.csv' o
GROUP BY o.occurrence_code
ORDER BY count DESC;
```

### Python (pandas)

```python
# Load tables
events = pd.read_csv('data/events.csv')
aircraft = pd.read_csv('data/aircraft.csv')
findings = pd.read_csv('data/Findings.csv')

# Join events with aircraft
events_aircraft = events.merge(aircraft, on='ev_id', how='left')

# Join with findings
full_data = events_aircraft.merge(findings, on='ev_id', how='left')

print(full_data.head())
```

## 🗺️ Geospatial Analysis

```python
import geopandas as gpd
import folium

# Load events with coordinates
events = pd.read_csv('data/events.csv')

# Filter events with valid coordinates
geo_events = events.dropna(subset=['latitude', 'longitude'])

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    geo_events,
    geometry=gpd.points_from_xy(geo_events.longitude, geo_events.latitude),
    crs='EPSG:4326'
)

# Create interactive map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add markers
for idx, row in gdf.head(100).iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"{row['ev_id']}: {row['ev_date']}",
        tooltip=row['ev_city']
    ).add_to(m)

m.save('outputs/accident_map.html')
```

## 📝 Code Lookup Reference

Quick reference for common NTSB codes (see `ref_docs/codman.pdf` for complete list):

### Occurrence Codes (100-430)
- 100: ABRUPT MANEUVER
- 110: AIRFRAME/COMPONENT/SYSTEM FAILURE/MALFUNCTION
- 200: ENGINE FAILURE
- 250: FIRE
- 300: FUEL EXHAUSTION
- 350: MIDAIR COLLISION
- 400: WEATHER

### Phase Codes (500-610)
- 500: STANDING
- 510: TAXI
- 520: TAKEOFF
- 530: CLIMB
- 540: CRUISE
- 560: DESCENT
- 580: APPROACH
- 590: LANDING
- 600: MANEUVERING

### Aircraft Components (10000-17710)
- 10000-11700: Airframe
- 12000-13500: Systems
- 14000-17710: Powerplant

## 🎯 Analysis Ideas

### Safety Trends
```python
# Accident rate by year
yearly = events.groupby('ev_year').agg({
    'ev_id': 'count',
    'inj_tot_f': 'sum',
    'inj_tot_s': 'sum'
})
```

### Causal Analysis
```python
# Most common occurrence types
occurrences = pd.read_csv('data/Occurrences.csv')
top_occurrences = occurrences['occurrence_code'].value_counts()
```

### Weather Impact
```python
# Filter weather-related
weather_events = events[events['ev_type'].str.contains('WX', na=False)]
```

### Geographic Patterns
```python
# Accidents by state
state_counts = events['ev_state'].value_counts()
```

## 💡 Performance Tips

### For Large Datasets
```python
# Use Polars instead of pandas (10x faster)
import polars as pl
df = pl.read_csv('data/events.csv')

# DuckDB for SQL analytics (very fast)
import duckdb
result = duckdb.query("SELECT * FROM 'data/events.csv' WHERE ev_year >= 2020").pl()

# Read CSV in chunks (pandas)
for chunk in pd.read_csv('data/events.csv', chunksize=10000):
    process(chunk)

# Use categorical dtypes
events['ev_state'] = events['ev_state'].astype('category')
```

## 📚 Help & Documentation

- **Complete tool list**: `TOOLS_AND_UTILITIES.md`
- **Repository guide**: `CLAUDE.md`
- **Database schema**: `ref_docs/eadmspub.pdf`
- **Coding manual**: `ref_docs/codman.pdf`
- **Example notebook**: `examples/starter_notebook.ipynb`

## 🔧 Troubleshooting

### "mdb-export: command not found"
`mdbtools` is in AUR, not in official repos. If installation fails with autoconf errors:
```fish
# Quick fix using provided script
./fix_mdbtools_pkgbuild.fish

# Or install via paru (may fail - see INSTALLATION.md for fix)
paru -S mdbtools
```

### Python module not found
```fish
source .venv/bin/activate.fish
pip install <module_name>
```

### Out of memory errors
- Use Polars instead of pandas
- Use DuckDB for querying
- Read data in chunks
- Convert to Parquet format

### Encoding issues
```python
# Try different encodings
df = pd.read_csv('file.csv', encoding='latin1')
df = pd.read_csv('file.csv', encoding='utf-8')
```

### Failed qsv installation
If qsv fails to compile during setup:

```fish
# Quick cleanup
./cleanup_qsv.fish

# Use xsv instead (already installed, similar functionality)
xsv --help

# Or install qsv from git
cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'
```

See **INSTALLATION.md** for detailed cleanup instructions.
