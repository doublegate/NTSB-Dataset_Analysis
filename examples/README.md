# Examples Directory

Python analysis examples and Jupyter notebooks for NTSB aviation accident data.

## 📋 Available Examples

### `quick_analysis.py`
Basic analysis using pandas and DuckDB.

**Features**:
- Load data efficiently with DuckDB
- Query recent events
- Example pandas operations

**Usage**:
```fish
source .venv/bin/activate.fish
python examples/quick_analysis.py
```

**Requirements**:
- pandas
- duckdb

---

### `advanced_analysis.py`
Comprehensive statistical analysis of accident data.

**Analyses Included**:
- Trends by year (events, fatalities, injuries)
- Accidents by aircraft type
- Geographic patterns (by state)
- Flight phase analysis
- Common causes/findings
- Fatal vs non-fatal comparison
- Seasonal patterns (by month)
- Export summary reports

**Usage**:
```fish
source .venv/bin/activate.fish
python examples/advanced_analysis.py
```

**Output**:
- Console: Statistical summaries
- File: `outputs/summary_report.csv`

**Requirements**:
- pandas
- numpy
- duckdb

---

### `geospatial_analysis.py`
Create interactive maps and geospatial visualizations.

**Features**:
- Interactive accident location map with clustering
- Heatmap showing accident density
- Fatal accidents map (sized by fatality count)
- Regional analysis

**Usage**:
```fish
source .venv/bin/activate.fish
python examples/geospatial_analysis.py
```

**Output** (in `outputs/` directory):
- `accident_map.html` - Interactive map with all accidents
- `accident_heatmap.html` - Density heatmap
- `fatal_accidents_map.html` - Fatal accidents only

**Requirements**:
- pandas
- duckdb
- geopandas
- folium

**Installation**:
```fish
pip install geopandas folium
```

---

### `starter_notebook.ipynb`
Jupyter notebook with visualizations and exploratory analysis.

**Sections**:
1. Data loading with DuckDB
2. Basic statistics
3. Accident trends over time
4. Event type analysis
5. Fatality analysis
6. Geographic visualization
7. Aircraft analysis
8. Advanced SQL queries

**Usage**:
```fish
source .venv/bin/activate.fish
jupyter lab
# Open starter_notebook.ipynb
```

**Requirements**:
- pandas
- numpy
- matplotlib
- seaborn
- duckdb
- jupyter/jupyterlab

---

## 🚀 Quick Start

### 1. Setup Environment
```fish
# Run setup (if not already done)
./setup.fish

# Activate Python environment
source .venv/bin/activate.fish
```

### 2. Extract Data
```fish
# Extract all tables from database
./scripts/extract_all_tables.fish datasets/avall.mdb
```

### 3. Run Examples
```fish
# Quick analysis
python examples/quick_analysis.py

# Advanced analysis with reports
python examples/advanced_analysis.py

# Geospatial maps (requires geopandas)
python examples/geospatial_analysis.py

# Interactive notebook
jupyter lab
# Then open starter_notebook.ipynb
```

## 📊 Example Workflows

### Workflow 1: Quick Data Exploration
```fish
# 1. Extract data
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Run quick analysis
source .venv/bin/activate.fish
python examples/quick_analysis.py

# 3. View in Jupyter for visualizations
jupyter lab
```

### Workflow 2: Comprehensive Analysis
```fish
# 1. Extract data (if not done)
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Run all analyses
source .venv/bin/activate.fish
python examples/advanced_analysis.py

# 3. Create maps
python examples/geospatial_analysis.py

# 4. View outputs
ls -lh outputs/
open outputs/accident_map.html  # Or use your browser
```

### Workflow 3: Custom Analysis in Jupyter
```fish
# 1. Extract data
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Start Jupyter
source .venv/bin/activate.fish
jupyter lab

# 3. Use starter_notebook.ipynb as template
# 4. Modify and experiment with your own analysis
```

## 🎯 Analysis Ideas

### Safety Trends
```python
# Analyze accident rate trends
import duckdb

df = duckdb.query("""
    SELECT
        ev_year,
        COUNT(*) as events,
        SUM(inj_tot_f) as fatalities
    FROM 'data/events.csv'
    GROUP BY ev_year
    ORDER BY ev_year
""").to_df()

# Plot trends
df.plot(x='ev_year', y=['events', 'fatalities'])
```

### Aircraft Comparison
```python
# Compare accident rates by aircraft make
df = duckdb.query("""
    SELECT
        a.acft_make,
        COUNT(*) as accident_count,
        SUM(e.inj_tot_f) as total_fatalities
    FROM 'data/events.csv' e
    JOIN 'data/aircraft.csv' a ON e.ev_id = a.ev_id
    WHERE a.acft_make IN ('BOEING', 'CESSNA', 'PIPER', 'BEECH')
    GROUP BY a.acft_make
""").to_df()
```

### Seasonal Patterns
```python
# Analyze accidents by month
df = duckdb.query("""
    SELECT
        CAST(SUBSTR(ev_date, 6, 2) AS INTEGER) as month,
        COUNT(*) as count
    FROM 'data/events.csv'
    WHERE ev_date IS NOT NULL
    GROUP BY month
    ORDER BY month
""").to_df()

df.plot(kind='bar', x='month', y='count')
```

### Geographic Analysis
```python
# Find accident hotspots
df = duckdb.query("""
    SELECT
        ev_state,
        ev_city,
        COUNT(*) as accidents,
        SUM(inj_tot_f) as fatalities
    FROM 'data/events.csv'
    WHERE latitude IS NOT NULL
    GROUP BY ev_state, ev_city
    HAVING accidents > 10
    ORDER BY accidents DESC
""").to_df()
```

## 📚 Additional Resources

### Data Science Libraries
- **pandas**: DataFrame manipulation
- **polars**: Faster alternative to pandas
- **DuckDB**: SQL analytics on CSV
- **matplotlib/seaborn**: Visualization
- **plotly**: Interactive plots
- **geopandas**: Geospatial analysis

### Jupyter Tips
- Use `Shift+Enter` to run cells
- Use `Tab` for autocomplete
- Use `?` after function for help: `duckdb.query?`
- Save notebooks frequently

### Performance Tips
- Use DuckDB for large queries (faster than pandas)
- Use `polars` instead of pandas for 10x speedup
- Load only needed columns: `SELECT col1, col2 FROM ...`
- Filter early: `WHERE year >= 2020`

## 🔗 Related Documentation

- **scripts/README.md** - Fish shell helper scripts
- **QUICKSTART.md** - Quick reference guide
- **TOOLS_AND_UTILITIES.md** - Tool installation guide
- **CLAUDE.md** - Database schema and structure
