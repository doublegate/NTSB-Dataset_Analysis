# Scripts Quick Reference

Complete reference for all scripts and utilities in this repository.

## 📁 Directory Structure

```
NTSB_Datasets/
├── scripts/          # Fish shell helper scripts
├── examples/         # Python analysis examples
├── datasets/         # MDB database files
├── ref_docs/         # PDF documentation
├── data/            # Extracted CSV files (auto-created)
├── outputs/         # Analysis results (auto-created)
└── figures/         # Generated plots (auto-created)
```

## 🐚 Fish Shell Scripts (`scripts/`)

All Fish scripts are located in `scripts/` directory and are executable.

### Database Operations

| Script | Purpose | Usage |
|--------|---------|-------|
| `extract_all_tables.fish` | Extract all tables from MDB to CSV | `./scripts/extract_all_tables.fish datasets/avall.mdb` |
| `extract_table.fish` | Extract single table from MDB | `./scripts/extract_table.fish datasets/avall.mdb events` |
| `show_database_info.fish` | Display database information | `./scripts/show_database_info.fish datasets/avall.mdb` |
| `convert_to_sqlite.fish` | Convert MDB to SQLite format | `./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db` |

### Data Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| `quick_query.fish` | Run SQL queries on CSV with DuckDB | `./scripts/quick_query.fish "SELECT COUNT(*) FROM 'data/events.csv'"` |
| `analyze_csv.fish` | Show CSV statistics | `./scripts/analyze_csv.fish data/events.csv` |
| `search_data.fish` | Search text in CSV files | `./scripts/search_data.fish "Boeing"` |

## 🐍 Python Examples (`examples/`)

All Python scripts require the virtual environment to be activated:
```fish
source .venv/bin/activate.fish
```

### Analysis Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `quick_analysis.py` | Basic data exploration | Console output |
| `advanced_analysis.py` | Comprehensive statistical analysis | Console + `outputs/summary_report.csv` |
| `geospatial_analysis.py` | Interactive maps and geospatial viz | `outputs/*.html` maps |

### Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| `starter_notebook.ipynb` | Interactive analysis with visualizations |

## ⚙️ Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.fish` | Install all tools and dependencies | `./setup.fish` |

## 📝 Common Commands

### First Time Setup
```fish
# 1. Install all tools
./setup.fish

# 2. View database contents
./scripts/show_database_info.fish datasets/avall.mdb

# 3. Extract all data
./scripts/extract_all_tables.fish datasets/avall.mdb

# 4. Activate Python environment
source .venv/bin/activate.fish
```

### Quick Data Extraction
```fish
# Extract specific tables only
./scripts/extract_table.fish datasets/avall.mdb events
./scripts/extract_table.fish datasets/avall.mdb aircraft
./scripts/extract_table.fish datasets/avall.mdb Findings
```

### Quick Analysis
```fish
# View CSV info
./scripts/analyze_csv.fish data/events.csv

# Run SQL query
./scripts/quick_query.fish "SELECT ev_year, COUNT(*) FROM 'data/events.csv' GROUP BY ev_year ORDER BY ev_year DESC LIMIT 10"

# Search for text
./scripts/search_data.fish "Cessna"
```

### Python Analysis
```fish
source .venv/bin/activate.fish

# Quick analysis
python examples/quick_analysis.py

# Advanced analysis with reports
python examples/advanced_analysis.py

# Create interactive maps
python examples/geospatial_analysis.py

# Open Jupyter
jupyter lab
```

### Database Conversion
```fish
# Convert to SQLite for better query performance
./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db

# Query SQLite
sqlite3 data/avall.db "SELECT * FROM events WHERE ev_year >= 2023 LIMIT 10;"
```

## 🎯 Common Workflows

### Workflow 1: Quick Exploration
```fish
# 1. Show what's in database
./scripts/show_database_info.fish datasets/avall.mdb

# 2. Extract tables you need
./scripts/extract_table.fish datasets/avall.mdb events
./scripts/extract_table.fish datasets/avall.mdb aircraft

# 3. Quick statistics
./scripts/analyze_csv.fish data/events.csv

# 4. Run some queries
./scripts/quick_query.fish "SELECT COUNT(*) FROM 'data/events.csv'"
```

### Workflow 2: Full Analysis
```fish
# 1. Extract all data
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Activate Python
source .venv/bin/activate.fish

# 3. Run analyses
python examples/advanced_analysis.py

# 4. Create maps
python examples/geospatial_analysis.py

# 5. View outputs
ls -lh outputs/
open outputs/accident_map.html
```

### Workflow 3: Interactive Exploration
```fish
# 1. Extract data (if needed)
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Start Jupyter
source .venv/bin/activate.fish
jupyter lab

# 3. Open starter_notebook.ipynb
# 4. Experiment and create custom analysis
```

### Workflow 4: SQLite for Power Users
```fish
# 1. Convert to SQLite (one time)
./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db

# 2. Query with full SQL
sqlite3 data/avall.db

# Example complex query:
sqlite> SELECT
          e.ev_year,
          COUNT(*) as accidents,
          SUM(e.inj_tot_f) as fatalities,
          a.acft_make
        FROM events e
        JOIN aircraft a ON e.ev_id = a.ev_id
        WHERE e.ev_year >= 2020
        GROUP BY e.ev_year, a.acft_make
        ORDER BY accidents DESC
        LIMIT 20;
```

## 💡 Pro Tips

### Fish Shell Abbreviations
Add to `~/.config/fish/config.fish`:
```fish
# NTSB shortcuts
abbr -a nq './scripts/quick_query.fish'
abbr -a ne './scripts/extract_table.fish'
abbr -a ns './scripts/search_data.fish'
abbr -a ndb './scripts/show_database_info.fish'
abbr -a npy 'source .venv/bin/activate.fish && python'
```

Then use:
```fish
nq "SELECT COUNT(*) FROM 'data/events.csv'"
ne datasets/avall.mdb events
ns "Boeing 737"
```

### Performance Optimization
```fish
# Use Polars instead of pandas for large datasets
pip install polars

# Convert CSV to Parquet for faster loading
python -c "import pandas as pd; pd.read_csv('data/events.csv').to_parquet('data/events.parquet')"

# Query Parquet with DuckDB (much faster)
./scripts/quick_query.fish "SELECT * FROM 'data/events.parquet' WHERE ev_year >= 2023"
```

### Batch Processing
```fish
# Extract from all three databases
for db in datasets/*.mdb datasets/*.MDB
    echo "Processing $db..."
    ./scripts/extract_all_tables.fish "$db"
end

# Or specific table from all databases
for db in datasets/*.mdb datasets/*.MDB
    ./scripts/extract_table.fish "$db" events
end
```

### Piping and Combining Tools
```fish
# Query and analyze results
./scripts/quick_query.fish "SELECT * FROM 'data/events.csv' WHERE ev_state = 'CA'" | csvstat

# Query to CSV for further processing
./scripts/quick_query.fish "SELECT * FROM 'data/events.csv' WHERE ev_year >= 2023" > data/recent_events.csv
./scripts/analyze_csv.fish data/recent_events.csv

# Search and export
./scripts/search_data.fish "Boeing" > outputs/boeing_accidents.txt
```

## 🆘 Troubleshooting

### Script not found
```fish
chmod +x scripts/*.fish examples/*.py
```

### Fish syntax errors
All scripts use proper Fish syntax (no bash heredocs). If you see errors, verify Fish version:
```fish
fish --version  # Should be 3.0+
```

### Python module not found
```fish
source .venv/bin/activate.fish
pip install <module_name>
```

### DuckDB not installed
```fish
sudo pacman -S duckdb
```

### mdbtools not installed
```fish
sudo pacman -S mdbtools
```

### Data directory empty
```fish
./scripts/extract_all_tables.fish datasets/avall.mdb
```

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview |
| `QUICKSTART.md` | Quick reference for common tasks |
| `CLAUDE.md` | Database schema and structure |
| `TOOLS_AND_UTILITIES.md` | Complete tool installation guide |
| `SCRIPTS_REFERENCE.md` | This file - comprehensive script reference |
| `scripts/README.md` | Detailed Fish script documentation |
| `examples/README.md` | Python example documentation |

## 🔗 External Resources

- [NTSB Aviation Database](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)
- [MDB Tools Documentation](https://github.com/mdbtools/mdbtools)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Fish Shell Documentation](https://fishshell.com/docs/current/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [GeoPandas Documentation](https://geopandas.org/)
