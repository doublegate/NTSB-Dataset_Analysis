# Scripts Directory

Helper scripts for working with NTSB aviation accident data.

## 📋 Available Scripts

### Database Operations

#### `extract_all_tables.fish`
Extract all tables from an MDB database to CSV files.

```fish
./scripts/extract_all_tables.fish datasets/avall.mdb
```

**Output**: All tables exported to `data/` directory as CSV files with database prefix.

**Naming Convention**: Files are named as `{database}-{table}.csv`
- `avall-events.csv` - events table from avall.mdb
- `Pre2008-aircraft.csv` - aircraft table from Pre2008.mdb
- `PRE1982-tblFirstHalf.csv` - tblFirstHalf table from PRE1982.MDB

This prevents conflicts when extracting from multiple databases.

---

#### `extract_table.fish`
Extract a specific table from an MDB database.

```fish
./scripts/extract_table.fish datasets/avall.mdb events
./scripts/extract_table.fish datasets/Pre2008.mdb aircraft
```

**Features**:
- Validates table exists before export
- Shows available tables if name is invalid
- Displays row count and file size after export
- Uses database prefix in filename (e.g., `avall-events.csv`)

---

#### `show_database_info.fish`
Display information about an MDB database.

```fish
./scripts/show_database_info.fish datasets/avall.mdb
```

**Output**:
- Database file size
- List of all tables
- Table count

---

#### `convert_to_sqlite.fish`
Convert an entire MDB database to SQLite format.

```fish
./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db
```

**Benefits**:
- Easier querying with standard SQL
- Better performance than CSV
- No need for mdbtools after conversion

**Usage after conversion**:
```fish
sqlite3 data/avall.db
sqlite> SELECT * FROM events WHERE ev_year >= 2020 LIMIT 10;
```

---

### Data Analysis

#### `quick_query.fish`
Execute SQL queries on CSV files using DuckDB.

```fish
# Count total events
./scripts/quick_query.fish "SELECT COUNT(*) FROM 'data/events.csv'"

# Events by state
./scripts/quick_query.fish "SELECT ev_state, COUNT(*) as count FROM 'data/events.csv' GROUP BY ev_state ORDER BY count DESC LIMIT 10"

# Fatal accidents in 2023
./scripts/quick_query.fish "SELECT * FROM 'data/events.csv' WHERE ev_year = 2023 AND inj_tot_f > 0"
```

**Requirements**: DuckDB (`sudo pacman -S duckdb`)

---

#### `analyze_csv.fish`
Show statistics and information about a CSV file.

```fish
./scripts/analyze_csv.fish data/events.csv
```

**Output**:
- File size
- Row count
- Column names
- Statistical summary (if csvstat/xsv installed)

**Enhanced output with**:
- `pip install csvkit` for detailed statistics
- `cargo install xsv` for fast Rust-based stats

---

#### `search_data.fish`
Search for text across CSV files.

```fish
# Search all columns in all CSV files
./scripts/search_data.fish "Boeing"

# Search specific column
./scripts/search_data.fish "Los Angeles" ev_city

# Search and filter by file
./scripts/search_data.fish "Cessna" | grep events.csv
```

**Requirements**:
- For column search: csvkit (`pip install csvkit`)
- For general search: grep (installed by default)

---

### Maintenance & Troubleshooting

#### `cleanup_qsv.fish`
Clean up failed qsv installation and unused Rust dependencies.

```fish
# Quick cleanup of failed qsv installation
./scripts/cleanup_qsv.fish
```

**What it does**:
- Removes qsv binary (if partially installed)
- Cleans temporary build artifacts from `/tmp/`
- Removes qsv from cargo registry cache
- Shows current cache size

**When to use**:
- qsv compilation failed during setup
- Want to retry qsv installation
- Need to free up disk space from failed Rust builds

**Alternative manual cleanup**:
```fish
cargo uninstall qsv
rm -rf /tmp/cargo-install*
find ~/.cargo/registry/cache -name "qsv-*.crate" -delete
find ~/.cargo/registry/src -type d -name "qsv-*" -exec rm -rf {} +
```

**Deep cleanup** (removes ALL unused Rust dependencies):
```fish
cargo install cargo-cache
cargo cache --autoclean
```

See **INSTALLATION.md** and **TOOLS_AND_UTILITIES.md** for more details on qsv installation issues.

---

#### `fix_mdbtools_pkgbuild.fish`
Fix and install mdbtools when AUR build fails with autoconf errors.

```fish
# Automated fix and installation
./scripts/fix_mdbtools_pkgbuild.fish
```

**What it does**:
1. Clones mdbtools from AUR to a temporary directory
2. Patches the PKGBUILD to add `-I /usr/share/gettext/m4` flag
3. Builds and installs mdbtools with the fix
4. Cleans up temporary directory

**When to use**:
- mdbtools installation fails with "possibly undefined macro: AC_LIB_PREPARE_PREFIX"
- `paru -S mdbtools` or `yay -S mdbtools` fails during prepare() step
- After running setup.fish and mdbtools build fails

**Related documentation**:
- See **INSTALLATION.md** troubleshooting section
- See **MDBTOOLS_FIX_README.md** for detailed explanation
- Upstream issue: https://github.com/mdbtools/mdbtools/issues/370

---

#### `EXTRACTION_FIX.md`
Documentation file explaining the table extraction script fixes.

**Covers**:
- Root cause of extraction filename issues
- Solution applied to extract scripts
- New naming convention: `{database}-{table}.csv`
- Verification steps

---

## 🔧 Common Workflows

### Initial Setup
```fish
# 1. Show what's in the database
./scripts/show_database_info.fish datasets/avall.mdb

# 2. Extract all tables
./scripts/extract_all_tables.fish datasets/avall.mdb

# 3. Or extract specific tables only
./scripts/extract_table.fish datasets/avall.mdb events
./scripts/extract_table.fish datasets/avall.mdb aircraft
./scripts/extract_table.fish datasets/avall.mdb Findings
```

### Quick Analysis
```fish
# Analyze a CSV file
./scripts/analyze_csv.fish data/events.csv

# Run SQL queries
./scripts/quick_query.fish "SELECT ev_year, COUNT(*) FROM 'data/events.csv' GROUP BY ev_year ORDER BY ev_year DESC"

# Search for specific aircraft
./scripts/search_data.fish "Boeing 737"
```

### Advanced: SQLite Conversion
```fish
# Convert to SQLite for better performance
./scripts/convert_to_sqlite.fish datasets/avall.mdb data/avall.db

# Query SQLite directly
sqlite3 data/avall.db << EOF
SELECT
    e.ev_id,
    e.ev_date,
    e.ev_state,
    a.acft_make,
    a.acft_model
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
WHERE e.ev_year >= 2020
LIMIT 10;
EOF
```

## 📊 Example Queries

### DuckDB Queries (via quick_query.fish)

**Accidents by year**:
```fish
./scripts/quick_query.fish "SELECT ev_year as Year, COUNT(*) as Accidents FROM 'data/events.csv' GROUP BY ev_year ORDER BY ev_year DESC LIMIT 10"
```

**Top 10 states by accidents**:
```fish
./scripts/quick_query.fish "SELECT ev_state as State, COUNT(*) as Count FROM 'data/events.csv' GROUP BY ev_state ORDER BY Count DESC LIMIT 10"
```

**Fatal accidents summary**:
```fish
./scripts/quick_query.fish "SELECT ev_year, SUM(inj_tot_f) as Fatalities FROM 'data/events.csv' WHERE inj_tot_f > 0 GROUP BY ev_year ORDER BY ev_year DESC"
```

**Join events with aircraft**:
```fish
./scripts/quick_query.fish "SELECT e.ev_id, e.ev_date, a.acft_make, a.acft_model FROM 'data/events.csv' e JOIN 'data/aircraft.csv' a ON e.ev_id = a.ev_id WHERE e.ev_year = 2023 LIMIT 20"
```

**Most common occurrence codes**:
```fish
./scripts/quick_query.fish "SELECT occurrence_code, COUNT(*) as count FROM 'data/Occurrences.csv' GROUP BY occurrence_code ORDER BY count DESC LIMIT 10"
```

## 🐍 Python Examples

See `examples/` directory for Python analysis scripts:
- `quick_analysis.py` - Basic analysis with pandas and DuckDB
- `starter_notebook.ipynb` - Jupyter notebook with visualizations

## 💡 Tips

1. **Extract once, query many times**: Extract tables to CSV once, then use `quick_query.fish` for fast SQL queries

2. **Use SQLite for complex analysis**: Convert to SQLite if you need to run many complex joins

3. **Pipe output for further processing**:
   ```fish
   ./scripts/quick_query.fish "SELECT * FROM 'data/events.csv' WHERE ev_state = 'CA'" | csvcut -c ev_id,ev_date,ev_city
   ```

4. **Combine with other tools**:
   ```fish
   # Export query results to new CSV
   ./scripts/quick_query.fish "SELECT * FROM 'data/events.csv' WHERE ev_year >= 2020" > data/recent_events.csv

   # Analyze results
   ./scripts/analyze_csv.fish data/recent_events.csv
   ```

5. **Use Fish abbreviations** for frequently used commands:
   ```fish
   # Add to ~/.config/fish/config.fish
   abbr -a ntsb-query './scripts/quick_query.fish'
   abbr -a ntsb-extract './scripts/extract_table.fish'
   ```

## 🔗 Related Documentation

- **QUICKSTART.md** - Quick reference for common operations
- **TOOLS_AND_UTILITIES.md** - Complete tool installation guide
- **CLAUDE.md** - Repository structure and database schema
