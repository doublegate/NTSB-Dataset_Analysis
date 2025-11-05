# Installation and Setup Guide

Complete installation guide for NTSB Aviation Accident Database analysis environment on CachyOS Linux with Fish shell.

## 📋 Prerequisites

- CachyOS Linux (Arch-based)
- Fish shell
- Internet connection
- ~5GB free disk space

## 🚀 Quick Setup (Automated)

The easiest way to get started:

```fish
# Clone or download this repository
cd /path/to/NTSB_Datasets

# Run automated setup
./setup.fish
```

This will install:
- ✅ System packages (sqlite, python, gdal, etc.)
- ✅ AUR packages (mdbtools, duckdb, DBeaver, Quarto) if paru is installed
- ✅ Python virtual environment
- ✅ All Python packages (pandas, polars, duckdb, jupyter, etc.)
- ✅ Rust tools (xsv, qsv, polars-cli) if cargo is installed
- ✅ NLP models for text analysis

## 📦 Manual Installation

If you prefer to install components individually:

### 1. System Packages
```fish
sudo pacman -S --needed \
    sqlite \
    postgresql \
    python \
    python-pip \
    gdal \
    hdf5 \
    jq \
    yq \
    bat \
    ripgrep \
    fd \
    fzf \
    cmake \
    base-devel \
    gettext \
    autoconf-archive \
    txt2man
```

**Note**:
- `cmake` and `base-devel` are required for building Rust tools like `qsv`
- `gettext`, `autoconf-archive`, and `txt2man` are required for building `mdbtools` from AUR

### 2. AUR Packages (Required + Optional)
```fish
# Requires paru (recommended AUR helper)
# mdbtools and duckdb are REQUIRED for this project
# dbeaver and quarto-cli are optional
paru -S --needed mdbtools duckdb dbeaver quarto-cli
```

**Important**: Both `mdbtools` and `duckdb` are not in the official Arch repositories and must be installed from AUR:
- **mdbtools** - Required for extracting data from .mdb database files
- **duckdb** - Required for fast SQL queries on CSV files (used by scripts)

### 3. Python Environment
```fish
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate.fish

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### 4. Python Packages

**Core Data Science**:
```fish
pip install pandas polars numpy scipy statsmodels scikit-learn
```

**Visualization**:
```fish
pip install matplotlib seaborn plotly altair
```

**Geospatial**:
```fish
pip install geopandas folium geopy shapely
```

**Text Analysis**:
```fish
pip install nltk spacy wordcloud textblob
python -m spacy download en_core_web_sm
```

**Jupyter & Notebooks**:
```fish
pip install jupyter jupyterlab ipython jupyterlab-git
```

**Dashboards**:
```fish
pip install streamlit dash panel
```

**High Performance**:
```fish
pip install dask[complete] pyarrow fastparquet
```

**Database Tools**:
```fish
pip install duckdb sqlalchemy
```

**CLI Tools**:
```fish
pip install csvkit
```

**Utilities**:
```fish
pip install python-dateutil pytz
```

### 5. Rust Tools (Optional)

If you have Rust/Cargo installed:
```fish
# Install each tool separately for better control
cargo install xsv --locked                           # CSV toolkit (simpler, stable)
cargo install qsv --locked --features feature_capable # CSV toolkit (advanced features)
cargo install polars-cli --locked                     # Polars CLI for DataFrames
cargo install datafusion-cli --locked                 # SQL query engine
```

**⚠️ Known Issue with qsv v9.1.0**: The current version (9.1.0) has compilation errors. If installation fails:

**Option 1 - Use xsv instead** (recommended):
```fish
cargo install xsv --locked  # Similar tool, more stable
```

**Option 2 - Install qsv from git** (latest fixes):
```fish
cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'
```

**Option 3 - Wait for qsv v9.1.1**: The issue is known to maintainers and will be fixed in the next release.

**Note**: `qsv` requires the `feature_capable` feature for the full binary. Use `--features lite` for a lighter version.

To install Rust:
```fish
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## 🗂️ Data Extraction

After installation, extract data from the MDB databases:

```fish
# Extract all tables from current database (2008-present)
./scripts/extract_all_tables.fish datasets/avall.mdb

# Optional: Extract from historical databases
./scripts/extract_all_tables.fish datasets/Pre2008.mdb
./scripts/extract_all_tables.fish datasets/PRE1982.MDB
```

## ✅ Verify Installation

### Check System Tools
```fish
# mdbtools
mdb-tables --version

# DuckDB
duckdb --version

# SQLite
sqlite3 --version

# Python
python --version

# Cargo (optional)
cargo --version
```

### Check Python Packages
```fish
source .venv/bin/activate.fish

# List installed packages
pip list

# Test imports
python -c "import pandas, polars, duckdb, matplotlib; print('All packages OK')"
```

### Check Scripts
```fish
# List all scripts
ls -lh scripts/*.fish examples/*.py

# Test database info script
./scripts/show_database_info.fish datasets/avall.mdb

# Test extract script (dry run)
./scripts/extract_table.fish datasets/avall.mdb events
```

## 🎯 Post-Installation

### 1. Activate Virtual Environment

Every time you start a new shell session:
```fish
source .venv/bin/activate.fish
```

**Optional**: Add to `~/.config/fish/config.fish` for auto-activation:
```fish
# Auto-activate NTSB venv when in project directory
if test -d ~/Code/NTSB_Datasets/.venv; and test "$PWD" = ~/Code/NTSB_Datasets
    source .venv/bin/activate.fish
end
```

### 2. Set Up Fish Abbreviations

Add to `~/.config/fish/config.fish`:
```fish
# NTSB shortcuts
abbr -a ntsb-setup 'cd ~/Code/NTSB_Datasets && source .venv/bin/activate.fish'
abbr -a ntsb-query './scripts/quick_query.fish'
abbr -a ntsb-extract './scripts/extract_table.fish'
abbr -a ntsb-search './scripts/search_data.fish'
abbr -a ntsb-jupyter 'source .venv/bin/activate.fish && jupyter lab'
```

### 3. Test Analysis Pipeline

```fish
# 1. Extract data
./scripts/extract_all_tables.fish datasets/avall.mdb

# 2. Run quick analysis
source .venv/bin/activate.fish
python examples/quick_analysis.py

# 3. Open Jupyter
jupyter lab
```

## 🔧 Troubleshooting

### "mdbtools not found" or "target not found: mdbtools"
`mdbtools` is in AUR, not in the official Arch repositories:

```fish
# Install using paru (recommended)
paru -S mdbtools

# Or using yay
yay -S mdbtools

# Or manually from AUR
git clone https://aur.archlinux.org/mdbtools.git
cd mdbtools
makepkg -si
```

**Note**: If you don't have an AUR helper installed, install paru first:
```fish
sudo pacman -S --needed base-devel git
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -si
```

### mdbtools build failure: "possibly undefined macro: AC_LIB_PREPARE_PREFIX"
This error occurs because the mdbtools AUR PKGBUILD needs to be patched to find gettext m4 macros.

**Root Cause:** The PKGBUILD uses `autoreconf -i -f` but needs `-I /usr/share/gettext/m4` to locate gettext macros.

**Quick Fix - Use provided script:**
```fish
./fix_mdbtools_pkgbuild.fish
```

**Manual Fix:**
```fish
# 1. Clone mdbtools from AUR
cd /tmp
git clone https://aur.archlinux.org/mdbtools.git
cd mdbtools

# 2. Edit PKGBUILD - change line in prepare() function from:
#    autoreconf -i -f
# to:
#    autoreconf -i -f -I /usr/share/gettext/m4

# 3. Build and install
makepkg -si
```

**Error message looks like:**
```
configure:21182: error: possibly undefined macro: AC_LIB_PREPARE_PREFIX
configure:21183: error: possibly undefined macro: AC_LIB_RPATH
autoreconf: error: /usr/bin/autoconf failed with exit status: 1
==> ERROR: A failure occurred in prepare().
```

**Related Issue:** https://github.com/mdbtools/mdbtools/issues/370

### "DuckDB not found" or "target not found: duckdb"
`duckdb` is in AUR, not in the official Arch repositories:

```fish
# Install using paru (recommended)
paru -S duckdb

# Or using yay
yay -S duckdb

# Or manually from AUR
git clone https://aur.archlinux.org/duckdb.git
cd duckdb
makepkg -si
```

### "Python module not found"
```fish
source .venv/bin/activate.fish
pip install <module_name>
```

### "Permission denied" for scripts
```fish
chmod +x scripts/*.fish examples/*.py
```

### Virtual environment not activating
```fish
# Recreate venv
rm -rf .venv
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt  # if you create one
```

### Geospatial packages failing to install

GDAL dependency issues:
```fish
# Install system GDAL first
sudo pacman -S gdal

# Then install Python packages
pip install geopandas
```

### Out of disk space

The databases are large (1.6GB total). CSV exports will add ~2-3GB more.

```fish
# Check disk space
df -h

# Extract only needed tables
./scripts/extract_table.fish datasets/avall.mdb events
./scripts/extract_table.fish datasets/avall.mdb aircraft
```

### Slow Python package installation

Use faster mirror:
```fish
pip install --index-url https://pypi.org/simple <package>
```

Or install in batches instead of all at once.

### Failed qsv installation (v9.1.0 compilation error)

If qsv fails to compile during setup, clean up and use alternatives:

**Quick cleanup using script**:
```fish
./cleanup_qsv.fish
```

**Manual cleanup**:
```fish
# 1. Uninstall qsv (if partially installed)
cargo uninstall qsv

# 2. Remove temporary build directories
rm -rf /tmp/cargo-install*

# 3. Clean qsv from cargo registry
find ~/.cargo/registry/cache -name "qsv-*.crate" -delete
find ~/.cargo/registry/src -type d -name "qsv-*" -exec rm -rf {} +

# 4. Check cache size
du -sh ~/.cargo/registry
```

**Deep cleanup of all unused Rust dependencies** (optional):
```fish
# Install cargo-cache tool
cargo install cargo-cache

# Auto-clean unused dependencies
cargo cache --autoclean

# Or for more aggressive cleanup
cargo cache --autoclean-expensive
```

**Alternatives to qsv**:
1. **Use xsv** (already installed) - Similar functionality, more stable
2. **Install qsv from git** (has fixes): `cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'`
3. **Wait for qsv v9.1.1** - Official fix coming soon

## 🎓 Next Steps

After successful installation:

1. **Read the documentation**:
   - `README.md` - Project overview
   - `QUICKSTART.md` - Quick reference
   - `SCRIPTS_REFERENCE.md` - Complete script guide

2. **Extract and explore data**:
   ```fish
   ./scripts/show_database_info.fish datasets/avall.mdb
   ./scripts/extract_all_tables.fish datasets/avall.mdb
   ```

3. **Run example analyses**:
   ```fish
   source .venv/bin/activate.fish
   python examples/quick_analysis.py
   python examples/advanced_analysis.py
   ```

4. **Start Jupyter for interactive work**:
   ```fish
   jupyter lab
   # Open examples/starter_notebook.ipynb
   ```

5. **Explore the tools**:
   - Review `TOOLS_AND_UTILITIES.md` for advanced tools
   - Try different Fish scripts in `scripts/`
   - Experiment with SQL queries using `quick_query.fish`

## 📚 Additional Resources

- **Official Docs**:
  - [NTSB Aviation Database](https://www.ntsb.gov/Pages/AviationQueryV2.aspx)
  - [Fish Shell](https://fishshell.com/)
  - [DuckDB](https://duckdb.org/)
  - [Pandas](https://pandas.pydata.org/)

- **This Repository**:
  - `CLAUDE.md` - Database schema reference
  - `scripts/README.md` - Script documentation
  - `examples/README.md` - Python examples guide
  - `TOOLS_AND_UTILITIES.md` - Comprehensive tool list

## 🤝 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all prerequisites are installed
4. Check file permissions (`chmod +x scripts/*.fish`)
5. Ensure you're in the correct directory
6. Verify virtual environment is activated for Python commands

Common fixes:
```fish
# Reset permissions
chmod +x scripts/*.fish examples/*.py setup.fish

# Recreate virtual environment
rm -rf .venv && python -m venv .venv && source .venv/bin/activate.fish

# Update packages
pip install --upgrade pip
pip install --upgrade pandas duckdb jupyter
```
