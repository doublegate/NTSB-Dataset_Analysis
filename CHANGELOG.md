# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub repository topics for improved discoverability (aviation, ntsb, accident-analysis, aviation-safety, data-analysis, python, fish-shell, mdb-database, duckdb, jupyter-notebook)
- Dependabot configuration for automated dependency updates (.github/dependabot.yml)
- Weekly automated checks for Python (pip) dependencies with 5 PR limit
- Weekly automated checks for GitHub Actions dependencies with 3 PR limit
- Repository badges in README.md (repo size, last commit)
- Project Status section in README.md with version and production-ready status
- GitHub Topics section in README.md listing all repository topics

### Changed
- Updated README.md git clone commands with correct GitHub username (doublegate)
- Enhanced README.md with repository status and topic information
- Updated CHANGELOG.md version links with correct GitHub username (doublegate)
- Improved documentation accuracy across all repository references

### Fixed
- Replaced placeholder YOUR_USERNAME with actual GitHub username (doublegate) in:
  - README.md line 92: git clone command
  - CHANGELOG.md lines 205-207: version reference links

### Planned
- GitHub Actions CI/CD pipeline for automated testing
- Docker container support for cross-platform compatibility
- Web-based dashboard for interactive data exploration
- Additional example analyses (time series forecasting, ML models)
- Integration with FAA aircraft registry data
- Automated monthly updates of avall.mdb database

## [1.0.0] - 2025-11-05

### Added

#### Database Files
- avall.mdb (537MB): Aviation accident data from 2008 to present, updated monthly
- Pre2008.mdb (893MB): Aviation accident data from 1982 to 2007
- PRE1982.MDB (188MB): Aviation accident data from 1962 to 1981

#### Scripts (scripts/)
- `extract_all_tables.fish`: Bulk CSV export from MDB files with proper database-prefixed naming
- `extract_table.fish`: Single table extraction with validation and table availability checking
- `show_database_info.fish`: Database inspection and metadata display
- `convert_to_sqlite.fish`: MDB to SQLite conversion for SQL analytics
- `quick_query.fish`: Fast DuckDB queries on CSV files
- `analyze_csv.fish`: Statistical analysis and CSV inspection with csvkit/xsv support
- `search_data.fish`: Text search across CSV datasets with column filtering
- `cleanup_qsv.fish`: Maintenance script for failed qsv installations
- `fix_mdbtools_pkgbuild.fish`: Automated fix for mdbtools AUR build failures

#### Documentation
- **README.md**: Comprehensive project overview with badges, table of contents, and examples
- **INSTALLATION.md**: Complete setup guide for CachyOS/Arch Linux with Fish shell
- **QUICKSTART.md**: Essential commands and common workflows reference guide
- **TOOLS_AND_UTILITIES.md**: Comprehensive tool catalog (Python, Rust, CLI)
- **CLAUDE.md**: Repository guidance for AI assistants with schema details
- **scripts/README.md**: Detailed script documentation with usage examples
- **scripts/EXTRACTION_FIX.md**: Table extraction bug fix documentation
- **scripts/MDBTOOLS_FIX_README.md**: mdbtools build issue resolution guide
- **examples/README.md**: Python analysis examples guide

#### Reference Documentation (ref_docs/)
- `codman.pdf`: NTSB aviation coding manual (occurrence/phase/cause codes)
- `eadmspub.pdf`: Database schema and entity relationship documentation
- `eadmspub_legacy.pdf`: Legacy schema for historical databases (Pre-2008, PRE-1982)
- `MDB_Release_Notes.pdf`: Database release notes and schema changes (Release 3.0)

#### Analysis Tools
- `setup.fish`: Automated environment setup (mdbtools, Python packages, Rust tools)
- `examples/quick_analysis.py`: Python pandas/DuckDB analysis script
- `examples/advanced_analysis.py`: Comprehensive statistical analysis with summary reports
- `examples/geospatial_analysis.py`: Interactive mapping and hotspot identification
- `examples/starter_notebook.ipynb`: Jupyter notebook with visualizations

#### Features
- Automated extraction from Microsoft Access (.mdb) databases
- Fast SQL queries using DuckDB directly on CSV files
- SQLite conversion support for complex joins and analysis
- Geospatial analysis with folium interactive maps
- Text search across all extracted CSV files
- Statistical analysis with csvkit and xsv
- Python virtual environment with comprehensive data science stack
- Fish shell abbreviations for common workflows

### Database Schema

#### Primary Tables
- **events**: Master table for accident events (keyed by `ev_id`)
- **aircraft**: Aircraft involved in accidents (keyed by `Aircraft_Key`)
- **Flight_Crew**: Flight crew information
- **injury**: Injury details for crew and passengers
- **Findings**: Investigation findings and probable causes
- **Occurrences**: Specific occurrence events during accidents
- **seq_of_events**: Sequence of events leading to accidents
- **Events_Sequence**: Event ordering and relationships
- **engines**: Engine details for involved aircraft
- **narratives**: Textual accident narratives and descriptions
- **NTSB_Admin**: Administrative metadata

#### Key Relationships
- `ev_id` links events across most tables
- `Aircraft_Key` identifies specific aircraft within events
- Foreign key relationships documented in entity relationship diagrams

### Coding System

#### Occurrence Codes (100-430)
Event types: ABRUPT MANEUVER, ENGINE FAILURE, MIDAIR COLLISION, FUEL EXHAUSTION, etc.

#### Phase of Operation (500-610)
Flight phases: STANDING, TAXI, TAKEOFF, CRUISE, APPROACH, LANDING, MANEUVERING, etc.

#### Section IA: Aircraft/Equipment Subjects (10000-21104)
Hierarchical codes for aircraft components:
- 10000-11700: Airframe (wings, fuselage, landing gear, flight controls)
- 12000-13500: Systems (hydraulic, electrical, environmental, fuel)
- 14000-17710: Powerplant (engines, propellers, turbines, exhaust)

#### Section IB: Performance/Operations (22000-25000)
- 22000-23318: Performance subjects (stall, altitude, airspeed, weather)
- 24000-24700: Operations (pilot technique, procedures, planning)
- 25000: ATC and maintenance

#### Section II: Direct Underlying Causes (30000-84200)
Detailed cause codes organized by aircraft component and failure mode

#### Section III: Indirect Underlying Causes (90000-93300)
Contributing factors: design, maintenance, organizational, regulatory

### Tools Ecosystem

#### Database Tools
- mdbtools: MDB file extraction (AUR package)
- DBeaver: Universal database GUI
- DuckDB: Fast analytical SQL queries (AUR package)
- SQLite: Converted database format

#### Python Libraries
- **Core**: pandas, polars, numpy, scipy, statsmodels, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly, altair
- **Geospatial**: geopandas, folium, geopy, shapely
- **Text Analysis**: nltk, spacy, wordcloud, textblob
- **Jupyter**: jupyterlab, ipython, jupyterlab-git
- **Dashboards**: streamlit, dash, panel
- **High Performance**: dask, pyarrow, fastparquet
- **Database**: duckdb, sqlalchemy
- **CLI**: csvkit

#### Rust Tools
- xsv: Fast CSV toolkit (stable, recommended)
- qsv: Extended CSV toolkit with advanced features (v9.1.0 has build issues)
- polars-cli: Polars DataFrame CLI
- datafusion-cli: SQL query engine

#### CLI Tools
- csvkit: CSV swiss army knife
- jq/yq: JSON/YAML querying
- bat: Better cat with syntax highlighting
- ripgrep: Faster grep
- fd: Better find
- fzf: Fuzzy finder

### Setup Requirements

#### System
- CachyOS/Arch Linux with Fish shell
- AUR helper: paru (for mdbtools, duckdb)
- Python 3.11+ with venv
- Rust toolchain (optional, for xsv and qsv)
- ~5GB free disk space

#### Known Issues
- qsv v9.1.0 has compilation issues (use xsv or git install as alternatives)
- mdbtools requires PKGBUILD patch for gettext m4 macros (automated fix provided via `fix_mdbtools_pkgbuild.fish`)

### Technical Details

#### Performance Optimizations
- Polars for 10x+ speedup over pandas on large datasets
- DuckDB for fast SQL analytics directly on CSV files
- Parquet format support for 5-10x better compression
- Dask for out-of-memory datasets larger than RAM

#### Database Coverage
- **Current data** (avall.mdb): Updated monthly from NTSB
- **Historical data**: Pre2008.mdb and PRE1982.MDB are static snapshots
- **Schema differences**: Legacy schema documented in `eadmspub_legacy.pdf`
- **Total coverage**: 1962 to present (60+ years of aviation accidents)

### Security & Privacy
- All NTSB data is public domain U.S. government data
- No personal identifiable information (PII) included
- Repository scripts and tools are MIT licensed

## [0.1.0] - Internal Development

### Added
- Initial project structure
- Basic extraction scripts
- Preliminary documentation

---

## Release Notes

### Version Numbering
- **Major.Minor.Patch** (e.g., 1.0.0)
- **Major**: Breaking changes, major feature additions
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, documentation updates

### Support
For questions, issues, or contributions:
- Open an issue on GitHub
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Check [INSTALLATION.md](INSTALLATION.md) for setup help

[Unreleased]: https://github.com/doublegate/NTSB-Dataset_Analysis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/doublegate/NTSB-Dataset_Analysis/releases/tag/v1.0.0
[0.1.0]: https://github.com/doublegate/NTSB-Dataset_Analysis/releases/tag/v0.1.0
