# NTSB Aviation Accident Dashboard

**Interactive Streamlit dashboard for 64 years of aviation safety data (1962-2025)**

Production-ready analytics platform with 5 pages, 20+ visualizations, and interactive maps.

## Features

- **📊 Overview**: Summary statistics and key metrics (179,809 events)
- **📈 Temporal Trends**: Time series analysis, seasonality, and decade comparison
- **🗺️ Geographic Analysis**: Interactive maps (markers, heatmap, clusters)
- **✈️ Aircraft Safety**: Aircraft type-specific risk assessment
- **🔍 Cause Factors**: Root cause analysis and contributing factors

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 18.0 with PostGIS
- NTSB Aviation Accident Database (ntsb_aviation)
- Virtual environment at `/home/parobek/Code/NTSB_Datasets/.venv`

### Installation

```bash
# Navigate to project root
cd /home/parobek/Code/NTSB_Datasets

# Activate virtual environment
source .venv/bin/activate

# Install dashboard dependencies
cd dashboard
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# From dashboard/ directory
streamlit run app.py

# Or from project root
cd /home/parobek/Code/NTSB_Datasets
source .venv/bin/activate
cd dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
dashboard/
├── app.py                          # Main application entry point
├── pages/
│   ├── 1_📊_Overview.py           # Overview dashboard
│   ├── 2_📈_Temporal_Trends.py    # Time series analysis
│   ├── 3_🗺️_Geographic_Analysis.py # Interactive maps
│   ├── 4_✈️_Aircraft_Safety.py    # Aircraft type analysis
│   └── 5_🔍_Cause_Factors.py      # Cause factor analysis
├── components/
│   ├── __init__.py
│   ├── filters.py                 # Shared filter widgets
│   ├── charts.py                  # Plotly chart functions
│   └── maps.py                    # Folium map functions
├── utils/
│   ├── __init__.py
│   ├── database.py                # Connection pooling
│   └── queries.py                 # Cached query functions
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Dashboard Pages

### 1. Overview (📊)

**Purpose**: High-level statistics and key insights

**Features**:
- 5 hero metrics (total events, years coverage, fatalities, etc.)
- Long-term trend chart with 5-year moving average
- Geographic choropleth map
- Top aircraft makes and weather conditions
- Key findings and safety improvements

**Data Sources**:
- `get_summary_stats()`: Overall statistics
- `get_yearly_stats()`: Annual trends (mv_yearly_stats)
- `get_state_stats()`: State-level data (mv_state_stats)
- `get_aircraft_stats()`: Aircraft statistics
- `get_weather_stats()`: Weather patterns

### 2. Temporal Trends (📈)

**Purpose**: Time series analysis and patterns

**Features**:
- Seasonal patterns (monthly event counts)
- Decade comparison (1960s-2020s)
- Day of week analysis
- Multi-metric trend charts (zoomable)
- Weekend vs weekday comparison

**Data Sources**:
- `get_monthly_stats()`: Monthly aggregates
- `get_decade_stats()`: Decade-level trends (mv_decade_stats)
- `get_dow_stats()`: Day of week patterns
- `get_yearly_stats()`: Annual data for trend analysis

**Interactive Features**:
- Year range slider
- Multi-metric selection (accidents, fatalities, etc.)
- Comparative statistics

### 3. Geographic Analysis (🗺️)

**Purpose**: Spatial patterns and regional analysis

**Features**:
- Interactive Folium maps (3 types: markers, heatmap, clusters)
- State rankings (top 15 by events and fatalities)
- Regional analysis (5 US regions)
- Choropleth map (color-coded by event count)
- Downloadable state data table

**Map Types**:
- **Markers**: Individual events (red=fatal, blue=non-fatal)
- **Heatmap**: Density visualization weighted by fatalities
- **Clusters**: Grouped markers for better performance

**Data Sources**:
- `get_events()`: Event locations with coordinates
- `get_state_stats()`: State-level aggregates

**Interactive Features**:
- Map type selector
- Data limit selector (1K-10K events)
- Severity filter
- State search and filtering

### 4. Aircraft Safety (✈️)

**Purpose**: Aircraft type-specific risk assessment

**Features**:
- Top 20 aircraft makes (horizontal bar chart)
- Aircraft category distribution (pie chart)
- Accidents vs fatalities scatter plot
- Severity score analysis
- Complete aircraft statistics table (sortable, searchable)

**Data Sources**:
- `get_aircraft_stats()`: Aircraft data (mv_aircraft_stats)
- `get_aircraft_category_stats()`: Category aggregates

**Interactive Features**:
- Minimum accident count slider
- Make/model search
- Sort by multiple metrics
- CSV download

### 5. Cause Factors (🔍)

**Purpose**: Root cause identification and patterns

**Features**:
- Top 30 finding codes (NTSB investigation findings)
- Weather impact analysis (VMC vs IMC)
- Phase of flight treemap
- Finding code statistics
- Complete finding codes reference table (searchable)

**Data Sources**:
- `get_top_finding_codes()`: Finding codes with descriptions
- `get_weather_stats()`: Weather condition analysis
- `get_phase_stats()`: Flight phase aggregates
- `get_finding_stats()`: Finding statistics (mv_finding_stats)

**Interactive Features**:
- Finding code search
- Description search
- CSV download

## Configuration

### Database Connection

The dashboard uses environment variables for database configuration:

- `DB_USER`: Database user (default: current user)
- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: ntsb_aviation)
- `DB_PASSWORD`: Database password (optional)

### Streamlit Settings

Configuration in `.streamlit/config.toml`:

- **Theme**: Custom colors (primary: #FF6B6B, red accent)
- **Server**: Port 8501, CORS disabled
- **Caching**: Enabled with magic commands
- **Upload Size**: 200MB max

## Performance

### Caching Strategy

All database queries use `@st.cache_data(ttl=3600)` for 1-hour caching:

- Reduces database load
- Improves page load times
- Automatic cache invalidation after 1 hour

### Connection Pooling

Database connections use `psycopg2.pool.SimpleConnectionPool`:

- Min connections: 1
- Max connections: 10
- Automatic connection management

### Data Limits

For performance, large datasets are limited:

- **Maps**: Maximum 10,000 events
- **Tables**: Paginated or scrollable
- **Materialized Views**: Pre-aggregated statistics

### Expected Performance

- **Page Load**: <3 seconds (first load)
- **Query Execution**: <200ms (cached queries <50ms)
- **Map Rendering**: <2 seconds (10K markers)
- **Filter Updates**: <500ms

## Usage Tips

### Navigation

- Use sidebar to navigate between pages
- Each page is independent (no state sharing)
- Filters are page-specific

### Data Exploration

1. Start with **Overview** for high-level statistics
2. Use **Temporal Trends** to identify time-based patterns
3. Explore **Geographic Analysis** for spatial patterns
4. Investigate **Aircraft Safety** for type-specific risks
5. Examine **Cause Factors** for root cause analysis

### Interactive Features

- **Hover**: See detailed information on charts
- **Zoom**: Use mouse wheel or controls on Plotly charts
- **Pan**: Click and drag on maps and charts
- **Download**: Export data as CSV from tables
- **Search**: Filter tables by text search
- **Select**: Multi-select for metrics and filters

## Database Schema

The dashboard queries the following tables and views:

### Tables

- `events`: Master event table (179,809 rows)
- `aircraft`: Aircraft details
- `findings`: Investigation findings
- `narratives`: Accident descriptions

### Materialized Views

- `mv_yearly_stats`: Annual statistics (refreshed daily)
- `mv_state_stats`: State-level aggregates
- `mv_aircraft_stats`: Aircraft type statistics
- `mv_decade_stats`: Decade-level trends
- `mv_finding_stats`: Finding code statistics

### Refresh Materialized Views

If data appears stale:

```bash
psql -d ntsb_aviation -c "SELECT * FROM refresh_all_materialized_views();"
```

## Troubleshooting

### Database Connection Errors

**Problem**: `OperationalError: could not connect to server`

**Solution**:
```bash
# Check PostgreSQL is running
systemctl status postgresql

# Check database exists
psql -l | grep ntsb_aviation

# Test connection
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
# Ensure .venv is activated
source /home/parobek/Code/NTSB_Datasets/.venv/bin/activate

# Install dependencies
cd dashboard
pip install -r requirements.txt
```

### Map Not Displaying

**Problem**: Folium map not rendering

**Solution**:
- Check `streamlit-folium` is installed
- Verify event data has valid coordinates
- Try reducing data limit (performance issue)

### Slow Performance

**Problem**: Dashboard is slow to load

**Solutions**:
- Clear Streamlit cache: `streamlit cache clear`
- Reduce data limits in filters
- Refresh materialized views
- Check database performance

### Cache Issues

**Problem**: Data not updating

**Solution**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Or restart the app
# (Ctrl+C and re-run streamlit run app.py)
```

## Development

### Adding New Pages

1. Create new file in `pages/` with naming convention: `N_ICON_Name.py`
2. Import required utilities from `utils/` and `components/`
3. Set page config and title
4. Implement visualizations using shared components
5. Test with `streamlit run app.py`

### Adding New Queries

1. Add function to `utils/queries.py`
2. Use `@st.cache_data(ttl=3600)` decorator
3. Follow connection pool pattern (get/release)
4. Handle errors gracefully

### Adding New Charts

1. Add function to `components/charts.py`
2. Use Plotly for consistency
3. Return `go.Figure` object
4. Include hover data and labels

## Data Sources

- **NTSB**: National Transportation Safety Board
- **Coverage**: 1962-2025 (64 years)
- **Events**: 179,809 aviation accidents and incidents
- **Update Frequency**: Monthly (automated via Airflow)
- **Database**: PostgreSQL 18.0 + PostGIS

## Version

**Dashboard Version**: 1.0.0
**Database Version**: PostgreSQL 18.0
**Streamlit Version**: 1.51.0
**Last Updated**: 2025-11-08

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Check troubleshooting section above
- Review database documentation in `/docs`
- Check API documentation in `/api/README.md`
- Review CLAUDE.local.md for current project state

## Credits

- **Data Source**: NTSB Aviation Accident Database
- **Dashboard**: Built with Streamlit, Plotly, Folium
- **Database**: PostgreSQL 18.0 + PostGIS
- **Development**: Phase 2 Sprint 5 (Interactive Dashboard)
