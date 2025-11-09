# NTSB Aviation Accident Database - Interactive Dashboard

**5-page Streamlit dashboard with 25+ visualizations for exploring 64 years of aviation safety data**

## Overview

This interactive dashboard provides comprehensive visualizations and analysis tools for the NTSB Aviation Accident Database. Built with Streamlit and Plotly, it offers 5 specialized pages covering temporal trends, geographic patterns, aircraft safety, and cause factor analysis.

## Dashboard Pages

### 1. 📊 Overview
High-level statistics and key insights from 64 years of aviation safety data
- **Key Metrics**: Total events, years coverage, fatalities, fatal event rate
- **Long-term Trends**: Annual accident rates with 5-year moving average (1962-2025)
- **Geographic Distribution**: Choropleth map and state rankings
- **Quick Statistics**: Top aircraft makes and weather conditions

### 2. 📈 Temporal Trends
Time series patterns, seasonality, and trend forecasting
- **Seasonal Patterns**: Monthly accident distribution
- **Decade Comparisons**: Long-term safety improvements
- **Day of Week Analysis**: Weekly patterns
- **Trend Forecasting**: Future projections with confidence intervals

### 3. 🗺️ Geographic Analysis
Geographic distribution and regional patterns
- **State-level Statistics**: Accident counts and fatal rates
- **Regional Analysis**: Geographic clustering and hotspots
- **Interactive Maps**: Choropleth maps with drill-down capability
- **Ranked Tables**: Top states by various metrics

### 4. ✈️ Aircraft Safety
Aircraft-specific safety analysis
- **Aircraft Makes**: Top manufacturers by accident count
- **Category Analysis**: Airplane vs helicopter vs glider comparisons
- **Age Analysis**: Aircraft age correlation with fatality rates
- **Severity Breakdown**: Damage and injury classification

### 5. 🔍 Cause Factors
Investigation findings and contributing factors
- **Top Finding Codes**: Most common NTSB investigation findings
- **Weather Impact**: VMC vs IMC conditions and fatal rates
- **Phase of Flight**: Takeoff, cruise, approach, landing analysis
- **Finding Statistics**: Detailed breakdowns with severity metrics

## Database Connection

This dashboard uses **SQLAlchemy** for database connections instead of raw psycopg2.

### Connection Pooling
- **Pool size**: 10 connections
- **Max overflow**: 5 additional connections
- **Connection pre-ping**: Enabled (auto-reconnect on stale connections)
- **Caching**: SQLAlchemy engine cached with `@st.cache_resource`

### Migration from psycopg2 (2025-11-09)

**Previous Implementation** (psycopg2.pool):
```python
import psycopg2
from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(1, 10,
    dbname='ntsb_aviation', user='parobek')
```

**Current Implementation** (SQLAlchemy):
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

@st.cache_resource
def get_engine():
    return create_engine(
        'postgresql://parobek@localhost/ntsb_aviation',
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=5,
        pool_pre_ping=True
    )
```

**Benefits**:
- ✅ **No pandas warnings**: Eliminates UserWarning about DBAPI2 connections
- ✅ **Better connection pooling**: SQLAlchemy's QueuePool is more robust
- ✅ **Auto-reconnect**: `pool_pre_ping=True` handles stale connections
- ✅ **Caching**: `@st.cache_resource` ensures single engine instance
- ✅ **Compatibility**: Works seamlessly with `pd.read_sql()`

## Installation & Setup

### Prerequisites
- Python 3.13+ with virtual environment
- PostgreSQL 18.0+ with ntsb_aviation database
- NTSB data loaded (see main repository README.md)

### Install Dependencies

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Install dashboard dependencies (already included in main requirements.txt)
pip install streamlit plotly pandas sqlalchemy

# Verify SQLAlchemy version
pip list | grep -i sqlalchemy
# Expected: SQLAlchemy==2.0.44
```

### Configure Database Connection

By default, the dashboard connects to:
- **Host**: localhost
- **Port**: 5432
- **Database**: ntsb_aviation
- **User**: Current system user (from $USER)

To override defaults, set environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=ntsb_aviation
export DB_USER=parobek
export DB_PASSWORD=""  # Optional, leave empty for peer authentication
```

### Run Dashboard

```bash
# From dashboard directory
cd dashboard

# Activate virtual environment
source ../.venv/bin/activate

# Run Streamlit
streamlit run app.py

# Access dashboard at http://localhost:8501
```

## Dashboard Features

### Visualizations (25+ Total)
- **Line Charts**: Time series with moving averages
- **Bar Charts**: Categorical comparisons with color-coded severity
- **Choropleth Maps**: Geographic distributions with state-level detail
- **Scatter Plots**: Correlation analysis (aircraft age vs fatality rates)
- **Pie Charts**: Proportional breakdowns
- **Treemaps**: Hierarchical data visualization
- **Heatmaps**: Correlation matrices and 2D distributions

### Interactive Features
- **Year Range Filters**: Filter data by time period (sidebar)
- **Top N Selectors**: Adjust number of results displayed
- **Hover Details**: Rich tooltips on all visualizations
- **Drill-down Tables**: Detailed data tables with sorting
- **Export**: Download data as CSV from tables
- **Responsive Layout**: Wide layout with 2-column grids

### Performance
- **Query Speed**: <500ms for all database queries
- **Page Load**: <2 seconds for initial page load
- **Visualization Rendering**: <1 second per chart
- **Connection Pooling**: 10 concurrent connections, auto-scaling to 15

## Code Quality

### Recent Improvements (2025-11-09)

**Fixed Warnings**:
- ✅ **SQLAlchemy Migration**: Eliminated pandas UserWarning (12+ instances)
- ✅ **Streamlit Deprecations**: Replaced `use_container_width` → `width` (32 instances)
- ✅ **Zero Warnings**: Clean console output for production deployment

**Code Formatting**:
```bash
# Format all dashboard code
ruff format dashboard/

# Check for issues
ruff check dashboard/

# Known non-critical warnings:
# - E402: Module-level imports after sys.path (required for Streamlit multi-page)
```

### Code Structure

```
dashboard/
├── app.py                          # Main dashboard entry point
├── pages/                          # Multi-page app pages
│   ├── 1_📊_Overview.py            # Overview dashboard
│   ├── 2_📈_Temporal_Trends.py     # Time series analysis
│   ├── 3_🗺️_Geographic_Analysis.py # Geographic patterns
│   ├── 4_✈️_Aircraft_Safety.py     # Aircraft analysis
│   └── 5_🔍_Cause_Factors.py       # Investigation findings
├── components/                     # Reusable UI components
│   ├── charts.py                   # Plotly chart templates
│   ├── filters.py                  # Filter widgets
│   └── maps.py                     # Map visualizations
├── utils/                          # Utility functions
│   ├── database.py                 # SQLAlchemy connection pooling
│   └── queries.py                  # Database queries (12 functions)
└── README.md                       # This file
```

## Database Queries

The dashboard uses **12 optimized SQL queries** (defined in `utils/queries.py`):

1. **get_summary_stats()**: Overall database statistics
2. **get_yearly_stats()**: Annual accident trends
3. **get_monthly_stats()**: Seasonal patterns
4. **get_dow_stats()**: Day of week analysis
5. **get_decade_stats()**: Decade comparisons
6. **get_state_stats()**: State-level statistics
7. **get_aircraft_stats()**: Aircraft make/model analysis
8. **get_weather_stats()**: Weather condition breakdowns
9. **get_finding_stats()**: Investigation findings
10. **get_phase_stats()**: Phase of flight analysis
11. **get_aircraft_age_stats()**: Age correlation analysis
12. **get_severity_stats()**: Damage/injury severity

All queries:
- Use SQLAlchemy engine (no pandas warnings)
- Leverage materialized views for aggregations
- Return pandas DataFrames for visualization
- Execute in <500ms (p95 latency)

## Troubleshooting

### Dashboard Won't Start

**Error**: `ModuleNotFoundError: No module named 'sqlalchemy'`
```bash
source ../.venv/bin/activate
pip install sqlalchemy
```

**Error**: `sqlalchemy.exc.OperationalError: could not connect to server`
```bash
# Verify PostgreSQL is running
systemctl status postgresql

# Check database exists
psql -l | grep ntsb_aviation

# Test connection manually
psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"
```

### No Data Displayed

**Issue**: Dashboard loads but shows no data

**Solution**:
1. Verify database is loaded: `psql -d ntsb_aviation -c "SELECT COUNT(*) FROM events;"`
2. Check connection settings in `utils/database.py`
3. Review Streamlit console output for SQL errors

### Slow Performance

**Issue**: Dashboard pages load slowly

**Solutions**:
- Refresh materialized views: `SELECT * FROM refresh_all_materialized_views();`
- Check database size: `SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));`
- Monitor connection pool: Increase `pool_size` in `database.py` if needed
- Run VACUUM ANALYZE: `psql -d ntsb_aviation -c "VACUUM ANALYZE;"`

### Visualization Rendering Issues

**Issue**: Charts don't display correctly

**Solutions**:
- Clear Streamlit cache: `streamlit cache clear`
- Update Plotly: `pip install --upgrade plotly`
- Check browser console for JavaScript errors
- Try different browser (Chrome/Firefox recommended)

## Development

### Adding New Pages

1. Create new file in `pages/` directory:
   ```python
   # pages/6_🔬_New_Analysis.py
   import streamlit as st
   import sys
   from pathlib import Path

   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   from dashboard.utils.queries import get_summary_stats

   st.set_page_config(page_title="New Analysis", page_icon="🔬", layout="wide")
   st.title("🔬 New Analysis")
   ```

2. Add query function to `utils/queries.py`
3. Add chart templates to `components/charts.py` if needed
4. Test page: `streamlit run app.py` and navigate to new page

### Adding New Queries

```python
# utils/queries.py
def get_new_stats() -> pd.DataFrame:
    """Get new statistics from database.

    Returns:
        DataFrame with new_column_1, new_column_2, etc.
    """
    conn = get_connection()  # Returns SQLAlchemy engine
    try:
        query = """
            SELECT column_1, column_2
            FROM your_table
            WHERE conditions
        """
        df = pd.read_sql(query, conn)  # No warnings with SQLAlchemy
        return df
    finally:
        release_connection(conn)  # No-op for SQLAlchemy
```

## Production Deployment

### Deployment Checklist
- ✅ All warnings eliminated (SQLAlchemy + Streamlit deprecations)
- ✅ Database connection pooling configured
- ✅ Environment variables for database credentials
- ✅ Code formatted with ruff
- ✅ Materialized views refreshed
- ✅ Database health score: 98/100

### Performance Optimization
- Use `@st.cache_data` for expensive computations
- Leverage materialized views for aggregations
- Enable Streamlit server-side session state caching
- Configure connection pool size based on concurrent users

### Security
- Use environment variables for database credentials
- Enable PostgreSQL SSL connections for production
- Implement authentication with Streamlit auth (if deploying publicly)
- Use read-only database user for dashboard queries

## Technical Specifications

- **Framework**: Streamlit 1.51+
- **Database**: PostgreSQL 18.0+ with SQLAlchemy 2.0.44
- **Visualization**: Plotly 5.18+
- **Data Processing**: pandas 2.1+
- **Python**: 3.13+
- **Connection Pooling**: SQLAlchemy QueuePool (10 base + 5 overflow)

## Support

For issues or questions:
1. Check this README for troubleshooting steps
2. Review main repository documentation
3. Check console output for error messages
4. Verify database connection and data loading
5. Test queries manually with psql

## Changelog

### 2025-11-09 - Production Readiness
- **Fixed**: Migrated from psycopg2 to SQLAlchemy (eliminates pandas warnings)
- **Fixed**: Replaced deprecated `use_container_width` with `width` parameter (32 instances)
- **Improved**: Zero warnings in console output
- **Improved**: Production-ready code quality

### 2025-11-08 - Initial Release
- **Added**: 5-page interactive dashboard
- **Added**: 25+ visualizations across temporal, geographic, aircraft, and cause factor analysis
- **Added**: 12 optimized database queries with materialized view support
- **Added**: Connection pooling with psycopg2

## License

This dashboard is part of the NTSB Aviation Accident Database project.
See main repository LICENSE file for details.
