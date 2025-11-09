# NTSB Aviation Accident API

**FastAPI REST API for 64 years of NTSB aviation accident data (1962-2025)**

Production-ready API with 15+ endpoints, OpenAPI documentation, full-text search, and geospatial queries.

## Features

- **Events API**: Query 179,809 aviation accidents with pagination and filtering
- **Statistics API**: Yearly, state, aircraft, and seasonal statistics from materialized views
- **Full-Text Search**: PostgreSQL tsvector search across accident narratives
- **Geospatial Queries**: PostGIS-powered radius, bounding box, density, and clustering
- **GeoJSON Export**: Export events as GeoJSON for mapping applications
- **OpenAPI Docs**: Auto-generated interactive documentation at `/docs`
- **Connection Pooling**: Efficient SQLAlchemy connection management
- **CORS Support**: Configurable cross-origin resource sharing
- **Docker Ready**: Production Dockerfile included

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your database credentials

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t ntsb-api .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user@host/ntsb_aviation" \
  ntsb-api
```

### Documentation

Once running, visit:
- **OpenAPI (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health/` | GET | API health check |
| `/api/v1/health/database` | GET | Database health with connection stats |

### Events Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events/` | GET | List events (paginated, filterable) |
| `/api/v1/events/{ev_id}` | GET | Get event details |
| `/api/v1/events/{ev_id}/aircraft` | GET | Get event aircraft |
| `/api/v1/events/{ev_id}/findings` | GET | Get event findings |
| `/api/v1/events/{ev_id}/narratives` | GET | Get event narratives |

**Filters**: `state`, `start_date`, `end_date`, `severity`, `ev_type`

### Statistics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/statistics/summary` | GET | Overall statistics |
| `/api/v1/statistics/yearly` | GET | Statistics by year |
| `/api/v1/statistics/states` | GET | Statistics by state |
| `/api/v1/statistics/aircraft` | GET | Statistics by aircraft type |
| `/api/v1/statistics/decades` | GET | Statistics by decade |
| `/api/v1/statistics/seasonal` | GET | Seasonal patterns |

### Search Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/` | GET | Full-text search narratives |

### Geospatial Endpoints (Sprint 4)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/geospatial/radius` | GET | Events within radius |
| `/api/v1/geospatial/bbox` | GET | Events in bounding box |
| `/api/v1/geospatial/density` | GET | Event density heatmap |
| `/api/v1/geospatial/clusters` | GET | Spatial event clusters |
| `/api/v1/geospatial/geojson` | GET | Export as GeoJSON |

## Example Requests

### List Recent Accidents in California

```bash
curl "http://localhost:8000/api/v1/events/?state=CA&page=1&page_size=10"
```

### Get Event Details

```bash
curl "http://localhost:8000/api/v1/events/20250101X00001"
```

### Search for "engine failure"

```bash
curl "http://localhost:8000/api/v1/search/?q=engine+failure&limit=50"
```

### Get Yearly Statistics

```bash
curl "http://localhost:8000/api/v1/statistics/yearly"
```

### Get Events Near Los Angeles (50km radius)

```bash
curl "http://localhost:8000/api/v1/geospatial/radius?lat=34.0522&lon=-118.2437&radius_km=50"
```

## Response Format

All list endpoints return paginated responses:

```json
{
  "total": 179809,
  "page": 1,
  "page_size": 100,
  "total_pages": 1799,
  "results": [...]
}
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://parobek@localhost/ntsb_aviation` | PostgreSQL connection URL |
| `API_HOST` | `0.0.0.0` | API host address |
| `API_PORT` | `8000` | API port |
| `API_WORKERS` | `4` | Number of workers |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## Performance

- **Connection Pooling**: 20 connections + 10 overflow
- **Response Times**: <200ms for simple queries, <500ms for spatial queries
- **Pagination**: Configurable (1-1000 items per page)
- **Database Indexes**: 59 indexes for optimal query performance
- **Materialized Views**: Pre-computed statistics for instant retrieval

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Project Structure

```
api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── database.py          # Connection pool
│   ├── routers/
│   │   ├── events.py        # Events endpoints
│   │   ├── statistics.py    # Statistics endpoints
│   │   ├── search.py        # Search endpoints
│   │   ├── geospatial.py    # Geospatial endpoints
│   │   └── health.py        # Health endpoints
│   ├── crud/
│   │   ├── events.py        # Event queries
│   │   ├── statistics.py    # Stats queries
│   │   └── geospatial.py    # Spatial queries
│   └── schemas/
│       ├── common.py        # Common schemas
│       ├── event.py         # Event schemas
│       └── statistics.py    # Stats schemas
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_health.py
│   ├── test_events.py
│   ├── test_statistics.py
│   └── test_geospatial.py
├── requirements.txt
├── Dockerfile
└── README.md (this file)
```

## Database Requirements

- PostgreSQL 18.0+
- PostGIS extension (for geospatial queries)
- pg_trgm extension (for full-text search)
- Database: `ntsb_aviation`
- Tables: 11 core tables (events, aircraft, findings, etc.)
- Materialized Views: 6 views (yearly_stats, state_stats, etc.)

## Development Notes

- **Code Quality**: All code is ruff-formatted and PEP 8 compliant
- **Type Hints**: All functions have type hints
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Proper HTTP status codes (404, 400, 500)
- **Logging**: Structured logging for debugging
- **Security**: Environment-based configuration, no hardcoded credentials

## Next Steps (Phase 2 Sprint 4-5)

- [ ] Geospatial router implementation
- [ ] GeoJSON export functionality
- [ ] Streamlit dashboard (5 pages)
- [ ] Interactive visualizations
- [ ] Map components (Folium)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, open a GitHub issue or contact the project maintainers.
