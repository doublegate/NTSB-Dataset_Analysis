# Docker Quick Start Guide

**Phase 3 Sprint 1**: Production-ready containerized NTSB Aviation Accident Database platform

---

## Prerequisites

- Docker 20.10+ installed ([get Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ installed (included with Docker Desktop)
- 8 GB RAM minimum (16 GB recommended)
- 20 GB disk space

---

## Quick Start (3 Steps)

### 1. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and set your database password:
```bash
POSTGRES_PASSWORD=your_secure_password_here  # CHANGE THIS!
```

### 2. Build and Start All Services

```bash
docker-compose build  # First time: 8-13 minutes
docker-compose up -d  # Start all services in background
```

### 3. Verify Services Running

```bash
docker-compose ps  # All services should show "healthy"
```

**Access the platform**:
- **API**: http://localhost:8000/docs (OpenAPI documentation)
- **Dashboard**: http://localhost:8501 (Interactive Streamlit UI)
- **Database**: localhost:5432 (PostgreSQL connection)

---

## Architecture

```
┌────────────────────────────────────────┐
│  Dashboard (Streamlit)                 │
│  http://localhost:8501                 │
│  - Interactive visualizations          │
│  - Connects to API internally          │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│  API (FastAPI)                         │
│  http://localhost:8000                 │
│  - 21 REST endpoints                   │
│  - OpenAPI docs at /docs               │
└────────┬───────────────┬───────────────┘
         │               │
┌────────▼──────┐ ┌──────▼────────────────┐
│  PostgreSQL   │ │  Redis                 │
│  Port 5432    │ │  Port 6379             │
│  - 179,809    │ │  - API cache           │
│    events     │ │  - 256MB max           │
│  - PostGIS    │ │  - LRU eviction        │
└───────────────┘ └────────────────────────┘
```

**4 Services**:
1. **postgres** - PostgreSQL 18 + PostGIS (primary database)
2. **redis** - Redis 7 (API caching layer)
3. **api** - FastAPI REST API (backend)
4. **dashboard** - Streamlit dashboard (frontend)

---

## Common Commands

### Start Services

```bash
docker-compose up -d              # Start in background
docker-compose up                 # Start with logs (Ctrl+C to stop)
```

### Check Status

```bash
docker-compose ps                 # Service status
docker-compose logs -f api        # Follow API logs
docker-compose logs -f dashboard  # Follow dashboard logs
```

### Stop Services

```bash
docker-compose stop               # Stop (keep data)
docker-compose down               # Stop and remove containers
docker-compose down -v            # Stop and DELETE ALL DATA ⚠️
```

### Rebuild After Code Changes

```bash
docker-compose build api          # Rebuild API only
docker-compose up -d api          # Restart API with new build
```

### Database Access

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U ntsb_user -d ntsb_aviation

# Run SQL file
docker-compose exec postgres psql -U ntsb_user -d ntsb_aviation -f /scripts/schema.sql

# Backup database
docker-compose exec postgres pg_dump -U ntsb_user ntsb_aviation > backup.sql
```

### Cleanup & Reset

```bash
# Remove all containers and volumes (fresh start)
docker-compose down -v
docker system prune -a --volumes

# Rebuild everything from scratch
docker-compose build --no-cache
docker-compose up -d
```

---

## Loading Data

**Option 1: Load from local .mdb files** (requires mdbtools)

```bash
# Copy loading script into container
docker cp scripts/load_with_staging.py ntsb-api:/app/

# Run loading script
docker-compose exec api python scripts/load_with_staging.py --source /data/avall.mdb
```

**Option 2: Restore from backup**

```bash
# Restore database from SQL backup
docker-compose exec -T postgres psql -U ntsb_user -d ntsb_aviation < backup.sql
```

**Option 3: Import schema only**

```bash
# Create schema without data
docker-compose exec postgres psql -U ntsb_user -d ntsb_aviation -f /scripts/schema.sql
```

---

## Troubleshooting

### Services Not Starting

**Check health status**:
```bash
docker-compose ps
```

**If postgres is unhealthy**:
```bash
docker-compose logs postgres
# Common issue: Invalid POSTGRES_PASSWORD in .env
```

**If api is unhealthy**:
```bash
docker-compose logs api
# Common issue: Database connection failed (check DATABASE_URL)
```

### Port Already in Use

**Error**: `Bind for 0.0.0.0:5432 failed: port is already allocated`

**Solution**: Change port in `.env`:
```bash
POSTGRES_PORT=5433  # Use different port
API_PORT=8001
DASHBOARD_PORT=8502
```

### Out of Disk Space

**Check Docker disk usage**:
```bash
docker system df
```

**Clean up**:
```bash
docker system prune -a --volumes  # Remove all unused data
```

### Slow Performance

**Increase Docker resources**:
- Docker Desktop → Settings → Resources
- Recommended: 8 GB RAM, 4 CPUs

**Check resource usage**:
```bash
docker stats  # Real-time resource usage
```

### API Returns 503 Database Error

**Check database connectivity**:
```bash
docker-compose exec api python -c "
from app.database import test_database_connection
print('Connected!' if test_database_connection() else 'Failed!')
"
```

---

## Development Workflow

### 1. Make Code Changes

Edit files in `api/`, `dashboard/`, or `database/`

### 2. Rebuild Affected Service

```bash
docker-compose build api      # If you changed API code
docker-compose build dashboard # If you changed dashboard code
```

### 3. Restart Service

```bash
docker-compose up -d api       # Restart with new build
```

### 4. View Logs

```bash
docker-compose logs -f api     # Check for errors
```

### 5. Test Changes

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## Production Deployment

See **to-dos/PHASE_3_PRODUCTION_ML.md** Sprint 6 for cloud deployment guide.

**Key Differences**:
- Use Docker secrets for credentials (not .env)
- Enable HTTPS with SSL certificates
- Use managed PostgreSQL (AWS RDS, DigitalOcean Managed Database)
- Add monitoring (Prometheus, Grafana)
- Configure auto-scaling and health checks

---

## Resource Requirements

### Minimum (Development)

- RAM: 4 GB
- CPU: 2 cores
- Disk: 10 GB
- Expected performance: Functional but slow

### Recommended (Local Testing)

- RAM: 8 GB
- CPU: 4 cores
- Disk: 20 GB
- Expected performance: Good for 1-2 users

### Production (Cloud Deployment)

- RAM: 16 GB
- CPU: 8 cores
- Disk: 50 GB SSD
- Expected performance: 100+ concurrent users

---

## Image Sizes (Estimated)

| Service | Size Target | Status |
|---------|-------------|--------|
| API | <300 MB | ✅ Multi-stage build |
| Dashboard | <400 MB | ✅ Multi-stage build |
| Database | ~200 MB | ✅ Alpine base |
| Redis | ~30 MB | ✅ Official image |
| **Total** | **~930 MB** | Optimized |

**Verify actual sizes**:
```bash
docker images | grep ntsb
```

---

## Next Steps

1. ✅ **Test locally**: `docker-compose up -d`
2. ✅ **Load data**: Use `scripts/load_with_staging.py`
3. ✅ **Explore API**: http://localhost:8000/docs
4. ✅ **Try dashboard**: http://localhost:8501
5. 📚 **Read docs**: `docs/PHASE_3_SPRINT_1_COMPLETION.md`
6. 🚀 **Deploy**: See Sprint 6 in `to-dos/PHASE_3_PRODUCTION_ML.md`

---

## Support

- **Documentation**: `docs/` directory
- **Issues**: Check `docs/PHASE_3_SPRINT_1_COMPLETION.md` troubleshooting section
- **Phase 3 Plan**: `to-dos/PHASE_3_PRODUCTION_ML.md`

---

**Created**: Phase 3 Sprint 1 (November 18, 2025)
**Status**: ✅ Production-ready containerized platform
**Next**: Sprint 2 - ML Model Enhancement (XGBoost, SHAP, >85% accuracy)
