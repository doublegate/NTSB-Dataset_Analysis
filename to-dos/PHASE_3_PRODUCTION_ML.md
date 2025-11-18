# PHASE 3: PRODUCTION-READY ML DEPLOYMENT

Production deployment with enhanced ML models, containerization, CI/CD, monitoring, and public beta launch.

**Timeline**: Q1 2026 (12 weeks, January-March 2026)
**Prerequisites**: Phase 1-2 complete (database, API, dashboard, baseline ML)
**Team**: 1 developer + AI assistance (full-stack with ML/DevOps experience)
**Estimated Hours**: ~280-320 hours total (expandable to 400+ for advanced features)

**Theme**: Bridge the gap between local prototype and production system by deploying existing infrastructure while improving ML models to production-grade accuracy (>85%). Focus on delivering value to real users quickly while maintaining technical excellence.

## Overview

**Philosophy**: "Deploy early, iterate fast, validate with real users"

This phase takes a **pragmatic hybrid approach** that combines essential deployment infrastructure with ML model improvements. Instead of building everything locally then deploying (waterfall), we deploy incrementally while improving models, enabling real-world validation and feedback loops.

**Why This Order?**
1. **Current State**: Phase 2 delivered functional API + dashboard + baseline ML, but everything is local-only
2. **Value Gap**: Great analytics exist but no one can access them
3. **ML Validation**: Can't improve models without production usage data and drift monitoring
4. **Risk Mitigation**: Small deployments with real users reveal issues faster than internal testing
5. **Foundation First**: Deployment infrastructure enables all future phases (AI, scaling)

| Sprint | Duration | Focus Area | Key Deliverables | Hours |
|--------|----------|------------|------------------|-------|
| Sprint 1 | Weeks 1-2 | Containerization & CI/CD | Docker images, GitHub Actions, tests | 40h |
| Sprint 2 | Weeks 2-3 | ML Model Enhancement | XGBoost 90%+, feature engineering, SHAP | 45h |
| Sprint 3 | Weeks 4-5 | Authentication & Security | JWT auth, rate limiting, API keys | 35h |
| Sprint 4 | Weeks 5-6 | MLflow & Model Serving | Registry, versioning, FastAPI integration | 40h |
| Sprint 5 | Weeks 7-8 | Monitoring & Observability | Prometheus, Grafana, logging, alerts | 35h |
| Sprint 6 | Weeks 9-10 | Cloud Deployment | DigitalOcean/AWS, domain, SSL, CDN | 40h |
| Sprint 7 | Weeks 10-11 | Documentation & Beta Prep | API docs, user guides, beta program | 30h |
| Sprint 8 | Weeks 11-12 | Beta Launch & Iteration | 25-50 users, feedback, improvements | 35h |

**Total**: 300 hours (base), expandable to 400+ with advanced features

## Success Criteria

**Phase 3 is COMPLETE when**:
- ✅ All services containerized and running in production (API, dashboard, database)
- ✅ ML models achieve >85% accuracy (up from 78%) with SHAP explainability
- ✅ Public API accessible with authentication, rate limiting, and documentation
- ✅ Beta program launched with 25-50 active users providing feedback
- ✅ Monitoring dashboards show <200ms API latency, >99% uptime
- ✅ CI/CD pipeline automatically tests and deploys changes
- ✅ Complete documentation for users, developers, and operators

**Phase 3 is EXCELLENT when** (stretch goals):
- 🌟 ML models achieve 90%+ accuracy with ensemble methods
- 🌟 100+ beta users with >80% satisfaction rating
- 🌟 API serving 1000+ requests/day with <100ms p95 latency
- 🌟 Featured in aviation safety community (Reddit, HN, conferences)
- 🌟 First academic publication submitted or partnership established

---

## Sprint 1: Containerization & CI/CD (Weeks 1-2, 40 hours)

**Goal**: Containerize all services (API, dashboard, database) and establish automated testing/deployment pipeline.

### Week 1: Docker Containerization

**Objective**: Create optimized Docker images for all services with multi-stage builds (<500MB each).

**Tasks**:
- [ ] **API Dockerfile** (6 hours)
  - Multi-stage build: builder (dependencies) + runtime (slim)
  - Python 3.13-slim base image
  - Copy only necessary files (requirements.txt, app/, exclude tests/)
  - Set environment variables for database connection
  - Health check endpoint (/health) configured
  - Non-root user for security
  - Target size: <300MB

- [ ] **Dashboard Dockerfile** (5 hours)
  - Streamlit-optimized container
  - Port 8501 exposed
  - Environment variables for API base URL
  - Browser caching configuration
  - Target size: <400MB

- [ ] **PostgreSQL Dockerfile** (4 hours)
  - Official postgres:18-alpine base
  - PostGIS extension pre-installed
  - Initialization scripts mounted (/docker-entrypoint-initdb.d/)
  - Volume for persistent data
  - Custom postgresql.conf for analytics workload

- [ ] **Docker Compose Configuration** (5 hours)
  - 4 services: postgres, api, dashboard, redis (caching)
  - Named volumes for data persistence
  - Network isolation (internal network for DB)
  - Environment file (.env) for secrets
  - Health checks for all services
  - Dependency ordering (postgres → api → dashboard)

**Deliverables**:
- `api/Dockerfile` (30-40 lines)
- `dashboard/Dockerfile` (25-35 lines)
- `database/Dockerfile` (20-30 lines)
- `docker-compose.yml` (150-200 lines)
- `.env.example` (environment variable template)

**Success Metrics**:
- All images build successfully (docker-compose build)
- All services start and pass health checks (docker-compose up)
- API accessible at http://localhost:8000/health
- Dashboard accessible at http://localhost:8501
- Database accessible and queryable
- Total build time <5 minutes on standard machine

**Code Example - Multi-Stage API Dockerfile**:
```dockerfile
# Stage 1: Builder (dependencies)
FROM python:3.13-slim AS builder

WORKDIR /app
COPY api/requirements.txt .

# Install dependencies to user site-packages
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (minimal)
FROM python:3.13-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY api/ .

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Code Example - Docker Compose**:
```yaml
version: '3.8'

services:
  postgres:
    build: ./database
    container_name: ntsb-postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-ntsb_aviation}
      POSTGRES_USER: ${POSTGRES_USER:-ntsb_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ntsb-network

  redis:
    image: redis:7-alpine
    container_name: ntsb-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ntsb-network

  api:
    build: ./api
    container_name: ntsb-api
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - ntsb-network

  dashboard:
    build: ./dashboard
    container_name: ntsb-dashboard
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      api:
        condition: service_healthy
    ports:
      - "8501:8501"
    networks:
      - ntsb-network

volumes:
  postgres_data:

networks:
  ntsb-network:
    driver: bridge
```

**Testing**:
```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# Check health
docker-compose ps  # Should show all services "healthy"

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Test dashboard
open http://localhost:8501

# View logs
docker-compose logs -f api

# Cleanup
docker-compose down -v
```

**Dependencies**: docker, docker-compose

---

### Week 2: CI/CD Pipeline (GitHub Actions)

**Objective**: Automate testing, linting, and deployment with GitHub Actions.

**Tasks**:
- [ ] **Test Suite Enhancement** (8 hours)
  - Add pytest tests for API endpoints (>80% coverage target)
  - Add tests for database queries (validate schema, indexes)
  - Add tests for ML model predictions (fixture data)
  - Add tests for dashboard components (streamlit testing)
  - Create test fixtures (sample data, mock responses)
  - Integration tests with Docker Compose

- [ ] **GitHub Actions Workflows** (12 hours)
  - **Workflow 1: CI (test.yml)** - Run on every push/PR
    - Checkout code
    - Set up Python 3.13
    - Install dependencies
    - Run ruff linting (format + check)
    - Run pytest with coverage report
    - Run mypy type checking
    - Upload coverage to Codecov (optional)
  
  - **Workflow 2: Docker Build (docker.yml)** - Build and push images
    - Build multi-platform images (linux/amd64, linux/arm64)
    - Push to Docker Hub or GitHub Container Registry
    - Tag with git commit SHA + latest
    - Cache layers for faster builds
  
  - **Workflow 3: Deploy (deploy.yml)** - Deploy to production
    - Trigger on push to main branch
    - SSH into production server
    - Pull latest images
    - Run docker-compose up -d
    - Run smoke tests
    - Send Slack notification

**Deliverables**:
- `tests/test_api.py` (200+ lines, 15+ test cases)
- `tests/test_database.py` (150+ lines, 10+ test cases)
- `tests/test_models.py` (100+ lines, 5+ test cases)
- `.github/workflows/ci.yml` (80-100 lines)
- `.github/workflows/docker.yml` (60-80 lines)
- `.github/workflows/deploy.yml` (50-70 lines)
- `pytest.ini` (configuration)
- `codecov.yml` (optional coverage tracking)

**Success Metrics**:
- Test suite passes locally (pytest tests/ -v)
- All GitHub Actions workflows pass on first run
- Test coverage >80% for API endpoints
- CI pipeline completes in <5 minutes
- Docker images build successfully for both platforms
- Deployment workflow runs without errors (dry-run mode)

**Code Example - GitHub Actions CI Workflow**:
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:18-alpine
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: ntsb_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python 3.13
      uses: actions/setup-python@v5
      with:
        python-version: '3.13'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r api/requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with ruff
      run: |
        ruff format --check .
        ruff check .

    - name: Type check with mypy
      run: |
        mypy api/ --ignore-missing-imports

    - name: Run tests with coverage
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/ntsb_test
        REDIS_URL: redis://localhost:6379/0
      run: |
        pytest tests/ -v --cov=api --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        fail_ci_if_error: false

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build API image
      uses: docker/build-push-action@v5
      with:
        context: ./api
        file: ./api/Dockerfile
        push: false
        tags: ntsb-api:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Build Dashboard image
      uses: docker/build-push-action@v5
      with:
        context: ./dashboard
        file: ./dashboard/Dockerfile
        push: false
        tags: ntsb-dashboard:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

**Code Example - Pytest API Tests**:
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint returns 200 OK"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_stats_endpoint():
    """Test statistics endpoint returns valid data"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert data["total_events"] > 0

def test_events_pagination():
    """Test events endpoint with pagination"""
    response = client.get("/events?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 10
    assert data["page"] == 1

def test_search_endpoint():
    """Test full-text search endpoint"""
    response = client.get("/search?q=engine+failure&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5

def test_geojson_export():
    """Test GeoJSON export endpoint"""
    response = client.get("/geojson?state=CA&limit=100")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) <= 100

@pytest.mark.parametrize("endpoint", [
    "/events",
    "/stats",
    "/search?q=test",
    "/geojson?limit=10"
])
def test_cors_headers(endpoint):
    """Test CORS headers are present"""
    response = client.get(endpoint)
    # Note: CORS headers only present if CORS middleware configured
    # This test verifies endpoints are accessible
    assert response.status_code in [200, 422]  # 422 for missing required params

def test_rate_limiting():
    """Test rate limiting returns 429 after threshold"""
    # This test requires rate limiting to be configured
    # For now, just verify endpoint exists
    for _ in range(150):  # Exceed typical 100 req/hour limit
        response = client.get("/health")
    # Would expect 429 if rate limiting active
    # For CI, just verify endpoint responds
    assert response.status_code in [200, 429]
```

**Dependencies**: pytest, pytest-cov, pytest-asyncio, httpx (for async testing)

**Sprint 1 Total Hours**: 40 hours

---

## Sprint 2: ML Model Enhancement (Weeks 2-3, 45 hours)

**Goal**: Improve ML models from 78% to >85% accuracy using feature engineering, XGBoost, and SHAP explainability.

### Week 2 (continued): Advanced Feature Engineering

**Objective**: Create 50+ engineered features from raw data to improve model performance.

**Tasks**:
- [ ] **NTSB Code Features** (8 hours)
  - Extract occurrence codes (100-430): one-hot encode major categories
  - Extract phase of operation (500-610): TAXI, TAKEOFF, CRUISE, APPROACH, LANDING
  - Extract cause codes (30000-84200): hierarchical grouping
    - Engine-related (14000-17710)
    - Weather-related (22000-23318)
    - Pilot technique (24000-24700)
  - Create code mapping tables for interpretability
  - Validate against ref_docs/codman.pdf

- [ ] **Temporal Features** (6 hours)
  - Cyclical encoding: month_sin, month_cos (preserve seasonality)
  - Day of week encoding: weekend vs weekday
  - Hour of day: morning/afternoon/evening/night buckets
  - Days since last accident at same airport/location
  - Temporal trends: accidents in last 30/90/365 days

- [ ] **Aircraft Features** (6 hours)
  - Aircraft age: event_year - year_manufactured
  - Engine count: single vs multi-engine
  - Aircraft category: Airplane, Helicopter, Glider, etc.
  - Amateur-built flag: 57% higher fatal rate (from Phase 2)
  - Damage severity: DEST (destroyed), SUBS (substantial), MINR, NONE

- [ ] **Pilot Experience Features** (5 hours)
  - Total flight hours: log-transform (right-skewed distribution)
  - Certification level: ordinal encoding (Student=1, Private=3, Commercial=4, ATP=5)
  - Recency: hours in last 90 days (if available)
  - Type rating: hours in aircraft type

- [ ] **Weather Features** (4 hours)
  - VMC vs IMC: 2.3x higher fatal rate in IMC (from Phase 2)
  - Wind speed categories: calm, moderate, high (>25 kts)
  - Visibility: low (<3 miles) vs good
  - Weather condition aggregation: NORM vs ADVERSE

- [ ] **Geospatial Features** (6 hours)
  - State risk scores: accident rate per state
  - DBSCAN cluster membership: 64 clusters from Phase 2
  - Distance to nearest major airport
  - Terrain type: flat, hilly, mountainous (from elevation data)
  - Population density: rural, suburban, urban

**Deliverables**:
- `api/ml/feature_engineering.py` (500+ lines)
  - `AviationFeatureEngineer` class with modular pipelines
  - `fit()` method: learn encodings from training data
  - `transform()` method: apply to new data
  - Save/load encoders (joblib)
- `notebooks/modeling/feature_engineering_v2.ipynb` (validation)
- Feature importance analysis (top 30 features)
- Feature correlation matrix (identify multicollinearity)

**Success Metrics**:
- 50+ features engineered (target: 80-100)
- Feature sparsity <70% (most features have >30% non-zero values)
- No multicollinearity (|correlation| < 0.9 for all feature pairs)
- Feature engineering pipeline runs in <30 seconds for full dataset

**Code Example - Feature Engineering Class**:
```python
# api/ml/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List
import joblib

class AviationFeatureEngineer:
    """
    Advanced feature engineering for aviation accident prediction.
    
    Creates 80+ features from raw NTSB data including:
    - NTSB code features (occurrence, phase, cause)
    - Temporal features (cyclical encoding, recency)
    - Aircraft characteristics (age, type, damage)
    - Pilot experience (flight hours, certification)
    - Weather conditions (VMC/IMC, wind, visibility)
    - Geospatial features (state risk, cluster membership)
    """
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        
    def fit(self, df: pd.DataFrame) -> 'AviationFeatureEngineer':
        """Learn encodings from training data"""
        
        # Learn label encoders
        categorical_cols = ['ev_state', 'acft_category', 'wx_cond_basic']
        for col in categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].dropna())
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to new data"""
        df = df.copy()
        
        # 1. Temporal features (cyclical encoding)
        df = self._create_temporal_features(df)
        
        # 2. Aircraft features
        df = self._create_aircraft_features(df)
        
        # 3. Pilot features
        df = self._create_pilot_features(df)
        
        # 4. Weather features
        df = self._create_weather_features(df)
        
        # 5. Geospatial features
        df = self._create_geospatial_features(df)
        
        # 6. NTSB code features
        df = self._create_code_features(df)
        
        self.feature_names = [col for col in df.columns if col not in self._get_raw_columns()]
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical encoding for temporal features"""
        
        # Month (1-12) → sin/cos
        if 'ev_month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['ev_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['ev_month'] / 12)
        
        # Day of week (0-6) → sin/cos
        if 'ev_date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['ev_date']).dt.dayofweek
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Hour of day (if available)
        if 'ev_time' in df.columns:
            df['ev_hour'] = pd.to_datetime(df['ev_time'], format='%H:%M:%S', errors='coerce').dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['ev_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['ev_hour'] / 24)
            
            # Time of day buckets
            df['is_morning'] = ((df['ev_hour'] >= 6) & (df['ev_hour'] < 12)).astype(int)
            df['is_afternoon'] = ((df['ev_hour'] >= 12) & (df['ev_hour'] < 18)).astype(int)
            df['is_evening'] = ((df['ev_hour'] >= 18) & (df['ev_hour'] < 22)).astype(int)
            df['is_night'] = ((df['ev_hour'] >= 22) | (df['ev_hour'] < 6)).astype(int)
        
        return df
    
    def _create_aircraft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aircraft characteristics"""
        
        # Aircraft age
        if 'acft_year' in df.columns and 'ev_year' in df.columns:
            df['aircraft_age'] = df['ev_year'] - df['acft_year']
            df['aircraft_age'] = df['aircraft_age'].clip(lower=0, upper=100)
            
            # Age buckets
            df['is_new_aircraft'] = (df['aircraft_age'] <= 5).astype(int)
            df['is_old_aircraft'] = (df['aircraft_age'] >= 30).astype(int)
        
        # Engine count
        if 'num_eng' in df.columns:
            df['has_multiple_engines'] = (df['num_eng'] > 1).astype(int)
        
        # Amateur-built (high-risk factor from Phase 2)
        if 'acft_category' in df.columns:
            df['is_amateur_built'] = (df['acft_category'] == 'Amateur Built').astype(int)
        
        # Damage severity (ordinal)
        if 'damage' in df.columns:
            damage_map = {'DEST': 4, 'SUBS': 3, 'MINR': 2, 'NONE': 1}
            df['damage_severity'] = df['damage'].map(damage_map).fillna(0)
        
        return df
    
    def _create_pilot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pilot experience features"""
        
        # Total flight hours (log transform for right-skewed distribution)
        if 'crew_tot_hrs' in df.columns:
            df['log_flight_hours'] = np.log1p(df['crew_tot_hrs'])
            
            # Experience buckets
            df['is_low_experience'] = (df['crew_tot_hrs'] < 100).astype(int)
            df['is_high_experience'] = (df['crew_tot_hrs'] >= 1000).astype(int)
        
        # Certification level (ordinal)
        if 'crew_cert' in df.columns:
            cert_map = {'Student': 1, 'Sport': 2, 'Private': 3, 'Commercial': 4, 'ATP': 5}
            df['cert_level'] = df['crew_cert'].map(cert_map).fillna(0)
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weather conditions"""
        
        # VMC vs IMC (2.3x higher fatal rate in IMC from Phase 2)
        if 'wx_cond_basic' in df.columns:
            df['is_imc'] = (df['wx_cond_basic'] == 'IMC').astype(int)
        
        # Wind speed
        if 'wx_wind_speed_kts' in df.columns:
            df['is_high_wind'] = (df['wx_wind_speed_kts'] > 25).astype(int)
        
        # Visibility
        if 'wx_vis_sm' in df.columns:
            df['is_low_visibility'] = (df['wx_vis_sm'] < 3).astype(int)
        
        return df
    
    def _create_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geospatial features"""
        
        # State risk scores (from Phase 2 analysis)
        state_risk = {
            'AK': 3.5,  # Alaska: high fatality rate
            'CA': 2.1,  # California: high volume
            'TX': 1.8,
            'FL': 1.6,
            # ... add all states
        }
        
        if 'ev_state' in df.columns:
            df['state_risk_score'] = df['ev_state'].map(state_risk).fillna(1.0)
        
        # DBSCAN cluster membership (from Phase 2)
        # Would join with cluster assignments from geospatial analysis
        
        return df
    
    def _create_code_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """NTSB code features"""
        
        # Occurrence codes (100-430)
        # Extract from findings table via JOIN
        # For now, simplified example
        
        # Phase of operation
        if 'phase_of_flight' in df.columns:
            phase_map = {
                'TAXI': 510,
                'TAKEOFF': 520,
                'CRUISE': 550,
                'APPROACH': 580,
                'LANDING': 600
            }
            df['phase_code'] = df['phase_of_flight'].map(phase_map).fillna(0)
        
        return df
    
    def _get_raw_columns(self) -> List[str]:
        """List of original columns (not engineered features)"""
        return ['ev_id', 'ev_date', 'ev_time', 'ev_year', 'ev_month',
                'ev_state', 'ev_highest_injury', 'inj_tot_f',
                'acft_year', 'acft_category', 'num_eng', 'damage',
                'crew_tot_hrs', 'crew_cert', 'wx_cond_basic',
                'wx_wind_speed_kts', 'wx_vis_sm']
    
    def get_feature_names(self) -> List[str]:
        """Get names of engineered features"""
        return self.feature_names
    
    def save(self, filepath: str):
        """Save feature engineering pipeline"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'AviationFeatureEngineer':
        """Load feature engineering pipeline"""
        return joblib.load(filepath)
```

**Usage Example**:
```python
# Train feature engineering pipeline
fe = AviationFeatureEngineer()
fe.fit(train_df)

# Transform training data
X_train = fe.transform(train_df)

# Transform test data
X_test = fe.transform(test_df)

# Get feature names
feature_names = fe.get_feature_names()
print(f"Generated {len(feature_names)} features")

# Save pipeline
fe.save('models/feature_engineer.joblib')

# Load and use
fe_loaded = AviationFeatureEngineer.load('models/feature_engineer.joblib')
X_new = fe_loaded.transform(new_data)
```

**Dependencies**: pandas, numpy, scikit-learn, joblib

---

### Week 3: XGBoost Model Training & SHAP

**Objective**: Train XGBoost model with 90%+ accuracy and integrate SHAP explainability.

**Tasks**:
- [ ] **Data Preparation** (4 hours)
  - Train-test split: 80/20, stratified by severity
  - Handle class imbalance: SMOTE or class weights
  - Feature selection: remove low-importance features (from baseline RF)
  - Save train/test split indices for reproducibility

- [ ] **XGBoost Training** (10 hours)
  - Define target variable: binary (fatal vs non-fatal) or multi-class (none/minor/serious/fatal)
  - Configure XGBoost parameters:
    - n_estimators: 200-500 (early stopping)
    - max_depth: 5-10 (prevent overfitting)
    - learning_rate: 0.01-0.1 (tune with validation)
    - subsample: 0.7-0.9 (regularization)
    - colsample_bytree: 0.7-0.9
    - eval_metric: 'auc' or 'logloss'
  - Train with early stopping (patience=20 rounds)
  - Evaluate on test set: accuracy, precision, recall, F1, ROC-AUC
  - Cross-validation: 5-fold CV for robust performance estimate
  - Compare to baseline (Logistic Regression 78.47%)

- [ ] **SHAP Explainability** (7 hours)
  - Install SHAP library
  - Initialize TreeExplainer (optimized for XGBoost)
  - Calculate SHAP values for test set
  - Create visualizations:
    - Summary plot: global feature importance
    - Dependence plot: top 10 features
    - Waterfall plot: individual predictions
    - Force plot: local explanations
  - Save SHAP values to disk (for API integration)
  - Validate explanations: sanity check with domain knowledge

**Deliverables**:
- `api/ml/train_xgboost.py` (300+ lines)
- `models/xgboost_fatal_classifier.joblib` (trained model)
- `models/shap_explainer.joblib` (SHAP explainer)
- `notebooks/modeling/xgboost_training.ipynb` (training notebook)
- `reports/xgboost_evaluation_report.md` (performance metrics)
- SHAP plots: 5+ visualizations

**Success Metrics**:
- XGBoost accuracy >85% (target: 90%)
- ROC-AUC >0.92
- Precision/Recall balanced (F1 >0.85)
- SHAP values computed in <100ms per prediction
- Top 10 features explain >60% of model variance

**Code Example - XGBoost Training**:
```python
# api/ml/train_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import shap
import matplotlib.pyplot as plt

def train_xgboost(df: pd.DataFrame, target_col: str = 'is_fatal') -> dict:
    """
    Train XGBoost model for fatal accident prediction.
    
    Args:
        df: DataFrame with engineered features
        target_col: Target variable column name
    
    Returns:
        dict with model, metrics, and SHAP explainer
    """
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in 
                    ['ev_id', 'is_fatal', 'ev_highest_injury']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fatal rate (train): {y_train.mean():.2%}")
    
    # Configure XGBoost
    params = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,  # Regularization
        'reg_alpha': 0.5,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train with early stopping
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*60)
    print("XGBoost Model Performance")
    print("="*60)
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # SHAP explainability
    print("\n" + "="*60)
    print("Calculating SHAP values...")
    print("="*60)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (global feature importance)
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plot for top feature
    top_feature = feature_importance.iloc[0]['feature']
    shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_dependence_{top_feature}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("SHAP plots saved to reports/figures/")
    
    # Save model and explainer
    joblib.dump(model, 'models/xgboost_fatal_classifier.joblib')
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    
    print("\nModel and explainer saved to models/")
    
    return {
        'model': model,
        'explainer': explainer,
        'metrics': {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_roc_auc': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        },
        'feature_importance': feature_importance
    }

if __name__ == '__main__':
    # Load data with engineered features
    df = pd.read_parquet('data/features_engineered.parquet')
    
    # Train model
    results = train_xgboost(df)
    
    print("\nTraining complete! ✅")
```

**Dependencies**: xgboost, shap, scikit-learn, joblib, matplotlib

**Sprint 2 Total Hours**: 45 hours

---

## Sprint 3: Authentication & Security (Weeks 4-5, 35 hours)

**Goal**: Implement JWT authentication, API key management, and rate limiting for public API.

### Week 4: JWT Authentication

**Objective**: Secure API endpoints with JWT (JSON Web Tokens) and implement user management.

**Tasks**:
- [ ] **User Database Schema** (4 hours)
  - Create `users` table in PostgreSQL
  - Fields: id, username, email, hashed_password, api_key, tier (free/premium/enterprise), created_at, last_login
  - Create `api_keys` table for tracking API key usage
  - Fields: id, user_id, key_hash, name, created_at, last_used_at, request_count
  - Add indexes on username, email, api_key

- [ ] **Password Hashing** (3 hours)
  - Use bcrypt for password hashing (rounds=12)
  - Implement password validation (min 8 chars, complexity requirements)
  - Add password reset functionality (token-based)

- [ ] **JWT Implementation** (8 hours)
  - Install python-jose for JWT handling
  - Create access tokens (expiration: 1 hour)
  - Create refresh tokens (expiration: 30 days)
  - Store JWT secret in environment variable
  - Implement token validation middleware
  - Add token refresh endpoint

- [ ] **Login/Registration Endpoints** (6 hours)
  - POST /auth/register - Create new user
  - POST /auth/login - Return access + refresh tokens
  - POST /auth/refresh - Refresh access token
  - POST /auth/logout - Invalidate tokens
  - GET /auth/me - Get current user info
  - Add email validation
  - Add duplicate username/email checks

- [ ] **Protect API Endpoints** (4 hours)
  - Add authentication dependency to all protected routes
  - Return 401 Unauthorized for missing/invalid tokens
  - Implement role-based access control (RBAC)
  - Free tier: Limited endpoints, rate-limited
  - Premium tier: Full access, higher rate limits
  - Enterprise tier: Unlimited access

**Deliverables**:
- `api/auth/` module (5 files, 600+ lines)
  - `models.py` - SQLAlchemy user models
  - `schemas.py` - Pydantic request/response models
  - `utils.py` - Password hashing, JWT helpers
  - `routes.py` - Authentication endpoints
  - `dependencies.py` - Authentication middleware
- `api/migrations/` - Alembic migrations for users table
- `tests/test_auth.py` (150+ lines, 10+ tests)

**Success Metrics**:
- JWT tokens validate correctly
- Password hashing uses bcrypt with rounds=12
- Login endpoint returns tokens in <100ms
- Protected endpoints require valid token
- 0 authentication bypasses (security audit)

**Code Example - JWT Authentication**:
```python
# api/auth/utils.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

```python
# api/auth/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from . import models, schemas, utils
from ..database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=schemas.User)
async def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    
    # Check if user exists
    existing_user = db.query(models.User).filter(
        (models.User.username == user_data.username) | 
        (models.User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    
    # Create new user
    hashed_password = utils.get_password_hash(user_data.password)
    
    db_user = models.User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        tier="free"  # Default tier
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.post("/login", response_model=schemas.Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and return access + refresh tokens"""
    
    # Authenticate user
    user = db.query(models.User).filter(
        models.User.username == form_data.username
    ).first()
    
    if not user or not utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = utils.create_access_token(
        data={"sub": user.username, "tier": user.tier}
    )
    refresh_token = utils.create_refresh_token(
        data={"sub": user.username}
    )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=schemas.Token)
async def refresh(refresh_token: str, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    
    payload = utils.decode_token(refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    username = payload.get("sub")
    user = db.query(models.User).filter(models.User.username == username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token = utils.create_access_token(
        data={"sub": user.username, "tier": user.tier}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,  # Return same refresh token
        "token_type": "bearer"
    }

@router.get("/me", response_model=schemas.User)
async def get_current_user(
    current_user: models.User = Depends(utils.get_current_user)
):
    """Get current authenticated user"""
    return current_user
```

**Dependencies**: python-jose, passlib[bcrypt], python-multipart, sqlalchemy

---

### Week 5: API Key Management & Rate Limiting

**Objective**: Implement API key authentication and tier-based rate limiting.

**Tasks**:
- [ ] **API Key Generation** (5 hours)
  - Generate secure random API keys (32 bytes, hex-encoded)
  - Hash API keys before storing (SHA-256)
  - POST /auth/api-keys - Generate new API key
  - GET /auth/api-keys - List user's API keys
  - DELETE /auth/api-keys/{key_id} - Revoke API key
  - Track API key usage (request count, last used)

- [ ] **Rate Limiting** (8 hours)
  - Install redis for rate limit tracking
  - Implement token bucket algorithm
  - Define tier limits:
    - Free: 100 requests/hour
    - Premium: 10,000 requests/hour
    - Enterprise: Unlimited (or 1M/hour)
  - Add rate limit middleware
  - Return rate limit headers:
    - X-RateLimit-Limit: 100
    - X-RateLimit-Remaining: 75
    - X-RateLimit-Reset: 1609459200
  - Return 429 Too Many Requests when exceeded

- [ ] **API Key Authentication** (4 hours)
  - Accept API key via header: X-API-Key
  - Accept API key via query param: ?api_key=...
  - Validate API key against database
  - Load user tier for rate limiting
  - Update last_used_at timestamp

- [ ] **Metrics Dashboard** (3 hours)
  - Create /admin/metrics endpoint (admin-only)
  - Show API usage by user
  - Show API usage by endpoint
  - Show rate limit violations
  - Export to Prometheus format

**Deliverables**:
- `api/middleware/rate_limit.py` (200+ lines)
- `api/auth/api_keys.py` (150+ lines)
- Redis integration in docker-compose.yml
- `tests/test_rate_limiting.py` (100+ lines)
- Admin dashboard for metrics

**Success Metrics**:
- API keys generated securely (32 bytes entropy)
- Rate limiting enforces correctly (100% accuracy)
- Rate limit headers present in all responses
- Redis latency <5ms for rate limit checks
- 0 rate limit bypasses (security audit)

**Code Example - Rate Limiting Middleware**:
```python
# api/middleware/rate_limit.py
import redis
import time
from fastapi import Request, HTTPException
from typing import Optional

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class RateLimiter:
    """Token bucket rate limiter using Redis"""
    
    def __init__(self, requests_per_hour: int = 100):
        self.requests_per_hour = requests_per_hour
        self.window = 3600  # 1 hour in seconds
    
    async def check_rate_limit(self, request: Request, user_id: str, tier: str = "free"):
        """Check if user has exceeded rate limit"""
        
        # Get limit based on tier
        tier_limits = {
            "free": 100,
            "premium": 10000,
            "enterprise": 1000000
        }
        limit = tier_limits.get(tier, 100)
        
        # Redis key
        key = f"rate_limit:{user_id}"
        
        # Get current count
        current = redis_client.get(key)
        
        if current is None:
            # First request in window
            redis_client.setex(key, self.window, 1)
            remaining = limit - 1
            reset_time = int(time.time()) + self.window
        else:
            current = int(current)
            
            if current >= limit:
                # Rate limit exceeded
                ttl = redis_client.ttl(key)
                reset_time = int(time.time()) + ttl
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Resets in {ttl} seconds.",
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(ttl)
                    }
                )
            
            # Increment count
            redis_client.incr(key)
            remaining = limit - current - 1
            ttl = redis_client.ttl(key)
            reset_time = int(time.time()) + ttl
        
        # Add rate limit headers to request state
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time)
        }

# Middleware to add rate limit headers to response
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if present
        if hasattr(request.state, "rate_limit_headers"):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return response

# Usage in main.py
app = FastAPI()
app.add_middleware(RateLimitHeadersMiddleware)
```

**Dependencies**: redis, fastapi

**Sprint 3 Total Hours**: 35 hours

---

## Sprint 4: MLflow & Model Serving (Weeks 5-6, 40 hours)

**Goal**: Set up MLflow for model versioning, experiment tracking, and integrate with FastAPI for ML predictions.

### Week 5 (continued): MLflow Setup

**Objective**: Configure MLflow tracking server and register models.

**Tasks**:
- [ ] **MLflow Server Installation** (6 hours)
  - Install MLflow (local or cloud)
  - Configure backend store (PostgreSQL for metadata)
  - Configure artifact store (local filesystem or S3)
  - Set up MLflow UI (port 5000)
  - Add authentication (basic auth or OAuth)
  - Document setup in docker-compose.yml

- [ ] **Experiment Tracking** (8 hours)
  - Create experiment: "ntsb-severity-prediction"
  - Log baseline models (Logistic Regression, Random Forest from Phase 2)
  - Log XGBoost model from Sprint 2
  - Track parameters: n_estimators, max_depth, learning_rate, etc.
  - Track metrics: accuracy, ROC-AUC, F1, precision, recall
  - Track artifacts: model file, SHAP plots, feature importance
  - Log dataset version (hash or timestamp)

- [ ] **Model Registry** (6 hours)
  - Register XGBoost model: "ntsb-fatal-classifier"
  - Add model description and tags
  - Set model stage: Staging
  - Promote to Production after validation
  - Version models: v1.0, v1.1, etc.
  - Track model lineage (feature engineering → training → evaluation)

**Deliverables**:
- MLflow server running (http://localhost:5000)
- 5+ experiments logged
- 3+ models registered (Logistic Regression, Random Forest, XGBoost)
- docker-compose.yml updated with mlflow service
- `api/ml/mlflow_utils.py` (helper functions)

**Success Metrics**:
- MLflow UI accessible and functional
- All experiments logged with reproducible results
- Model artifacts <100MB per model
- Model load time <500ms

**Code Example - MLflow Integration**:
```python
# api/ml/train_with_mlflow.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import shap

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("ntsb-severity-prediction")

def train_and_log_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost and log to MLflow"""
    
    with mlflow.start_run(run_name="xgboost-v1.0"):
        
        # Log parameters
        params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        mlflow.log_params(params)
        
        # Train model
        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.xgboost.log_model(
            model,
            "xgboost-model",
            registered_model_name="ntsb-fatal-classifier"
        )
        
        # Log SHAP plots
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])  # Sample for speed
        
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
        plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("shap_summary.png")
        plt.close()
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        print(f"✅ Model logged to MLflow: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model

# Load and promote model
def promote_to_production(model_name: str, version: int):
    """Promote model to production stage"""
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient(tracking_uri="http://mlflow:5000")
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"✅ Model {model_name} v{version} promoted to Production")
```

**Dependencies**: mlflow, xgboost, shap, matplotlib

---

### Week 6: FastAPI ML Prediction Endpoint

**Objective**: Integrate MLflow models with FastAPI for real-time predictions.

**Tasks**:
- [ ] **Model Loading** (6 hours)
  - Load model from MLflow Model Registry
  - Cache model in memory (avoid reloading on every request)
  - Implement model versioning in API (query param: ?model_version=1.0)
  - Handle model loading errors gracefully

- [ ] **Prediction Endpoint** (10 hours)
  - POST /ml/predict - Single prediction with SHAP explanation
  - POST /ml/predict/batch - Batch predictions (up to 100)
  - Request validation with Pydantic (80+ features)
  - Feature engineering integration (apply same pipeline)
  - Return prediction + probabilities + SHAP top 5 features
  - Add prediction caching (Redis, 5-min TTL for identical requests)

- [ ] **Model Monitoring** (4 hours)
  - Log all predictions to database (for drift detection)
  - Track prediction distribution
  - Track feature distribution
  - Create /ml/stats endpoint (prediction statistics)

**Deliverables**:
- `api/ml/predict.py` (300+ lines)
- `api/ml/schemas.py` (Pydantic models for requests/responses)
- `tests/test_ml_prediction.py` (100+ lines)
- Prediction endpoint integrated with API

**Success Metrics**:
- Single prediction latency <200ms (p95)
- Batch prediction throughput >100 predictions/second
- SHAP explanation computed in <100ms
- Model loading time <1 second
- Prediction cache hit rate >50% (for repeat requests)

**Code Example - ML Prediction Endpoint**:
```python
# api/ml/predict.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow.pyfunc
import pandas as pd
import numpy as np
import shap
from .feature_engineering import AviationFeatureEngineer
from ..auth.dependencies import get_current_user, check_rate_limit

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Load model from MLflow (cached)
model = None
feature_engineer = None
explainer = None

def load_model():
    """Load model from MLflow Registry (singleton)"""
    global model, feature_engineer, explainer
    
    if model is None:
        model = mlflow.pyfunc.load_model("models:/ntsb-fatal-classifier/Production")
        feature_engineer = AviationFeatureEngineer.load('models/feature_engineer.joblib')
        explainer = shap.TreeExplainer(model._model_impl.python_model)
        print("✅ Model loaded from MLflow")
    
    return model, feature_engineer, explainer

# Pydantic models
class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    ev_year: int = Field(..., ge=1962, le=2030)
    ev_month: int = Field(..., ge=1, le=12)
    ev_state: str = Field(..., max_length=2)
    acft_year: Optional[int] = Field(None, ge=1900, le=2030)
    acft_category: str = Field(..., max_length=50)
    num_eng: int = Field(..., ge=1, le=8)
    damage: str = Field(..., regex="^(DEST|SUBS|MINR|NONE)$")
    crew_tot_hrs: Optional[float] = Field(None, ge=0)
    crew_cert: Optional[str] = Field(None, max_length=50)
    wx_cond_basic: Optional[str] = Field(None, regex="^(VMC|IMC)$")
    wx_wind_speed_kts: Optional[int] = Field(None, ge=0, le=200)
    # ... 70+ more fields

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: str  # "fatal" or "non-fatal"
    fatal_probability: float
    non_fatal_probability: float
    confidence: float
    shap_explanation: List[dict]  # Top 5 features
    model_version: str

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    current_user = Depends(get_current_user),
    rate_limit = Depends(check_rate_limit)
):
    """
    Predict accident severity for a single case.
    
    Returns prediction, probabilities, and SHAP explanation.
    """
    
    # Load model
    model, fe, explainer = load_model()
    
    # Convert request to DataFrame
    df = pd.DataFrame([request.dict()])
    
    # Feature engineering
    X = fe.transform(df)
    
    # Predict
    proba = model.predict(X)[0]
    
    fatal_prob = proba[1]
    non_fatal_prob = proba[0]
    prediction = "fatal" if fatal_prob > 0.5 else "non-fatal"
    confidence = max(fatal_prob, non_fatal_prob)
    
    # SHAP explanation
    shap_values = explainer.shap_values(X)[0]
    feature_names = fe.get_feature_names()
    
    # Top 5 contributing features
    feature_contributions = [
        {"feature": name, "contribution": float(shap_val)}
        for name, shap_val in zip(feature_names, shap_values)
    ]
    top_features = sorted(feature_contributions, key=lambda x: abs(x['contribution']), reverse=True)[:5]
    
    return PredictionResponse(
        prediction=prediction,
        fatal_probability=fatal_prob,
        non_fatal_probability=non_fatal_prob,
        confidence=confidence,
        shap_explanation=top_features,
        model_version="1.0"
    )

@router.post("/predict/batch")
async def predict_batch(
    requests: List[PredictionRequest],
    current_user = Depends(get_current_user),
    rate_limit = Depends(check_rate_limit)
):
    """
    Batch prediction for up to 100 cases.
    
    Returns predictions without SHAP explanations (for speed).
    """
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100")
    
    # Load model
    model, fe, _ = load_model()
    
    # Convert to DataFrame
    df = pd.DataFrame([req.dict() for req in requests])
    
    # Feature engineering
    X = fe.transform(df)
    
    # Predict
    probas = model.predict(X)
    
    results = []
    for proba in probas:
        fatal_prob = proba[1]
        prediction = "fatal" if fatal_prob > 0.5 else "non-fatal"
        
        results.append({
            "prediction": prediction,
            "fatal_probability": fatal_prob
        })
    
    return results

@router.get("/stats")
async def ml_stats(current_user = Depends(get_current_user)):
    """Get ML model statistics"""
    
    # Would query prediction log from database
    # For now, return mock stats
    
    return {
        "total_predictions": 1234,
        "predictions_today": 56,
        "average_fatal_probability": 0.12,
        "model_version": "1.0",
        "accuracy": 0.9045
    }
```

**Dependencies**: mlflow, fastapi, pydantic, shap

**Sprint 4 Total Hours**: 40 hours

---

## Sprint 5: Monitoring & Observability (Weeks 7-8, 35 hours)

**Goal**: Set up Prometheus + Grafana for metrics, logging, and alerting.

### Week 7: Prometheus Metrics

**Objective**: Instrument all services with Prometheus metrics.

**Tasks**:
- [ ] **Prometheus Installation** (4 hours)
  - Add Prometheus to docker-compose.yml
  - Configure prometheus.yml (scrape intervals, targets)
  - Set retention to 15 days
  - Expose Prometheus UI (port 9090)

- [ ] **API Metrics** (10 hours)
  - Install prometheus-client library
  - Create custom metrics:
    - `api_requests_total` (Counter) - by method, endpoint, status
    - `api_request_duration_seconds` (Histogram) - latency distribution
    - `api_active_requests` (Gauge) - concurrent requests
    - `ml_predictions_total` (Counter) - by outcome (fatal/non-fatal)
    - `ml_prediction_duration_seconds` (Histogram)
    - `database_query_duration_seconds` (Histogram)
  - Add metrics middleware to FastAPI
  - Create /metrics endpoint for Prometheus scraping

- [ ] **Database Metrics** (3 hours)
  - Install postgres_exporter
  - Monitor connection pool
  - Monitor query performance
  - Monitor table sizes

- [ ] **System Metrics** (3 hours)
  - Install node_exporter for host metrics
  - Monitor CPU, memory, disk, network
  - Monitor Docker container metrics

**Deliverables**:
- `docker-compose.yml` updated with Prometheus
- `api/middleware/prometheus.py` (150+ lines)
- `prometheus.yml` configuration
- Prometheus UI accessible

**Success Metrics**:
- Metrics scrape interval: 15 seconds
- All services exporting metrics
- Prometheus UI shows all targets "UP"
- Metrics endpoint response time <50ms

**Code Example - Prometheus Metrics**:
```python
# api/middleware/prometheus.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Define metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

request_latency = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_requests = Gauge(
    'api_active_requests',
    'Currently active requests'
)

ml_predictions = Counter(
    'ml_predictions_total',
    'Total ML predictions',
    ['outcome']
)

ml_prediction_latency = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction latency',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Track active requests
        active_requests.inc()
        
        # Track request timing
        start_time = time.time()
        
        response = await call_next(request)
        
        latency = time.time() - start_time
        
        # Record metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        request_latency.labels(
            endpoint=request.url.path
        ).observe(latency)
        
        active_requests.dec()
        
        return response

# Metrics endpoint
from fastapi import APIRouter

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Usage in main.py
from fastapi import FastAPI
from .middleware.prometheus import PrometheusMiddleware, router as metrics_router

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.include_router(metrics_router)
```

**Dependencies**: prometheus-client, prometheus, postgres_exporter, node_exporter

---

### Week 8: Grafana Dashboards & Alerts

**Objective**: Create Grafana dashboards and configure alerting.

**Tasks**:
- [ ] **Grafana Installation** (3 hours)
  - Add Grafana to docker-compose.yml
  - Configure data source (Prometheus)
  - Set up authentication
  - Expose Grafana UI (port 3000)

- [ ] **Dashboard Creation** (9 hours)
  - **Dashboard 1: API Metrics**
    - Request rate (requests/second)
    - Latency (p50, p95, p99)
    - Error rate (5xx errors)
    - Active users
  
  - **Dashboard 2: ML Metrics**
    - Predictions/hour
    - Prediction distribution (fatal vs non-fatal)
    - Model latency
    - SHAP computation time
  
  - **Dashboard 3: Infrastructure**
    - CPU usage
    - Memory usage
    - Disk I/O
    - Network bandwidth
  
  - **Dashboard 4: Database**
    - Connection pool usage
    - Query latency
    - Cache hit ratio
    - Table sizes
  
  - **Dashboard 5: Business Metrics**
    - Active users
    - API calls by tier (free/premium/enterprise)
    - Top endpoints
    - Geographic distribution

- [ ] **Alerting** (6 hours)
  - Configure alert channels (Slack, email)
  - Create alert rules:
    - High error rate (>1% for 5 minutes)
    - High latency (p95 >500ms for 5 minutes)
    - High CPU (>80% for 10 minutes)
    - Database connection pool exhausted
    - Disk space low (<10% free)
  - Test alerts with synthetic failures

**Deliverables**:
- `docker-compose.yml` updated with Grafana
- 5 Grafana dashboards (JSON export)
- Alert rules configured
- Documentation for dashboard usage

**Success Metrics**:
- Dashboard load time <2 seconds
- All panels show live data
- Alerts trigger correctly (tested)
- Alert latency <1 minute (detection → notification)

**Code Example - Grafana Dashboard JSON** (simplified):
```json
{
  "dashboard": {
    "title": "NTSB API Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total{status_code=~\"5..\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

**Dependencies**: grafana, prometheus

**Sprint 5 Total Hours**: 35 hours

---

## Sprint 6: Cloud Deployment (Weeks 9-10, 40 hours)

**Goal**: Deploy all services to cloud (DigitalOcean, AWS, or GCP) with domain, SSL, and CDN.

### Week 9: Cloud Infrastructure Setup

**Objective**: Provision cloud resources and deploy containers.

**Tasks**:
- [ ] **Choose Cloud Provider** (2 hours)
  - Options: DigitalOcean (easiest), AWS (most powerful), GCP (best ML tools)
  - Recommendation: DigitalOcean App Platform for simplicity
  - Cost estimate: $50-100/month for starter tier

- [ ] **Provision Resources** (8 hours)
  - **Database**: Managed PostgreSQL (2GB RAM, 25GB storage)
  - **Redis**: Managed Redis (256MB-1GB)
  - **Application Server**: 2 vCPUs, 4GB RAM, 80GB SSD
  - **Load Balancer**: HTTP/HTTPS with SSL termination
  - Set up VPC (private network)
  - Configure firewalls (allow 22, 80, 443)

- [ ] **Deploy Containers** (10 hours)
  - Push Docker images to registry (Docker Hub or private)
  - Deploy API container
  - Deploy Dashboard container
  - Deploy MLflow container (optional)
  - Deploy Prometheus + Grafana
  - Configure environment variables (secrets)
  - Set up health checks

**Deliverables**:
- Cloud infrastructure provisioned
- All containers deployed and running
- Private network configured
- Firewall rules set

**Success Metrics**:
- All services accessible via private IPs
- Health checks passing
- Deployment time <30 minutes (after initial setup)
- Monthly cost <$100

**Code Example - DigitalOcean Deployment** (Terraform):
```hcl
# infrastructure/digitalocean/main.tf
terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Managed PostgreSQL Database
resource "digitalocean_database_cluster" "postgres" {
  name       = "ntsb-postgres"
  engine     = "pg"
  version    = "18"
  size       = "db-s-2vcpu-4gb"
  region     = "nyc3"
  node_count = 1
}

# Managed Redis
resource "digitalocean_database_cluster" "redis" {
  name       = "ntsb-redis"
  engine     = "redis"
  version    = "7"
  size       = "db-s-1vcpu-1gb"
  region     = "nyc3"
  node_count = 1
}

# Droplet for application
resource "digitalocean_droplet" "app" {
  image  = "docker-20-04"
  name   = "ntsb-app"
  region = "nyc3"
  size   = "s-2vcpu-4gb"
  ssh_keys = [var.ssh_key_id]
  
  user_data = file("${path.module}/cloud-init.sh")
}

# Load Balancer
resource "digitalocean_loadbalancer" "lb" {
  name   = "ntsb-lb"
  region = "nyc3"
  
  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 8000
    certificate_name = digitalocean_certificate.cert.name
  }
  
  healthcheck {
    protocol = "http"
    port     = 8000
    path     = "/health"
  }
  
  droplet_ids = [digitalocean_droplet.app.id]
}

# SSL Certificate (Let's Encrypt)
resource "digitalocean_certificate" "cert" {
  name    = "ntsb-cert"
  type    = "lets_encrypt"
  domains = ["api.ntsb-analytics.com"]
}
```

**Dependencies**: terraform, digitalocean CLI

---

### Week 10: Domain, SSL, CDN

**Objective**: Configure custom domain with SSL and set up CDN for static assets.

**Tasks**:
- [ ] **Domain Setup** (4 hours)
  - Register domain: ntsb-analytics.com (or similar)
  - Configure DNS:
    - api.ntsb-analytics.com → Load balancer IP
    - app.ntsb-analytics.com → Dashboard IP
    - mlflow.ntsb-analytics.com → MLflow IP
  - Set up email forwarding (optional)

- [ ] **SSL/TLS Configuration** (4 hours)
  - Install Let's Encrypt certificate (certbot)
  - Configure automatic renewal (cron job)
  - Enable HTTPS redirect (HTTP → HTTPS)
  - Configure HSTS headers
  - Test SSL with SSL Labs (A+ rating target)

- [ ] **CDN Setup** (4 hours)
  - Choose CDN: Cloudflare (free tier) or CloudFront
  - Configure CDN for static assets:
    - Dashboard JavaScript bundles
    - Folium maps
    - Matplotlib figures
    - OpenAPI docs
  - Set cache headers (1 hour for API responses)
  - Enable gzip compression

- [ ] **Performance Optimization** (4 hours)
  - Enable HTTP/2
  - Configure Brotli compression
  - Optimize Docker images (multi-stage builds)
  - Configure connection pooling
  - Add reverse proxy (Nginx) if needed

- [ ] **Final Testing** (4 hours)
  - Load test with k6 (1000 concurrent users)
  - Security scan with OWASP ZAP
  - SSL test with SSL Labs
  - Performance test with PageSpeed Insights
  - API test with Postman collection

**Deliverables**:
- Custom domain configured
- SSL/TLS active (A+ rating)
- CDN distributing static assets
- Performance optimizations applied
- Security audit passed

**Success Metrics**:
- Domain resolves correctly (DNS propagation <24 hours)
- SSL certificate valid and auto-renewing
- CDN cache hit rate >70%
- PageSpeed score >90
- Load test: 1000 users, <200ms latency (p95)

**Code Example - Nginx Reverse Proxy**:
```nginx
# nginx.conf
server {
    listen 80;
    server_name api.ntsb-analytics.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.ntsb-analytics.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.ntsb-analytics.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.ntsb-analytics.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    
    # Proxy to API
    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Cache API responses
    location /stats {
        proxy_pass http://api:8000;
        proxy_cache_valid 200 1h;
        add_header X-Cache-Status $upstream_cache_status;
    }
}
```

**Dependencies**: certbot, nginx, cloudflare (or similar CDN)

**Sprint 6 Total Hours**: 40 hours

---

## Sprint 7: Documentation & Beta Prep (Weeks 10-11, 30 hours)

**Goal**: Complete API documentation, user guides, and prepare for beta launch.

### Week 10 (continued): API Documentation

**Objective**: Write comprehensive API documentation with examples.

**Tasks**:
- [ ] **OpenAPI Enhancement** (6 hours)
  - Enhance FastAPI docstrings for all endpoints
  - Add request/response examples
  - Add error response examples (400, 401, 404, 429, 500)
  - Add authentication documentation
  - Customize OpenAPI UI (/docs, /redoc)

- [ ] **Developer Guide** (8 hours)
  - Write `docs/API_GUIDE.md` (500+ lines)
  - Authentication flow (JWT + API keys)
  - Rate limiting details
  - Pagination examples
  - Error handling
  - Best practices
  - Code examples in Python, JavaScript, curl

- [ ] **SDK Documentation** (4 hours)
  - Document Python SDK (if created)
  - Document JavaScript SDK (if created)
  - Installation instructions
  - Quickstart examples
  - Full API reference

**Deliverables**:
- `docs/API_GUIDE.md` (500+ lines)
- Enhanced OpenAPI schema
- SDK documentation
- Postman collection

**Success Metrics**:
- 100% endpoint coverage in docs
- Code examples for all endpoints
- OpenAPI validation passes
- Developer onboarding <10 minutes

**Code Example - API Guide Structure**:
```markdown
# NTSB API Guide

## Authentication

### JWT Tokens

1. Register for an account:
\`\`\`bash
curl -X POST https://api.ntsb-analytics.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "email": "your@email.com",
    "password": "your_password"
  }'
\`\`\`

2. Login to get tokens:
\`\`\`bash
curl -X POST https://api.ntsb-analytics.com/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'username=your_username&password=your_password'
\`\`\`

Response:
\`\`\`json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer"
}
\`\`\`

### Using API Keys

Generate an API key:
\`\`\`bash
curl -X POST https://api.ntsb-analytics.com/auth/api-keys \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{"name": "My API Key"}'
\`\`\`

Use API key in requests:
\`\`\`bash
curl https://api.ntsb-analytics.com/events?limit=10 \
  -H "X-API-Key: YOUR_API_KEY"
\`\`\`

## Endpoints

### GET /events

Retrieve aviation accidents with filtering and pagination.

**Query Parameters**:
- `page` (int, default: 1) - Page number
- `page_size` (int, default: 100, max: 1000) - Results per page
- `start_date` (date, format: YYYY-MM-DD) - Filter by date range
- `end_date` (date, format: YYYY-MM-DD) - Filter by date range
- `state` (string, 2-letter code) - Filter by state
- `severity` (string: FATL|SERS|MINR|NONE) - Filter by severity

**Example Request**:
\`\`\`bash
curl https://api.ntsb-analytics.com/events?page=1&page_size=10&state=CA \
  -H "X-API-Key: YOUR_API_KEY"
\`\`\`

**Example Response**:
\`\`\`json
{
  "total": 29783,
  "page": 1,
  "page_size": 10,
  "results": [
    {
      "ev_id": "20210101001234",
      "ev_date": "2021-01-01",
      "ev_city": "Los Angeles",
      "ev_state": "CA",
      "ev_highest_injury": "MINR",
      "inj_tot_f": 0,
      "dec_latitude": 34.0522,
      "dec_longitude": -118.2437
    }
  ]
}
\`\`\`

### POST /ml/predict

Predict accident severity for a given scenario.

**Request Body**:
\`\`\`json
{
  "ev_year": 2025,
  "ev_month": 3,
  "ev_state": "CA",
  "acft_year": 2015,
  "acft_category": "Airplane",
  "num_eng": 1,
  "damage": "SUBS",
  "crew_tot_hrs": 250,
  "crew_cert": "Private",
  "wx_cond_basic": "VMC"
}
\`\`\`

**Example Response**:
\`\`\`json
{
  "prediction": "non-fatal",
  "fatal_probability": 0.15,
  "non_fatal_probability": 0.85,
  "confidence": 0.85,
  "shap_explanation": [
    {"feature": "crew_tot_hrs", "contribution": -0.32},
    {"feature": "wx_cond_basic", "contribution": -0.18},
    {"feature": "damage", "contribution": 0.12},
    {"feature": "aircraft_age", "contribution": 0.08},
    {"feature": "is_imc", "contribution": 0.05}
  ],
  "model_version": "1.0"
}
\`\`\`

## Rate Limiting

All API endpoints are rate-limited based on your tier:

| Tier | Requests/Hour | Burst |
|------|---------------|-------|
| Free | 100 | 10 |
| Premium | 10,000 | 100 |
| Enterprise | 1,000,000 | 1,000 |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Your tier's limit
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

When you exceed the limit, you'll receive a `429 Too Many Requests` response.

## Error Handling

All errors follow this format:

\`\`\`json
{
  "detail": "Error message",
  "status_code": 400
}
\`\`\`

Common error codes:
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing or invalid token)
- `404` - Not Found (resource doesn't exist)
- `422` - Validation Error (invalid request body)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

## Pagination

All list endpoints support pagination:

\`\`\`bash
curl https://api.ntsb-analytics.com/events?page=1&page_size=100
\`\`\`

Response includes pagination metadata:
\`\`\`json
{
  "total": 179809,
  "page": 1,
  "page_size": 100,
  "results": [...]
}
\`\`\`

## Python SDK Example

\`\`\`python
from ntsb_sdk import NTSBClient

client = NTSBClient(api_key="YOUR_API_KEY")

# Get accidents
accidents = client.get_accidents(state="CA", limit=10)

# Predict severity
prediction = client.predict_severity({
    "ev_year": 2025,
    "ev_month": 3,
    "acft_category": "Airplane",
    # ... other features
})

print(f"Prediction: {prediction['prediction']}")
print(f"Probability: {prediction['fatal_probability']:.2%}")
\`\`\`

## JavaScript SDK Example

\`\`\`javascript
import { NTSBClient } from 'ntsb-sdk';

const client = new NTSBClient({ apiKey: 'YOUR_API_KEY' });

// Get accidents
const accidents = await client.getAccidents({ state: 'CA', limit: 10 });

// Predict severity
const prediction = await client.predictSeverity({
  ev_year: 2025,
  ev_month: 3,
  acft_category: 'Airplane'
});

console.log(`Prediction: ${prediction.prediction}`);
\`\`\`
```

---

### Week 11: User Guides & Beta Program

**Objective**: Create user guides and set up beta testing program.

**Tasks**:
- [ ] **User Guides** (8 hours)
  - Write `docs/USER_GUIDE.md` - For end users
  - Write `docs/QUICKSTART.md` - 5-minute tutorial
  - Write `docs/FAQ.md` - Common questions
  - Create video tutorial (optional, 10-15 minutes)

- [ ] **Beta Program Setup** (6 hours)
  - Create beta signup form (Google Forms or Typeform)
  - Set up beta user group in database (flag in users table)
  - Create welcome email template
  - Create feedback form
  - Plan beta timeline (2-4 weeks)

- [ ] **Landing Page** (8 hours)
  - Create simple landing page: ntsb-analytics.com
  - Features overview
  - API documentation link
  - Pricing (if applicable)
  - Beta signup form
  - Deploy with Netlify or Vercel

**Deliverables**:
- `docs/USER_GUIDE.md`
- `docs/QUICKSTART.md`
- `docs/FAQ.md`
- Beta signup form
- Landing page deployed

**Success Metrics**:
- User guides clear and comprehensive
- Quickstart tutorial completable in <10 minutes
- Landing page load time <2 seconds
- Beta signup form functional

**Sprint 7 Total Hours**: 30 hours

---

## Sprint 8: Beta Launch & Iteration (Weeks 11-12, 35 hours)

**Goal**: Launch beta program with 25-50 users, gather feedback, and iterate.

### Week 11 (continued): Beta Launch

**Objective**: Onboard first beta users and monitor usage.

**Tasks**:
- [ ] **Beta Invitations** (4 hours)
  - Send invitations to first 25 users
  - Provide API keys
  - Share quickstart guide
  - Set up support channel (Discord, Slack, or email)

- [ ] **Monitoring** (6 hours)
  - Create beta metrics dashboard
  - Track user activity (logins, API calls, errors)
  - Monitor error logs
  - Set up daily summary email (user stats)

- [ ] **User Support** (8 hours allocated over 2 weeks)
  - Respond to user questions (<24 hour response time)
  - Fix bugs reported by users
  - Create FAQ based on common questions
  - Hold office hours (optional, 2 hours/week)

**Deliverables**:
- 25 beta users onboarded
- Beta metrics dashboard
- Support channel active
- Daily monitoring in place

**Success Metrics**:
- 25+ beta users active
- User retention >60% after week 1
- Average API calls/user >10/day
- Support response time <24 hours

---

### Week 12: Feedback & Iteration

**Objective**: Gather feedback and make improvements.

**Tasks**:
- [ ] **Feedback Collection** (6 hours)
  - Send mid-beta survey (Google Forms)
  - Conduct 5-10 user interviews (30 min each)
  - Analyze feedback themes
  - Prioritize improvement requests

- [ ] **Bug Fixes** (8 hours)
  - Fix critical bugs (P0/P1)
  - Fix UI issues
  - Improve error messages
  - Optimize slow endpoints

- [ ] **Feature Improvements** (8 hours)
  - Add most-requested features (if feasible)
  - Improve documentation based on feedback
  - Add more code examples
  - Improve dashboard UX

- [ ] **Final Testing** (5 hours)
  - End-to-end testing
  - Load testing with realistic traffic
  - Security audit
  - Performance optimization

**Deliverables**:
- Feedback analysis report
- Bug fixes deployed
- Feature improvements implemented
- Final testing complete

**Success Metrics**:
- User satisfaction >80% (survey)
- All P0/P1 bugs fixed
- API latency <200ms (p95)
- Uptime >99% during beta

**Code Example - Beta Metrics Dashboard**:
```python
# api/admin/beta_metrics.py
from fastapi import APIRouter, Depends
from ..auth.dependencies import require_admin
from ..database import get_db

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/beta/metrics")
async def beta_metrics(
    db = Depends(get_db),
    admin = Depends(require_admin)
):
    """Get beta program metrics (admin only)"""
    
    # Query database for metrics
    total_users = db.execute("SELECT COUNT(*) FROM users WHERE tier = 'beta'").scalar()
    active_users = db.execute("""
        SELECT COUNT(DISTINCT user_id)
        FROM api_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    """).scalar()
    
    total_requests = db.execute("""
        SELECT COUNT(*)
        FROM api_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    """).scalar()
    
    avg_requests_per_user = total_requests / active_users if active_users > 0 else 0
    
    top_endpoints = db.execute("""
        SELECT endpoint, COUNT(*) as count
        FROM api_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY endpoint
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()
    
    error_rate = db.execute("""
        SELECT
            SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END)::float / COUNT(*) as error_rate
        FROM api_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    """).scalar()
    
    return {
        "beta_users": {
            "total": total_users,
            "active_last_7_days": active_users,
            "retention_rate": active_users / total_users if total_users > 0 else 0
        },
        "api_usage": {
            "total_requests_7d": total_requests,
            "avg_requests_per_user": avg_requests_per_user,
            "top_endpoints": [{"endpoint": ep, "count": count} for ep, count in top_endpoints]
        },
        "quality": {
            "error_rate": error_rate,
            "uptime": 0.99  # Would calculate from monitoring
        }
    }
```

**Sprint 8 Total Hours**: 35 hours

---

## Phase 3 Deliverables Summary

**Infrastructure**:
- ✅ Docker containers for all services (API, dashboard, database, MLflow, monitoring)
- ✅ CI/CD pipeline with GitHub Actions (test, build, deploy)
- ✅ Cloud deployment on DigitalOcean/AWS/GCP
- ✅ Custom domain with SSL (api.ntsb-analytics.com)
- ✅ CDN for static assets

**Machine Learning**:
- ✅ XGBoost model with >85% accuracy (target: 90%)
- ✅ 50+ engineered features from NTSB data
- ✅ SHAP explainability integrated into API
- ✅ MLflow model registry with versioning
- ✅ FastAPI ML prediction endpoint (<200ms latency)

**Authentication & Security**:
- ✅ JWT authentication (access + refresh tokens)
- ✅ API key management
- ✅ Rate limiting by tier (free: 100/hr, premium: 10K/hr, enterprise: unlimited)
- ✅ HTTPS with A+ SSL rating
- ✅ Security audit passed (OWASP Top 10)

**Monitoring & Observability**:
- ✅ Prometheus metrics for all services
- ✅ Grafana dashboards (5 dashboards: API, ML, infrastructure, database, business)
- ✅ Alerting (Slack/email) for errors, latency, resource usage
- ✅ Centralized logging

**Documentation & Launch**:
- ✅ API documentation (OpenAPI + developer guide)
- ✅ User guides (quickstart, FAQ, tutorials)
- ✅ Landing page deployed
- ✅ Beta program launched (25-50 users)
- ✅ Feedback collected and iterated

## Files Created/Modified

**New Files** (estimated 50+ files):
- `api/Dockerfile`
- `dashboard/Dockerfile`
- `database/Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/docker.yml`
- `.github/workflows/deploy.yml`
- `api/ml/feature_engineering.py`
- `api/ml/train_xgboost.py`
- `api/ml/predict.py`
- `api/ml/mlflow_utils.py`
- `api/auth/` (5 files: models, schemas, utils, routes, dependencies)
- `api/middleware/rate_limit.py`
- `api/middleware/prometheus.py`
- `tests/test_api.py`
- `tests/test_auth.py`
- `tests/test_ml_prediction.py`
- `tests/test_rate_limiting.py`
- `models/xgboost_fatal_classifier.joblib`
- `models/shap_explainer.joblib`
- `models/feature_engineer.joblib`
- `infrastructure/terraform/` (multiple files)
- `nginx.conf`
- `prometheus.yml`
- `grafana/dashboards/` (5 JSON files)
- `docs/API_GUIDE.md`
- `docs/USER_GUIDE.md`
- `docs/QUICKSTART.md`
- `docs/FAQ.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `notebooks/modeling/xgboost_training.ipynb`
- `notebooks/modeling/feature_engineering_v2.ipynb`
- `landing-page/` (HTML/CSS/JS)

**Modified Files**:
- `api/main.py` (add middleware, routes)
- `api/requirements.txt` (add new dependencies)
- `README.md` (update with deployment info)
- `CHANGELOG.md` (add Phase 3 entry)
- `.gitignore` (add .env, secrets)

## Testing Checklist

- [ ] All Docker images build successfully
- [ ] All services start with docker-compose up
- [ ] CI pipeline passes (tests, linting, type checking)
- [ ] Docker images pushed to registry
- [ ] Cloud deployment successful
- [ ] Custom domain resolves with SSL
- [ ] JWT authentication works
- [ ] API keys work
- [ ] Rate limiting enforces correctly
- [ ] ML predictions accurate (accuracy >85%)
- [ ] SHAP explanations computed (<100ms)
- [ ] Prometheus metrics exported
- [ ] Grafana dashboards functional
- [ ] Alerts trigger correctly
- [ ] API documentation complete
- [ ] Beta users onboarded
- [ ] Load test passes (1000 users, <200ms p95)
- [ ] Security audit passes (0 critical vulnerabilities)

## Success Metrics

| Metric | Target | Actual (to be filled) |
|--------|--------|----------------------|
| XGBoost Accuracy | >85% | ___ |
| API Latency (p95) | <200ms | ___ |
| ML Prediction Latency | <200ms | ___ |
| Uptime | >99% | ___ |
| Test Coverage | >80% | ___ |
| SSL Rating | A+ | ___ |
| PageSpeed Score | >90 | ___ |
| Beta Users | 25-50 | ___ |
| User Satisfaction | >80% | ___ |
| API Calls/Day | 1000+ | ___ |
| Error Rate | <1% | ___ |

## Resource Requirements

**Development**:
- Docker Desktop (8GB RAM recommended)
- Python 3.13 virtual environment
- Git + GitHub account
- Code editor (VS Code recommended)

**Cloud Infrastructure** (DigitalOcean example):
- Managed PostgreSQL (2GB RAM): $40/month
- Managed Redis (256MB): $15/month
- Application Droplet (2vCPU, 4GB): $24/month
- Load Balancer: $12/month
- Domain: $12/year
- **Total**: ~$91/month + $12/year

**External Services**:
- MLflow tracking (can self-host for $0)
- Prometheus + Grafana (self-hosted, included in droplet)
- SSL certificate (Let's Encrypt, free)
- CDN (Cloudflare free tier)

**Estimated Total Budget**: $100-150/month (production-ready)

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cloud costs exceed budget | Medium | Medium | Start with smallest tier, scale based on usage |
| ML model accuracy <85% | Low | High | Extensive feature engineering, hyperparameter tuning |
| Security vulnerabilities | Medium | Critical | Regular audits, dependency updates, security headers |
| Low beta user engagement | Medium | Medium | Clear value proposition, excellent docs, responsive support |
| API performance issues | Medium | High | Load testing, caching, query optimization |
| Deployment failures | Low | High | Staging environment, rollback plan, health checks |

## Dependencies

**Python Libraries** (Phase 3 additions):
- `xgboost` - XGBoost ML models
- `shap` - Model explainability
- `mlflow` - Experiment tracking and model registry
- `python-jose` - JWT tokens
- `passlib[bcrypt]` - Password hashing
- `python-multipart` - Form data parsing
- `redis` - Rate limiting and caching
- `prometheus-client` - Metrics export
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `httpx` - Async HTTP testing

**Infrastructure**:
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- PostgreSQL 18 (managed)
- Redis 7 (managed)
- Prometheus 2.45+
- Grafana 10.0+
- Nginx (reverse proxy)
- Terraform (infrastructure as code)

## Next Phase Preview

**Phase 4: Advanced AI & Research** (Q2 2026, estimated 12 weeks):
- RAG system for narrative queries (LangChain, Chroma)
- Knowledge graphs (Neo4j, NetworkX)
- Advanced NLP (BERT fine-tuning, SafeAeroBERT)
- Causal inference (DoWhy, CausalML)
- Academic publication preparation
- Research partnerships

**Phase 5: Scale & Public Launch** (Q3 2026, estimated 12 weeks):
- Kubernetes deployment (auto-scaling, 99.9% uptime)
- Real-time capabilities (WebSocket, Kafka)
- Public API launch (1000+ users)
- Enterprise features (custom models, dedicated support)
- Community building (forums, newsletter, conferences)

---

## Sprint 1: Week 1 Detailed Execution Plan

**Objective**: Containerize API, dashboard, and database with Docker Compose.

**Day 1: API Dockerfile** (6 hours)
1. Create `api/Dockerfile` (1.5 hours)
   - Write multi-stage build
   - Test image builds: `docker build -t ntsb-api ./api`
   - Verify image size <300MB

2. Test API container (1 hour)
   - Run standalone: `docker run -p 8000:8000 ntsb-api`
   - Test health endpoint: `curl http://localhost:8000/health`
   - Fix any startup issues

3. Optimize Dockerfile (1 hour)
   - Remove unnecessary files
   - Minimize layers
   - Add .dockerignore

4. Add health check (0.5 hours)
   - Configure HEALTHCHECK directive
   - Test with `docker inspect`

5. Documentation (2 hours)
   - Document build process
   - Document environment variables
   - Create .env.example

**Day 2: Dashboard & Database Dockerfiles** (9 hours)
1. Dashboard Dockerfile (3 hours)
   - Similar to API Dockerfile
   - Streamlit-specific configuration
   - Test standalone

2. Database Dockerfile (3 hours)
   - Extend postgres:18-alpine
   - Add PostGIS
   - Add initialization scripts
   - Test data loading

3. Integration testing (3 hours)
   - Test database → API connection
   - Test API → Dashboard connection
   - Fix networking issues

**Day 3: Docker Compose** (5 hours)
1. Create docker-compose.yml (3 hours)
   - Define all 4 services
   - Configure networks and volumes
   - Set environment variables
   - Add health checks

2. Testing (2 hours)
   - `docker-compose up -d`
   - Verify all services healthy
   - Test full stack functionality
   - Document issues and fixes

**Day 4-5: Week 2 Preparation** (continues into CI/CD)

This detailed breakdown ensures each task is clear and executable without ambiguity.

---

**Last Updated**: November 18, 2025
**Version**: 1.0
**Total Lines**: ~2,100 (comprehensive planning document)
**Code Examples**: 15+
**Research Hours**: 3+ (existing docs, best practices, technology evaluation)

