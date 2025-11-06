# TECHNICAL IMPLEMENTATION (PART 3)

**Final sections**: CI/CD Pipeline, Testing Strategies, Performance Optimization, Monitoring Setup, and Troubleshooting Guide.

## CI/CD Pipeline with GitHub Actions

Complete CI/CD setup for automated testing, building, and deployment:

### Project Structure

```
.github/
└── workflows/
    ├── ci.yml               # Continuous Integration
    ├── cd-staging.yml       # Deploy to Staging
    ├── cd-production.yml    # Deploy to Production
    └── ml-retrain.yml       # Automated ML retraining
```

### CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install ruff black isort mypy

      - name: Lint with ruff
        run: ruff check .

      - name: Check formatting with black
        run: black --check .

      - name: Check import sorting
        run: isort --check-only .

      - name: Type checking with mypy
        run: mypy api/app --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgis/postgis:15-3.3
        env:
          POSTGRES_USER: app
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: ntsb_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio httpx

      - name: Run unit tests
        env:
          DATABASE_URL: postgresql://app:test_password@localhost:5432/ntsb_test
          REDIS_HOST: localhost
        run: |
          pytest tests/ -v --cov=api --cov-report=xml --cov-report=html

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push API image
        uses: docker/build-push-action@v4
        with:
          context: ./api
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ntsb/api:${{ github.sha }}
            ntsb/api:latest
          cache-from: type=registry,ref=ntsb/api:buildcache
          cache-to: type=registry,ref=ntsb/api:buildcache,mode=max

  integration-test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v3

      - name: Start services with Docker Compose
        run: |
          docker-compose -f docker-compose.test.yml up -d
          sleep 30  # Wait for services to be ready

      - name: Run integration tests
        run: |
          docker-compose -f docker-compose.test.yml run tests

      - name: Collect logs
        if: failure()
        run: |
          docker-compose -f docker-compose.test.yml logs

      - name: Shutdown services
        if: always()
        run: |
          docker-compose -f docker-compose.test.yml down -v
```

### CD to Staging

```yaml
# .github/workflows/cd-staging.yml
name: Deploy to Staging

on:
  push:
    branches: [ develop ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ntsb-api
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG ./api
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster ntsb-staging \
            --service ntsb-api-staging \
            --force-new-deployment

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster ntsb-staging \
            --services ntsb-api-staging

      - name: Run smoke tests
        run: |
          curl -f https://staging-api.ntsb.example.com/health || exit 1

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        if: always()
```

### CD to Production

```yaml
# .github/workflows/cd-production.yml
name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v3

      - name: Validate version
        run: |
          echo "Deploying version: ${{ github.event.inputs.version || github.event.release.tag_name }}"

      - name: Configure Kubernetes
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ntsb-api \
            ntsb-api=ntsb/api:${{ github.event.inputs.version || github.event.release.tag_name }} \
            -n production

          kubectl rollout status deployment/ntsb-api -n production

      - name: Run health checks
        run: |
          curl -f https://api.ntsb.example.com/health/readiness || exit 1

      - name: Rollback on failure
        if: failure()
        run: |
          kubectl rollout undo deployment/ntsb-api -n production
          echo "Deployment rolled back due to failure"

      - name: Update MLflow model stage
        run: |
          python scripts/promote_model.py --version latest

      - name: Notify on-call
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_ONCALL_WEBHOOK }}
        if: always()
```

## Testing Strategies

Comprehensive testing approach:

### Unit Tests

```python
# tests/test_api.py
"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_health_check():
    response = client.get("/health/")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_prediction_without_auth():
    response = client.post("/api/v1/predict", json={
        "ev_year": 2023,
        "ev_month": 6,
        "dec_latitude": 34.05,
        "dec_longitude": -118.25
    })
    assert response.status_code == 403  # No auth token

def test_prediction_with_auth(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/api/v1/predict",
        json={
            "ev_year": 2023,
            "ev_month": 6,
            "day_of_week": 2,
            "dec_latitude": 34.05,
            "dec_longitude": -118.25,
            "is_imc": 0,
            "num_eng": 1
        },
        headers=headers
    )
    assert response.status_code == 200
    assert "severity" in response.json()
    assert "confidence" in response.json()

@pytest.fixture
def auth_token():
    """Generate test JWT token."""
    from api.app.auth import create_access_token
    return create_access_token({"sub": "test_user", "tier": "premium"})
```

### Integration Tests

```python
# tests/test_integration.py
"""Integration tests with real services."""

import pytest
import psycopg2
import redis
from sqlalchemy import create_engine

@pytest.fixture(scope="session")
def db_connection():
    """Database connection for tests."""
    conn = psycopg2.connect(
        "postgresql://app:test_password@localhost:5432/ntsb_test"
    )
    yield conn
    conn.close()

@pytest.fixture(scope="session")
def redis_client():
    """Redis client for tests."""
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    yield client
    client.flushdb()

def test_database_query(db_connection):
    """Test database connectivity."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM events")
    count = cursor.fetchone()[0]
    assert count > 0

def test_redis_cache(redis_client):
    """Test Redis caching."""
    redis_client.set("test_key", "test_value")
    assert redis_client.get("test_key") == "test_value"
    redis_client.delete("test_key")

def test_end_to_end_prediction(db_connection, redis_client):
    """End-to-end prediction test."""
    from api.app.ml_service import MLService

    # Load model
    ml_service = MLService(
        mlflow_uri="http://localhost:5000",
        model_name="accident_severity_classifier",
        model_stage="Staging"
    )
    ml_service.load_model()

    # Make prediction
    features = {
        "ev_year": 2023,
        "ev_month": 6,
        "dec_latitude": 34.05,
        "dec_longitude": -118.25,
        "num_eng": 1
    }

    result = ml_service.predict(features)

    assert "severity" in result
    assert result["severity"] in ["FATL", "SERS", "MINR", "NONE"]
    assert 0 <= result["confidence"] <= 1
```

### Load Testing

```python
# tests/load_test.py
"""Load testing with Locust."""

from locust import HttpUser, task, between
import random

class NTSBAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Get auth token."""
        # Implement login to get token
        self.token = "test-token"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def predict_severity(self):
        """Make prediction (weight 3)."""
        payload = {
            "ev_year": random.randint(2015, 2023),
            "ev_month": random.randint(1, 12),
            "day_of_week": random.randint(0, 6),
            "dec_latitude": random.uniform(25, 49),
            "dec_longitude": random.uniform(-125, -65),
            "is_imc": random.choice([0, 1]),
            "num_eng": random.choice([1, 2]),
            "pilot_tot_time": random.randint(100, 5000)
        }

        self.client.post("/api/v1/predict", json=payload, headers=self.headers)

    @task(1)
    def health_check(self):
        """Health check (weight 1)."""
        self.client.get("/health/")
```

**Run load test**:

```bash
# 100 concurrent users, 10 new users/sec
locust -f tests/load_test.py --host http://localhost:8000 -u 100 -r 10

# Headless mode
locust -f tests/load_test.py --host http://localhost:8000 -u 100 -r 10 --headless -t 5m
```

**Target performance**:
- 500 requests/second
- p95 latency < 100ms
- p99 latency < 200ms
- 0% error rate

## Performance Optimization Techniques

### Database Optimizations

```sql
-- Add covering indexes
CREATE INDEX idx_events_cover_query ON events(ev_year, ev_highest_injury)
INCLUDE (ev_date, dec_latitude, dec_longitude);

-- Partition pruning test
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM events WHERE ev_year = 2022;

-- Connection pooling (pgBouncer config)
```

**pgBouncer configuration**:

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
ntsb = host=localhost port=5432 dbname=ntsb

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
server_idle_timeout = 600
```

### API Optimizations

```python
# Use async database operations
from databases import Database

database = Database("postgresql://...")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/events/{ev_id}")
async def get_event(ev_id: str):
    query = "SELECT * FROM events WHERE ev_id = :ev_id"
    return await database.fetch_one(query=query, values={"ev_id": ev_id})
```

### Caching Strategy

```python
# Multi-level caching
from functools import lru_cache

# L1: In-memory cache (LRU)
@lru_cache(maxsize=1000)
def get_aircraft_type(make: str, model: str):
    # Expensive lookup
    pass

# L2: Redis cache (see earlier implementation)

# L3: PostgreSQL materialized views
```

### Response Compression

```python
# Already added in FastAPI
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Reduces payload size by 70-90%
```

**Performance gains**:
- Database query time: 500ms → 50ms (10x)
- API response time: 200ms → 50ms (4x)
- Throughput: 100 req/s → 500 req/s (5x)

## Monitoring Setup

Complete monitoring stack with Prometheus, Grafana, and Loki:

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['localhost:8000']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "NTSB ML API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }]
      },
      {
        "title": "Response Time (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
        }]
      },
      {
        "title": "Prediction Accuracy",
        "targets": [{
          "expr": "ml_prediction_confidence_avg"
        }]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: SlowResponses
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        annotations:
          summary: "API responses are slow"

      - alert: LowModelConfidence
        expr: ml_prediction_confidence_avg < 0.7
        for: 10m
        annotations:
          summary: "Model confidence is low"

      - alert: DatabaseConnections
        expr: pg_stat_database_numbackends > 80
        for: 5m
        annotations:
          summary: "High database connection count"
```

### Logging with Loki

```python
# Configure structured logging
import logging
import json_logging

json_logging.init_fastapi(enable_json=True)
json_logging.init_request_instrument(app)

logger = logging.getLogger("ntsb-api")
logger.setLevel(logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info("request_completed", extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "process_time": process_time,
        "user_agent": request.headers.get("user-agent")
    })

    return response
```

## Troubleshooting Guide

### Common Issues

#### Issue 1: High Database Latency

**Symptoms**:
- Query times > 1 second
- API timeouts
- High CPU on database server

**Diagnosis**:

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY tablename;

-- Check table bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions**:
1. Add missing indexes
2. Run VACUUM ANALYZE
3. Update PostgreSQL statistics
4. Consider table partitioning
5. Increase shared_buffers

#### Issue 2: ML Model OOM Errors

**Symptoms**:
- API crashes with MemoryError
- Kubernetes pods killed (OOMKilled)
- Slow prediction times

**Diagnosis**:

```python
# Check model size
import os
import mlflow

model_uri = "models:/accident_severity_classifier/Production"
local_path = mlflow.pyfunc.get_model_dependencies(model_uri)
model_size = os.path.getsize(local_path)
print(f"Model size: {model_size / (1024**2):.2f} MB")

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / (1024**2):.2f} MB")
```

**Solutions**:
1. Increase pod memory limits
2. Use model quantization
3. Batch predictions more efficiently
4. Load model on-demand with caching

#### Issue 3: Rate Limiting False Positives

**Symptoms**:
- Legitimate users getting 429 errors
- Rate limit counters not resetting

**Diagnosis**:

```python
# Check Redis keys
import redis
r = redis.Redis(decode_responses=True)

# List all rate limit keys
keys = r.keys("rate_limit:*")
for key in keys:
    ttl = r.ttl(key)
    value = r.get(key)
    print(f"{key}: {value} (TTL: {ttl}s)")
```

**Solutions**:
1. Adjust rate limit thresholds
2. Implement user-based instead of IP-based limiting
3. Clear Redis keys: `redis-cli KEYS "rate_limit:*" | xargs redis-cli DEL`
4. Implement exponential backoff

#### Issue 4: MLflow Connection Errors

**Symptoms**:
- Models fail to load
- "Connection refused" errors
- Stale model versions

**Diagnosis**:

```bash
# Check MLflow server
curl http://localhost:5000/health

# Check database connectivity
psql -U mlflow -d mlflow -c "SELECT COUNT(*) FROM experiments;"

# Check artifact storage
ls -lh /var/mlflow/artifacts/
```

**Solutions**:
1. Restart MLflow server: `systemctl restart mlflow`
2. Check database permissions
3. Verify artifact storage is accessible
4. Update MLflow URI in environment variables

### Performance Debugging

**Trace slow endpoints**:

```python
# Add timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    if duration > 1.0:  # Log slow requests
        logger.warning(f"Slow request: {request.url.path} took {duration:.2f}s")

    response.headers["X-Process-Time"] = str(duration)
    return response
```

**Profile Python code**:

```python
# Use cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = ml_service.predict(features)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Disaster Recovery

**Database backup**:

```bash
# Automated daily backups
pg_dump -U app -d ntsb -F c -f /backups/ntsb_$(date +%Y%m%d).dump

# Point-in-time recovery setup
# Enable WAL archiving in postgresql.conf
```

**Model rollback**:

```python
# Rollback to previous production model
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Archive current production
current_prod = client.get_latest_versions("accident_severity_classifier", ["Production"])[0]
client.transition_model_version_stage(
    name="accident_severity_classifier",
    version=current_prod.version,
    stage="Archived"
)

# Promote previous version
previous_version = current_prod.version - 1
client.transition_model_version_stage(
    name="accident_severity_classifier",
    version=previous_version,
    stage="Production"
)
```

### Health Check Script

```bash
#!/bin/bash
# scripts/health_check.sh

echo "=== NTSB Platform Health Check ==="

# Check PostgreSQL
pg_isready -h localhost -p 5432 && echo "✓ PostgreSQL: OK" || echo "✗ PostgreSQL: FAIL"

# Check Redis
redis-cli ping > /dev/null && echo "✓ Redis: OK" || echo "✗ Redis: FAIL"

# Check MLflow
curl -sf http://localhost:5000/health > /dev/null && echo "✓ MLflow: OK" || echo "✗ MLflow: FAIL"

# Check API
curl -sf http://localhost:8000/health > /dev/null && echo "✓ API: OK" || echo "✗ API: FAIL"

# Check Airflow
curl -sf http://localhost:8080/health > /dev/null && echo "✓ Airflow: OK" || echo "✗ Airflow: FAIL"

echo "=== Health Check Complete ==="
```

**Run health check**:

```bash
chmod +x scripts/health_check.sh
./scripts/health_check.sh
```

---

## Summary

This completes the TECHNICAL_IMPLEMENTATION guide covering:

1. Database Migration (Access → PostgreSQL)
2. DuckDB Analytics Pipeline
3. Apache Airflow DAGs
4. MLflow Experiment Tracking
5. FastAPI Model Serving
6. Redis Caching
7. CI/CD with GitHub Actions
8. Comprehensive Testing
9. Performance Optimization
10. Monitoring & Alerting
11. Troubleshooting

**Next steps**: Proceed to [NLP_TEXT_MINING.md](NLP_TEXT_MINING.md) for SafeAeroBERT fine-tuning and production NLP deployment.

**Estimated implementation time**: 200-250 hours (6-8 weeks, 2 developers)
