# TECHNICAL DEBT

Code quality issues, refactoring priorities, and technical improvements across all project phases.

**Last Updated**: November 2025
**Priority Legend**: 🔴 High | 🟡 Medium | 🟢 Low

---

## Code Quality Issues

### Database Layer

#### 1. Complete MDB to PostgreSQL Migration 🔴
- **Issue**: Legacy .mdb files still in use; incomplete migration to PostgreSQL
- **Impact**: Poor query performance, difficult to maintain, no referential integrity
- **Effort**: 40 hours
- **Priority**: High (Phase 1 blocker)
- **Solution**:
  - Migrate all 3 MDB files (avall.mdb, Pre2008.mdb, PRE1982.MDB)
  - Implement foreign key constraints
  - Create materialized views for complex joins
  - Add full-text search indexes on narratives table

#### 2. Schema Normalization 🟡
- **Issue**: Denormalized tables with duplicate data (e.g., aircraft makes/models)
- **Impact**: Data inconsistencies, increased storage, update anomalies
- **Effort**: 20 hours
- **Priority**: Medium
- **Solution**:
  - Create lookup tables: aircraft_makes, aircraft_models, airports
  - Normalize events, aircraft, and crew tables
  - Add database constraints and triggers

#### 3. Missing Indexes 🔴
- **Issue**: Slow queries on frequently accessed columns (ev_id, Aircraft_Key, occurrence_code)
- **Impact**: API response time >500ms for complex queries
- **Effort**: 8 hours
- **Priority**: High
- **Solution**:
  - Add B-tree indexes on foreign keys
  - Add GIN indexes for full-text search
  - Add composite indexes for common query patterns
  - Monitor query performance with pg_stat_statements

### API Layer

#### 4. Missing Input Validation 🔴
- **Issue**: API endpoints lack comprehensive input validation (SQL injection risk)
- **Impact**: Security vulnerabilities, potential data corruption
- **Effort**: 15 hours
- **Priority**: High
- **Solution**:
  - Use Pydantic models for all request/response schemas
  - Add regex validation for dates, IDs, codes
  - Implement rate limiting and request size limits
  - Add input sanitization for user-generated content

#### 5. No Request Pagination 🟡
- **Issue**: GET /events returns all records (100K+), causing timeouts
- **Impact**: API performance degradation, poor UX
- **Effort**: 10 hours
- **Priority**: Medium
- **Solution**:
  - Implement cursor-based pagination (offset/limit)
  - Add page size limits (max 1000 records)
  - Return total count in headers
  - Document pagination in API docs

#### 6. Inconsistent Error Handling 🟡
- **Issue**: Some endpoints return 500 errors instead of 400/404
- **Impact**: Poor API experience, difficult debugging
- **Effort**: 12 hours
- **Priority**: Medium
- **Solution**:
  - Standardize error response format: `{"error": "message", "code": 400}`
  - Use FastAPI exception handlers
  - Add detailed error messages with context
  - Log all errors to centralized logging (Sentry, CloudWatch)

### Machine Learning

#### 7. No Model Versioning 🔴
- **Issue**: ML models not versioned; difficult to rollback if performance degrades
- **Impact**: Risk of deploying broken models, no audit trail
- **Effort**: 15 hours
- **Priority**: High
- **Solution**:
  - Implement MLflow Model Registry
  - Tag models with versions (v1.0, v1.1, etc.)
  - Track model lineage: training data → hyperparameters → metrics
  - Automate model deployment with CI/CD

#### 8. Feature Engineering Not Modular 🟡
- **Issue**: Feature engineering code scattered across notebooks; not reusable
- **Impact**: Difficult to reproduce, training/serving skew
- **Effort**: 25 hours
- **Priority**: Medium
- **Solution**:
  - Create AviationFeatureEngineer class
  - Implement fit/transform methods (scikit-learn API)
  - Serialize feature pipelines with joblib
  - Add unit tests for feature transformations

#### 9. No Model Monitoring 🔴
- **Issue**: Production models not monitored for drift, performance degradation
- **Impact**: Silent model failures, inaccurate predictions
- **Effort**: 30 hours
- **Priority**: High
- **Solution**:
  - Implement Evidently AI for drift detection
  - Track data drift (KS test, PSI)
  - Track prediction drift (distribution shifts)
  - Set up alerts (Slack/email) for drift detection

### NLP & AI

#### 10. BERT Model Not Optimized 🟡
- **Issue**: SafeAeroBERT model size 400MB+, slow inference (500ms)
- **Impact**: High latency for narrative classification
- **Effort**: 20 hours
- **Priority**: Medium
- **Solution**:
  - Apply model quantization (INT8, ONNX)
  - Use DistilBERT or TinyBERT (50% smaller)
  - Implement model caching (Redis)
  - Batch inference for throughput

#### 11. RAG Chunking Strategy 🟡
- **Issue**: Fixed chunk size (512 tokens) doesn't work for all narratives
- **Impact**: Loss of context, poor retrieval quality
- **Effort**: 15 hours
- **Priority**: Medium
- **Solution**:
  - Implement semantic chunking (split on sentences/paragraphs)
  - Use overlapping chunks (50-100 token overlap)
  - Experiment with chunk sizes (256, 512, 1024)
  - A/B test chunking strategies

#### 12. Knowledge Graph Disconnected 🟢
- **Issue**: 20%+ of nodes disconnected (no relationships)
- **Impact**: Incomplete graph, poor traversals
- **Effort**: 18 hours
- **Priority**: Low
- **Solution**:
  - Improve relationship extraction (LLM prompts)
  - Add implicit relationships (co-occurrence, temporal)
  - Run entity resolution to merge duplicates
  - Validate graph connectivity

---

## Refactoring Priorities

### High Impact

#### 1. Modularize ETL Pipeline 🔴
- **Current**: Single monolithic Airflow DAG
- **Target**: 5+ reusable DAGs (extract, transform, load, validate, feature_eng)
- **Benefit**: Easier to debug, parallel execution, independent scaling
- **Effort**: 35 hours

#### 2. Separate API from Business Logic 🔴
- **Current**: FastAPI routes contain business logic (queries, transformations)
- **Target**: 3-tier architecture (API → Service → Repository)
- **Benefit**: Testability, reusability, clean separation of concerns
- **Effort**: 40 hours

#### 3. Extract Configuration to Environment Variables 🔴
- **Current**: Hardcoded config values (database URLs, API keys)
- **Target**: Use .env files, Kubernetes ConfigMaps/Secrets
- **Benefit**: Security, flexibility, easier deployment
- **Effort**: 10 hours

### Medium Impact

#### 4. Replace mdbtools with DuckDB 🟡
- **Current**: mdbtools for MDB extraction (slow, dependencies)
- **Target**: Use DuckDB to query MDB files directly (10x faster)
- **Benefit**: Faster extraction, fewer dependencies, SQL queries
- **Effort**: 15 hours

#### 5. Consolidate Duplicate Code 🟡
- **Current**: Copy-pasted code for data validation, error handling
- **Target**: Create shared utility libraries (ntsb_utils/)
- **Benefit**: DRY principle, easier maintenance
- **Effort**: 20 hours

#### 6. Improve Notebook Organization 🟡
- **Current**: 50+ Jupyter notebooks with inconsistent naming
- **Target**: Organize by phase (eda/, modeling/, analysis/), add README
- **Benefit**: Easier to find, reproducibility
- **Effort**: 8 hours

### Low Impact

#### 7. Upgrade Python Dependencies 🟢
- **Current**: Some packages 2-3 versions behind (pandas 1.5.3 → 2.2.0)
- **Target**: Update to latest stable versions
- **Benefit**: Bug fixes, performance improvements, new features
- **Effort**: 12 hours (includes testing)

#### 8. Standardize Logging 🟢
- **Current**: Inconsistent logging (print statements, custom loggers)
- **Target**: Use structlog with JSON formatting
- **Benefit**: Centralized logging, easier to parse, better debugging
- **Effort**: 10 hours

#### 9. Add Type Hints 🟢
- **Current**: 30% of Python code lacks type hints
- **Target**: 90%+ coverage with mypy static type checking
- **Benefit**: Better IDE support, catch bugs early
- **Effort**: 15 hours

---

## Legacy Code Migration

### Phase 1: Database Layer

1. **MDB → PostgreSQL** (40 hours)
   - Extract all tables with mdb-export
   - Transform data (fix dates, clean nulls, normalize)
   - Load with COPY command (10x faster than INSERT)
   - Validate record counts, data types

2. **Add Foreign Keys** (10 hours)
   - events → aircraft (ev_id)
   - aircraft → engines (Aircraft_Key)
   - events → findings (ev_id)
   - Add ON DELETE CASCADE where appropriate

3. **Create Materialized Views** (8 hours)
   - accident_summary_view (pre-joined events + aircraft + injuries)
   - monthly_stats_view (aggregated monthly statistics)
   - Refresh strategy: daily or on-demand

### Phase 2: ETL Pipeline

1. **Airflow DAGs Refactoring** (35 hours)
   - Split monolithic DAG into 5 modular DAGs
   - Add task dependencies (TaskGroups)
   - Implement dynamic DAG generation (for monthly updates)
   - Add retries and alerting

2. **Data Quality Framework** (25 hours)
   - Replace custom validation with Great Expectations
   - Create 50+ expectations (data types, ranges, nulls)
   - Add data quality dashboard (Grafana)
   - Automate quality reports (weekly)

### Phase 3: API Layer

1. **FastAPI Refactoring** (40 hours)
   - Move business logic to service layer
   - Create repository pattern for database access
   - Add dependency injection
   - Improve test coverage (>80%)

2. **Authentication Overhaul** (20 hours)
   - Replace custom JWT with Auth0 or Clerk
   - Add OAuth2 support (Google, GitHub)
   - Implement RBAC with permissions

---

## Performance Bottlenecks

### Database

#### 1. Slow Queries 🔴
- **Query**: `SELECT * FROM events WHERE narrative LIKE '%engine failure%'`
- **Current**: 5-10 seconds (full table scan)
- **Target**: <100ms with full-text search index
- **Solution**:
  ```sql
  CREATE INDEX idx_narrative_fts ON events USING GIN(to_tsvector('english', narrative));

  -- Query with full-text search
  SELECT * FROM events WHERE to_tsvector('english', narrative) @@ to_tsquery('engine & failure');
  ```

#### 2. Large Table Joins 🔴
- **Query**: JOIN events + aircraft + findings (100K+ records)
- **Current**: 2-5 seconds
- **Target**: <500ms with materialized view
- **Solution**:
  ```sql
  CREATE MATERIALIZED VIEW accident_details AS
  SELECT e.*, a.make, a.model, f.cause_factor
  FROM events e
  LEFT JOIN aircraft a ON e.ev_id = a.ev_id
  LEFT JOIN findings f ON e.ev_id = f.ev_id;

  -- Refresh daily
  REFRESH MATERIALIZED VIEW CONCURRENTLY accident_details;
  ```

### API

#### 3. No Response Caching 🟡
- **Endpoint**: GET /stats (expensive aggregation query)
- **Current**: Computed on every request (2-3 seconds)
- **Target**: <50ms with Redis caching
- **Solution**:
  ```python
  import redis
  import json

  redis_client = redis.Redis(host='redis', port=6379)

  @app.get("/stats")
  async def get_stats():
      # Check cache
      cached = redis_client.get("stats")
      if cached:
          return json.loads(cached)

      # Compute stats
      stats = compute_stats()

      # Cache for 1 hour
      redis_client.setex("stats", 3600, json.dumps(stats))

      return stats
  ```

### Machine Learning

#### 4. Slow Feature Engineering 🟡
- **Current**: Feature engineering on single CPU (5 minutes for 100K records)
- **Target**: <1 minute with parallelization
- **Solution**:
  ```python
  from joblib import Parallel, delayed

  def engineer_features_batch(batch):
      return feature_engineer.transform(batch)

  # Parallel processing
  batches = np.array_split(df, 10)
  results = Parallel(n_jobs=10)(delayed(engineer_features_batch)(batch) for batch in batches)
  df_features = pd.concat(results)
  ```

#### 5. ML Model Inference 🟡
- **Current**: 200ms per prediction (XGBoost)
- **Target**: <50ms with batching
- **Solution**:
  ```python
  # Batch prediction API
  @app.post("/ml/predict/batch")
  async def predict_batch(requests: list[PredictionRequest]):
      X = pd.DataFrame([req.dict() for req in requests])
      X = feature_engineer.transform(X)

      # Batch prediction (10x faster than individual)
      predictions = model.predict_proba(X)

      return [
          {"fatal_probability": pred[1]}
          for pred in predictions
      ]
  ```

---

## Testing Coverage Gaps

### Current Coverage: 45%

#### Missing Tests

1. **API Endpoints** (20% coverage)
   - Unit tests for request validation
   - Integration tests for database queries
   - Load tests for performance

2. **Feature Engineering** (30% coverage)
   - Unit tests for each feature transformer
   - Edge cases (nulls, outliers, invalid codes)
   - Integration tests (end-to-end pipeline)

3. **ML Models** (40% coverage)
   - Unit tests for model training
   - Performance regression tests
   - Data drift detection tests

4. **ETL Pipeline** (50% coverage)
   - DAG validation tests
   - Data quality tests
   - Error handling tests

### Target Coverage: 80%+

**Testing Strategy**:
```python
# API endpoint test
def test_get_events():
    response = client.get("/events?limit=10")
    assert response.status_code == 200
    assert len(response.json()) <= 10

# Feature engineering test
def test_temporal_features():
    df = pd.DataFrame({'event_date': ['2024-01-15']})
    features = feature_engineer.create_temporal_features(df)
    assert 'month_sin' in features.columns
    assert -1 <= features['month_sin'].iloc[0] <= 1

# ML model test
def test_xgboost_prediction():
    X_test = pd.DataFrame({...})  # Sample features
    predictions = model.predict_proba(X_test)
    assert predictions.shape == (len(X_test), 2)
    assert np.allclose(predictions.sum(axis=1), 1.0)
```

---

## Documentation Gaps

### Missing Documentation

1. **API Reference** 🔴
   - Complete OpenAPI specification
   - Request/response examples for all endpoints
   - Error codes and handling

2. **Database Schema** 🔴
   - Entity-relationship diagrams (ERD)
   - Table descriptions, column definitions
   - Index and constraint documentation

3. **Deployment Guide** 🟡
   - Kubernetes deployment steps
   - Configuration management (Helm values)
   - Troubleshooting guide

4. **User Guides** 🟡
   - Quickstart tutorial (5 minutes)
   - Advanced usage (ML predictions, RAG queries)
   - Dashboard walkthrough

5. **Developer Guide** 🟢
   - Project structure overview
   - Contributing guidelines
   - Code style guide (Black, isort, flake8)

---

## Dependency Updates Needed

### Critical (Security)

- `requests` 2.28.2 → 2.31.0 (CVE-2023-32681)
- `pillow` 9.5.0 → 10.1.0 (CVE-2023-44271)
- `transformers` 4.30.0 → 4.36.0 (security fixes)

### Important (Features)

- `pandas` 1.5.3 → 2.2.0 (performance improvements, nullable dtypes)
- `polars` 0.18.0 → 0.20.0 (10-20% faster)
- `fastapi` 0.100.0 → 0.109.0 (new features, bug fixes)
- `pydantic` 1.10.12 → 2.5.3 (v2 rewrite, 20x faster validation)

### Nice-to-Have (Performance)

- `scikit-learn` 1.3.0 → 1.4.0 (optimized algorithms)
- `torch` 2.0.1 → 2.1.2 (faster inference, better CUDA support)
- `xgboost` 1.7.6 → 2.0.3 (GPU training improvements)

---

## Estimated Total Effort

| Category | Hours |
|----------|-------|
| Database refactoring | 78 |
| API improvements | 72 |
| ML/NLP optimization | 88 |
| Testing & documentation | 85 |
| Dependency updates | 30 |
| **Total** | **353 hours** |

**Recommendation**: Prioritize high-impact, high-priority items (🔴) first. Tackle over 2-3 months alongside new feature development.

---

**Last Updated**: November 2025
**Version**: 1.0
