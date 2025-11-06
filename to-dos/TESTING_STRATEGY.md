# TESTING STRATEGY

Comprehensive testing approach for NTSB Aviation Accident Database Analysis Platform.

**Last Updated**: November 2025
**Target Coverage**: 80%+ (unit + integration)

---

## Test Pyramid Structure

```
         /\
        /  \  E2E Tests (5%)
       /____\
      /      \  Integration Tests (15%)
     /________\
    /          \  Unit Tests (80%)
   /__________  \
```

**Distribution**:
- **Unit Tests**: 80% (fast, isolated, high coverage)
- **Integration Tests**: 15% (API endpoints, database queries)
- **End-to-End Tests**: 5% (critical user flows)

---

## Unit Testing

### Database Layer

#### 1. Data Validation Tests
```python
import pytest
from validation import DataValidator

def test_validate_coordinates():
    """Test geospatial coordinate validation"""
    validator = DataValidator()

    # Valid coordinates
    assert validator.validate_lat_lon(37.7749, -122.4194) == True  # San Francisco

    # Invalid coordinates
    assert validator.validate_lat_lon(91.0, 0.0) == False  # Latitude > 90
    assert validator.validate_lat_lon(0.0, 181.0) == False  # Longitude > 180
    assert validator.validate_lat_lon(None, None) == False  # Null values

def test_validate_ntsb_codes():
    """Test NTSB code validation against lexicon"""
    validator = DataValidator()

    # Valid occurrence codes (100-430)
    assert validator.validate_occurrence_code(100) == True
    assert validator.validate_occurrence_code(200) == True

    # Invalid codes
    assert validator.validate_occurrence_code(99) == False
    assert validator.validate_occurrence_code(500) == False

def test_validate_dates():
    """Test date format validation"""
    validator = DataValidator()

    # Valid dates
    assert validator.validate_date("2024-01-15") == True
    assert validator.validate_date("2024-12-31") == True

    # Invalid dates
    assert validator.validate_date("2024-13-01") == False  # Invalid month
    assert validator.validate_date("invalid") == False
```

#### 2. ETL Pipeline Tests
```python
from etl import MDBExtractor, DataCleaner, PostgresLoader

def test_mdb_extraction():
    """Test MDB file extraction"""
    extractor = MDBExtractor("tests/fixtures/test.mdb")
    tables = extractor.extract_all_tables()

    assert "events" in tables
    assert len(tables["events"]) > 0
    assert "ev_id" in tables["events"].columns

def test_data_cleaning():
    """Test data cleaning transformations"""
    df = pd.DataFrame({
        'event_date': ['01/15/2024', '2024-03-20', 'invalid'],
        'latitude': ['37.7749', None, '91.0'],
    })

    cleaner = DataCleaner()
    cleaned = cleaner.clean(df)

    # Check date conversion
    assert pd.api.types.is_datetime64_any_dtype(cleaned['event_date'])

    # Check invalid coordinates removed
    assert cleaned['latitude'].max() <= 90.0
    assert cleaned['latitude'].isna().sum() == 2  # None and invalid

def test_postgres_loading():
    """Test PostgreSQL data loading"""
    loader = PostgresLoader(connection_string="postgresql://test@localhost/test_db")

    df = pd.DataFrame({
        'ev_id': ['20240001', '20240002'],
        'event_date': ['2024-01-15', '2024-01-16'],
    })

    # Load data
    loader.load(df, table='events')

    # Verify
    result = loader.query("SELECT COUNT(*) FROM events")
    assert result[0][0] == 2
```

### API Layer

#### 3. FastAPI Endpoint Tests
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_events():
    """Test GET /events endpoint"""
    response = client.get("/events?limit=10")

    assert response.status_code == 200
    assert len(response.json()) <= 10
    assert "ev_id" in response.json()[0]

def test_get_events_pagination():
    """Test pagination"""
    # First page
    response1 = client.get("/events?limit=10&offset=0")
    data1 = response1.json()

    # Second page
    response2 = client.get("/events?limit=10&offset=10")
    data2 = response2.json()

    # Verify different data
    assert data1[0]['ev_id'] != data2[0]['ev_id']

def test_get_event_by_id():
    """Test GET /events/{ev_id}"""
    response = client.get("/events/20240001")

    assert response.status_code == 200
    assert response.json()['ev_id'] == '20240001'

def test_get_event_not_found():
    """Test 404 error handling"""
    response = client.get("/events/invalid_id")

    assert response.status_code == 404
    assert "not found" in response.json()['detail'].lower()

def test_authentication_required():
    """Test JWT authentication"""
    response = client.post("/ml/predict", json={"features": {}})

    assert response.status_code == 401  # Unauthorized

def test_rate_limiting():
    """Test rate limiting enforcement"""
    # Make 101 requests (limit: 100)
    for i in range(101):
        response = client.get("/events?limit=1")

    assert response.status_code == 429  # Too Many Requests
```

### Feature Engineering

#### 4. Feature Transformer Tests
```python
from feature_engineering import AviationFeatureEngineer, TemporalFeatures, SpatialFeatures

def test_temporal_cyclical_encoding():
    """Test cyclical encoding of temporal features"""
    df = pd.DataFrame({
        'event_month': [1, 6, 12],
        'event_hour': [0, 12, 23]
    })

    transformer = TemporalFeatures()
    features = transformer.create_cyclical_features(df)

    # Check sine/cosine columns exist
    assert 'month_sin' in features.columns
    assert 'month_cos' in features.columns

    # Check value ranges [-1, 1]
    assert features['month_sin'].between(-1, 1).all()
    assert features['month_cos'].between(-1, 1).all()

    # Check December (12) is close to January (1)
    dec_sin = features.loc[features['event_month'] == 12, 'month_sin'].values[0]
    jan_sin = features.loc[features['event_month'] == 1, 'month_sin'].values[0]
    assert abs(dec_sin - jan_sin) < 0.3

def test_spatial_lag_features():
    """Test K-nearest neighbors spatial lag"""
    df = pd.DataFrame({
        'latitude': [37.7749, 37.8, 37.85, 40.7128],  # SF, SF, SF, NYC
        'longitude': [-122.4194, -122.3, -122.2, -74.0060],
        'severity': [1, 2, 1, 3]
    })

    transformer = SpatialFeatures()
    features = transformer.create_spatial_lag_features(df, k=2)

    # Check spatial lag columns exist
    assert 'spatial_lag_k1' in features.columns
    assert 'spatial_lag_k2' in features.columns

    # SF points should have similar neighbors, NYC should differ
    assert features.loc[0, 'spatial_lag_k1'] in [1, 2]  # SF neighbors
    assert features.loc[3, 'spatial_lag_k1'] != 3  # NYC neighbor is SF

def test_ntsb_code_extraction():
    """Test NTSB code extraction"""
    df = pd.DataFrame({
        'occurrence_code': ['100,200,300', '110', None],
        'phase_code': [520, 600, 500]  # TAKEOFF, LANDING, STANDING
    })

    transformer = AviationFeatureEngineer()
    features = transformer.extract_ntsb_codes(df)

    # Check binary features created
    assert 'occurrence_100' in features.columns
    assert 'occurrence_200' in features.columns
    assert 'phase_takeoff' in features.columns

    # Check values
    assert features.loc[0, 'occurrence_100'] == 1
    assert features.loc[1, 'occurrence_100'] == 0
    assert features.loc[0, 'phase_takeoff'] == 1
```

### Machine Learning

#### 5. ML Model Tests
```python
from models import AccidentSeverityClassifier
import numpy as np

def test_xgboost_prediction_shape():
    """Test XGBoost output shape"""
    model = AccidentSeverityClassifier.load('models/xgboost_v1.pkl')

    X_test = pd.DataFrame(np.random.rand(100, 50))  # 100 samples, 50 features
    predictions = model.predict_proba(X_test)

    # Binary classification: (n_samples, 2)
    assert predictions.shape == (100, 2)

    # Probabilities sum to 1
    assert np.allclose(predictions.sum(axis=1), 1.0)

def test_model_feature_importance():
    """Test feature importance exists"""
    model = AccidentSeverityClassifier.load('models/xgboost_v1.pkl')

    feature_importance = model.feature_importances_

    assert len(feature_importance) == 50  # 50 features
    assert feature_importance.sum() > 0

def test_model_performance():
    """Test model meets accuracy threshold"""
    model = AccidentSeverityClassifier.load('models/xgboost_v1.pkl')
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90  # 90% minimum accuracy

def test_shap_values():
    """Test SHAP value computation"""
    model = AccidentSeverityClassifier.load('models/xgboost_v1.pkl')
    X_test = pd.DataFrame(np.random.rand(10, 50))

    shap_values = model.compute_shap_values(X_test)

    assert shap_values.shape == X_test.shape
    assert not np.isnan(shap_values).any()
```

### NLP & RAG

#### 6. NLP Pipeline Tests
```python
from nlp import NarrativePreprocessor, SafeAeroBERT, RAGSystem

def test_text_preprocessing():
    """Test narrative preprocessing"""
    preprocessor = NarrativePreprocessor()

    text = "The Cessna 172 aircraft crashed during takeoff at KJFK airport due to engine failure."
    clean = preprocessor.preprocess(text)

    # Check stopwords removed
    assert 'the' not in clean.lower()
    assert 'at' not in clean.lower()

    # Check lemmatization
    assert 'crash' in clean or 'crashed' in clean  # lemmatized

def test_bert_classification():
    """Test SafeAeroBERT classification"""
    model = SafeAeroBERT.load('models/safe_aero_bert')

    narrative = "Aircraft experienced engine failure, resulting in fatal crash."
    prediction = model.predict(narrative)

    # Check output format
    assert 'label' in prediction
    assert 'confidence' in prediction
    assert prediction['confidence'] >= 0.0 and prediction['confidence'] <= 1.0

def test_rag_retrieval():
    """Test RAG document retrieval"""
    rag = RAGSystem(vector_db_path='tests/fixtures/test_faiss.index')

    query = "engine failure during takeoff"
    results = rag.retrieve(query, top_k=5)

    # Check results
    assert len(results) == 5
    assert all('ev_id' in r for r in results)
    assert all('similarity' in r for r in results)

    # Check similarity scores descending
    scores = [r['similarity'] for r in results]
    assert scores == sorted(scores, reverse=True)
```

---

## Integration Testing

### API Integration Tests

#### 7. Database Integration
```python
import pytest
from sqlalchemy import create_engine
from api import app

@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine("postgresql://test@localhost/test_ntsb")
    # Create tables, load fixtures
    yield engine
    # Teardown

def test_api_database_query(test_db):
    """Test API querying database"""
    client = TestClient(app)

    response = client.get("/events?start_date=2024-01-01&end_date=2024-01-31")

    assert response.status_code == 200
    data = response.json()

    # Verify database query executed
    assert all(d['event_date'] >= '2024-01-01' for d in data)
    assert all(d['event_date'] <= '2024-01-31' for d in data)

def test_ml_prediction_with_features(test_db):
    """Test ML prediction endpoint with database features"""
    client = TestClient(app)

    # Fetch event
    event = client.get("/events/20240001").json()

    # Predict severity
    response = client.post("/ml/predict", json={
        "ev_id": event['ev_id'],
        "use_stored_features": True
    })

    assert response.status_code == 200
    assert 'fatal_probability' in response.json()
```

### ETL Integration Tests

#### 8. End-to-End ETL
```python
def test_etl_pipeline_e2e():
    """Test complete ETL pipeline"""
    # Extract
    extractor = MDBExtractor("tests/fixtures/test.mdb")
    events_df = extractor.extract_table("events")

    # Transform
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean(events_df)

    # Load
    loader = PostgresLoader("postgresql://test@localhost/test_ntsb")
    loader.load(cleaned_df, table='events')

    # Validate
    result = loader.query("SELECT COUNT(*) FROM events")
    assert result[0][0] == len(events_df)

    # Check data quality
    quality_score = loader.query("SELECT AVG(CASE WHEN latitude IS NOT NULL THEN 1 ELSE 0 END) FROM events")[0][0]
    assert quality_score > 0.95  # 95%+ have valid coordinates
```

---

## Data Drift Testing

### Model Monitoring

#### 9. Data Drift Detection Tests
```python
from monitoring import DriftDetector
from scipy.stats import ks_2samp

def test_data_drift_detection():
    """Test Kolmogorov-Smirnov drift detection"""
    detector = DriftDetector()

    # Reference data (training)
    reference = pd.DataFrame(np.random.normal(0, 1, 1000))

    # No drift (same distribution)
    current_no_drift = pd.DataFrame(np.random.normal(0, 1, 1000))
    assert detector.detect_drift(reference, current_no_drift) == False

    # Drift detected (different distribution)
    current_drift = pd.DataFrame(np.random.normal(5, 1, 1000))
    assert detector.detect_drift(reference, current_drift) == True

def test_prediction_drift():
    """Test prediction distribution drift"""
    detector = DriftDetector()

    # Reference predictions (training)
    reference_preds = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5] * 200)

    # Current predictions (production)
    current_preds = pd.Series([0.6, 0.7, 0.8, 0.9, 1.0] * 200)

    drift = detector.detect_prediction_drift(reference_preds, current_preds)

    assert drift['drift_detected'] == True
    assert drift['p_value'] < 0.05

def test_feature_importance_drift():
    """Test feature importance stability"""
    model_v1 = XGBClassifier()
    model_v2 = XGBClassifier()

    # Train both models
    model_v1.fit(X_train, y_train)
    model_v2.fit(X_train, y_train)

    # Compare feature importances
    fi_v1 = model_v1.feature_importances_
    fi_v2 = model_v2.feature_importances_

    # Check correlation (should be high if stable)
    correlation = np.corrcoef(fi_v1, fi_v2)[0, 1]
    assert correlation > 0.90  # 90%+ correlation
```

### Performance Regression Tests

#### 10. Model Performance Monitoring
```python
def test_model_performance_regression():
    """Test model performance hasn't degraded"""
    model_current = load_model('models/xgboost_v1.pkl')
    X_test, y_test = load_test_data()

    # Current performance
    current_accuracy = accuracy_score(y_test, model_current.predict(X_test))

    # Baseline performance (from training logs)
    baseline_accuracy = 0.91

    # Allow 2% degradation
    assert current_accuracy >= baseline_accuracy - 0.02

def test_inference_latency():
    """Test model inference latency"""
    model = load_model('models/xgboost_v1.pkl')
    X_test = pd.DataFrame(np.random.rand(100, 50))

    import time
    start = time.time()
    predictions = model.predict_proba(X_test)
    latency = (time.time() - start) / len(X_test)

    # Average latency < 10ms per prediction
    assert latency < 0.01
```

---

## Load Testing

### API Load Tests (Locust)

#### 11. Load Test Script
```python
from locust import HttpUser, task, between

class NTSBAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(3)
    def get_events(self):
        """GET /events (most common endpoint)"""
        self.client.get("/events?limit=50")

    @task(2)
    def get_event_by_id(self):
        """GET /events/{ev_id}"""
        ev_ids = ["20240001", "20240002", "20240003"]
        self.client.get(f"/events/{random.choice(ev_ids)}")

    @task(1)
    def ml_predict(self):
        """POST /ml/predict"""
        self.client.post("/ml/predict", json={
            "aircraft_age": 25,
            "pilot_hours": 500,
            # ... 100+ features
        })

    @task(1)
    def rag_query(self):
        """POST /rag/query"""
        self.client.post("/rag/query", json={
            "query": "What causes engine failures?",
            "top_k": 5
        })

# Run: locust -f tests/load/test_api.py --host=https://api.ntsb-analytics.com
# Target: 1000 concurrent users, <200ms p95 latency
```

---

## Security Testing

### OWASP Top 10

#### 12. Security Test Cases
```python
def test_sql_injection_protection():
    """Test SQL injection vulnerability"""
    client = TestClient(app)

    # Attempt SQL injection
    response = client.get("/events?ev_id=20240001' OR '1'='1")

    # Should be sanitized, return 404 or 400 (not 200 with all data)
    assert response.status_code != 200

def test_xss_protection():
    """Test XSS vulnerability"""
    client = TestClient(app)

    # Attempt XSS
    response = client.post("/rag/query", json={
        "query": "<script>alert('XSS')</script>"
    })

    # Response should escape HTML
    assert "<script>" not in response.text

def test_authentication_bypass():
    """Test authentication can't be bypassed"""
    client = TestClient(app)

    # Attempt to access protected endpoint without token
    response = client.post("/ml/predict", json={})

    assert response.status_code == 401  # Unauthorized

    # Attempt with invalid token
    response = client.post("/ml/predict",
                           json={},
                           headers={"Authorization": "Bearer invalid_token"})

    assert response.status_code == 401

def test_rate_limit_bypass():
    """Test rate limiting can't be bypassed"""
    client = TestClient(app)

    # Make 101 requests (limit: 100)
    for i in range(101):
        response = client.get("/events?limit=1")

    # Last request should be rate limited
    assert response.status_code == 429

    # Try with different user agent (shouldn't bypass)
    response = client.get("/events?limit=1",
                          headers={"User-Agent": "Different"})
    assert response.status_code == 429
```

---

## CI/CD Testing Automation

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest tests/unit -v --cov=. --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Check coverage
        run: |
          coverage report --fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Test Coverage Goals

| Component | Current | Target |
|-----------|---------|--------|
| Database | 60% | 85% |
| API | 70% | 90% |
| Feature Engineering | 50% | 80% |
| ML Models | 40% | 75% |
| NLP/RAG | 30% | 70% |
| **Overall** | **45%** | **80%** |

---

## Testing Best Practices

1. **Test Naming**: Use descriptive names (`test_validate_coordinates` not `test_1`)
2. **Fixtures**: Use pytest fixtures for setup/teardown
3. **Mocking**: Mock external services (LLM APIs, databases in unit tests)
4. **Parameterization**: Use `@pytest.mark.parametrize` for multiple test cases
5. **Assertions**: Use specific assertions (`assert x == 5` not `assert x`)
6. **Test Data**: Use fixtures in `tests/fixtures/`, not production data
7. **Isolation**: Tests should be independent (no shared state)
8. **Speed**: Unit tests <1s, integration tests <10s, E2E tests <60s

---

**Last Updated**: November 2025
**Version**: 1.0
