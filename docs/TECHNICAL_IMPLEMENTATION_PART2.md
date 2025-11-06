# TECHNICAL IMPLEMENTATION (PART 2)

**Continuation from TECHNICAL_IMPLEMENTATION.md**

This document contains the remaining sections: FastAPI Model Serving, Redis Caching, CI/CD, Testing, Performance Optimization, Monitoring, and Troubleshooting.

## FastAPI Model Serving

Complete production-ready FastAPI application for model serving:

### Project Structure

```
api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── auth.py              # JWT authentication
│   ├── rate_limit.py        # Rate limiting
│   ├── models.py            # Pydantic models
│   ├── ml_service.py        # ML model wrapper
│   └── routers/
│       ├── __init__.py
│       ├── predictions.py   # Prediction endpoints
│       ├── health.py        # Health checks
│       └── admin.py         # Admin endpoints
├── tests/
│   ├── test_api.py
│   ├── test_auth.py
│   └── test_predictions.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Complete FastAPI Application

```python
# api/app/main.py
"""
Production FastAPI application for NTSB ML model serving.
"""

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator

from .config import settings
from .auth import verify_token
from .rate_limit import RateLimiter
from .ml_service import MLService
from .routers import predictions, health, admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global ML service instance
ml_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup/shutdown."""
    global ml_service

    # Startup
    logger.info("Starting NTSB ML API...")

    # Load ML model
    ml_service = MLService(
        mlflow_uri=settings.MLFLOW_TRACKING_URI,
        model_name=settings.MODEL_NAME,
        model_stage=settings.MODEL_STAGE
    )
    ml_service.load_model()

    logger.info("ML model loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down NTSB ML API...")

# Create FastAPI app
app = FastAPI(
    title="NTSB Accident Severity Prediction API",
    description="Production ML API for predicting aviation accident severity",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Rate limiter
rate_limiter = RateLimiter()

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "path": str(request.url)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "NTSB Accident Severity Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }
```

### Configuration

```python
# api/app/config.py
"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://ntsb.example.com"]

    # Authentication
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Rate limiting
    RATE_LIMIT_FREE: str = "100/hour"
    RATE_LIMIT_PREMIUM: str = "1000/hour"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = None

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MODEL_NAME: str = "accident_severity_classifier"
    MODEL_STAGE: str = "Production"

    # Database
    DATABASE_URL: str = "postgresql://app:dev_password@localhost:5432/ntsb"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
```

### JWT Authentication

```python
# api/app/auth.py
"""JWT authentication."""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from .config import settings

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify JWT token and return payload."""
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(payload: dict = Security(verify_token)) -> dict:
    """Get current user from token payload."""
    return payload
```

### Rate Limiting

```python
# api/app/rate_limit.py
"""Redis-based rate limiting."""

import redis
from fastapi import HTTPException, Request, status
from datetime import timedelta
import hashlib
from .config import settings

class RateLimiter:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )

    def _get_key(self, identifier: str, tier: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{tier}:{identifier}"

    def check_rate_limit(self, request: Request, user_tier: str = "free"):
        """Check if request is within rate limit."""

        # Get identifier (IP or user ID)
        identifier = request.client.host
        if hasattr(request.state, "user"):
            identifier = request.state.user.get("user_id", identifier)

        # Hash identifier
        identifier_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]

        # Rate limit configuration
        limits = {
            "free": (100, 3600),      # 100 requests per hour
            "premium": (1000, 3600),  # 1000 requests per hour
        }

        max_requests, window_seconds = limits.get(user_tier, limits["free"])

        key = self._get_key(identifier_hash, user_tier)

        # Increment counter
        current = self.redis_client.incr(key)

        # Set expiration on first request
        if current == 1:
            self.redis_client.expire(key, window_seconds)

        # Check limit
        if current > max_requests:
            ttl = self.redis_client.ttl(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {ttl} seconds.",
                headers={"Retry-After": str(ttl)}
            )

        return {
            "remaining": max_requests - current,
            "limit": max_requests,
            "reset": self.redis_client.ttl(key)
        }
```

### ML Service Wrapper

```python
# api/app/ml_service.py
"""ML model service wrapper."""

import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MLService:
    """ML model service with caching and error handling."""

    def __init__(self, mlflow_uri: str, model_name: str, model_stage: str = "Production"):
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_stage = model_stage
        self.model = None
        self.label_mapping = {0: "FATL", 1: "SERS", 2: "MINR", 3: "NONE"}

        mlflow.set_tracking_uri(mlflow_uri)

    def load_model(self):
        """Load model from MLflow registry."""
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model: {model_uri}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: Dict) -> Dict:
        """Make prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare features
            feature_df = pd.DataFrame([features])

            # Predict
            prediction_proba = self.model.predict(feature_df)

            # Get class and confidence
            if isinstance(prediction_proba, np.ndarray):
                if len(prediction_proba.shape) == 1:
                    # Single prediction
                    predicted_class = int(prediction_proba[0])
                    confidence = 1.0
                else:
                    # Probabilities
                    predicted_class = int(np.argmax(prediction_proba[0]))
                    confidence = float(prediction_proba[0][predicted_class])

                    # All probabilities
                    probabilities = {
                        self.label_mapping[i]: float(prediction_proba[0][i])
                        for i in range(len(prediction_proba[0]))
                    }
            else:
                predicted_class = int(prediction_proba)
                confidence = 1.0
                probabilities = {}

            severity_label = self.label_mapping.get(predicted_class, "UNKNOWN")

            return {
                "severity": severity_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "model": self.model_name,
                "version": self.model_stage
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise
```

### Prediction Router

```python
# api/app/routers/predictions.py
"""Prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict
from ..auth import get_current_user
from ..rate_limit import RateLimiter
from ..ml_service import MLService

router = APIRouter()
rate_limiter = RateLimiter()

class PredictionRequest(BaseModel):
    """Prediction request model."""
    ev_year: int = Field(..., ge=1962, le=2030)
    ev_month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=0, le=6)
    dec_latitude: float = Field(..., ge=-90, le=90)
    dec_longitude: float = Field(..., ge=-180, le=180)
    is_imc: int = Field(0, ge=0, le=1)
    is_low_vis: int = Field(0, ge=0, le=1)
    is_high_wind: int = Field(0, ge=0, le=1)
    wx_temp: Optional[float] = None
    wx_wind_speed: Optional[float] = None
    wx_vis: Optional[float] = None
    is_weekend: int = Field(0, ge=0, le=1)
    is_summer: int = Field(0, ge=0, le=1)
    crew_age: Optional[int] = Field(None, ge=16, le=100)
    pilot_tot_time: Optional[float] = Field(None, ge=0)
    pilot_make_time: Optional[float] = Field(None, ge=0)
    pilot_90_days: Optional[float] = Field(None, ge=0)
    pilot_experience: Optional[float] = Field(None, ge=0, le=4)
    multi_engine: int = Field(0, ge=0, le=1)
    high_risk_phase: int = Field(0, ge=0, le=1)
    num_eng: int = Field(1, ge=1, le=8)

class PredictionResponse(BaseModel):
    """Prediction response model."""
    severity: str
    confidence: float
    probabilities: Dict[str, float]
    model: str
    version: str

@router.post("/predict", response_model=PredictionResponse)
async def predict_severity(
    request: Request,
    pred_request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Predict accident severity."""

    # Rate limiting
    user_tier = user.get("tier", "free")
    rate_limit_info = rate_limiter.check_rate_limit(request, user_tier)

    # Get ML service
    from ..main import ml_service

    if ml_service is None:
        raise HTTPException(status_code=503, detail="ML service not available")

    # Prepare features
    features = pred_request.dict()

    # Make prediction
    try:
        result = ml_service.predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch_predict")
async def batch_predict(
    request: Request,
    predictions: list[PredictionRequest],
    user: dict = Depends(get_current_user)
):
    """Batch prediction endpoint."""

    # Check batch size
    if len(predictions) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100")

    # Rate limiting
    user_tier = user.get("tier", "free")
    rate_limiter.check_rate_limit(request, user_tier)

    from ..main import ml_service

    results = []
    for pred_request in predictions:
        features = pred_request.dict()
        result = ml_service.predict(features)
        results.append(result)

    return {"predictions": results, "count": len(results)}
```

### Health Check Router

```python
# api/app/routers/health.py
"""Health check endpoints."""

from fastapi import APIRouter, status
from datetime import datetime
import psutil

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/readiness")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    from ..main import ml_service

    if ml_service is None or ml_service.model is None:
        return {"status": "not_ready", "reason": "ML model not loaded"}, status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ready"}

@router.get("/liveness")
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}

@router.get("/metrics")
async def system_metrics():
    """System metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
```

### Running the API

**Development**:

```bash
# Start API
uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000

# With auto-reload
uvicorn api.app.main:app --reload --log-level info
```

**Production**:

```bash
# With Gunicorn
gunicorn api.app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
```

**Docker**:

```dockerfile
# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "api.app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

**Test API**:

```bash
# Get JWT token (implement login endpoint)
TOKEN="your-jwt-token"

# Make prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ev_year": 2023,
    "ev_month": 6,
    "day_of_week": 2,
    "dec_latitude": 34.05,
    "dec_longitude": -118.25,
    "is_imc": 0,
    "is_low_vis": 0,
    "is_high_wind": 0,
    "pilot_tot_time": 1500,
    "num_eng": 1,
    "multi_engine": 0,
    "high_risk_phase": 1
  }'
```

**Expected response**:

```json
{
  "severity": "MINR",
  "confidence": 0.87,
  "probabilities": {
    "FATL": 0.05,
    "SERS": 0.08,
    "MINR": 0.87,
    "NONE": 0.00
  },
  "model": "accident_severity_classifier",
  "version": "Production"
}
```

## Redis Caching Strategy

Multi-level caching for optimal performance:

### Installation

```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: maxmemory 2gb
# Set: maxmemory-policy allkeys-lru

# Restart
sudo systemctl restart redis-server
```

### Cache Implementation

```python
# api/app/cache.py
"""Redis caching utilities."""

import redis
import json
import hashlib
from functools import wraps
from typing import Callable, Any
from .config import settings

class Cache:
    """Redis cache wrapper."""

    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )

    def get(self, key: str) -> Any:
        """Get value from cache."""
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        self.client.setex(key, ttl, json.dumps(value))

    def delete(self, key: str):
        """Delete key from cache."""
        self.client.delete(key)

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern."""
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    @staticmethod
    def cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

# Global cache instance
cache = Cache()

def cached(ttl: int = 3600, prefix: str = ""):
    """Caching decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{prefix}:{cache.cache_key(*args, **kwargs)}"

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)

            return result
        return wrapper
    return decorator
```

**Usage**:

```python
from .cache import cached

@cached(ttl=3600, prefix="prediction")
async def get_prediction(features: dict):
    """Cached prediction."""
    return ml_service.predict(features)
```

### Cache Warming Script

```python
# scripts/warm_cache.py
"""Pre-warm Redis cache with common queries."""

import redis
import pandas as pd
from sqlalchemy import create_engine

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
engine = create_engine("postgresql://app:dev_password@localhost:5432/ntsb")

# Pre-compute common aggregations
queries = {
    "yearly_stats": "SELECT ev_year, COUNT(*) FROM events GROUP BY ev_year",
    "state_stats": "SELECT ev_state, COUNT(*) FROM events GROUP BY ev_state",
    "top_aircraft": "SELECT acft_make, acft_model, COUNT(*) FROM aircraft GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 100"
}

for key, query in queries.items():
    df = pd.read_sql(query, engine)
    redis_client.setex(f"cache:{key}", 86400, df.to_json())  # 24h TTL

print("Cache warmed successfully")
```

**Expected cache hit rate**: 60-80% for prediction endpoints

---

**Continue to TECHNICAL_IMPLEMENTATION_PART3.md for CI/CD, Testing, Performance, Monitoring, and Troubleshooting.**
