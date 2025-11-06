# PHASE 5: PRODUCTION

Kubernetes deployment, public API, real-time capabilities, and production monitoring.

**Timeline**: Q1 2026 (12 weeks, January-March 2026)
**Prerequisites**: Phase 1-4 complete, all services containerized
**Team**: 2-3 developers (DevOps engineer + backend engineer)
**Estimated Hours**: ~300 hours total

## Overview

| Sprint | Duration | Focus Area | Key Deliverables | Hours |
|--------|----------|------------|------------------|-------|
| Sprint 1 | Weeks 1-3 | Kubernetes Deployment | Helm charts, HPA, 99.9% uptime | 80h |
| Sprint 2 | Weeks 4-6 | Public API | Auth, rate limiting, SDKs | 75h |
| Sprint 3 | Weeks 7-9 | Real-time Capabilities | WebSocket, Kafka, <100ms latency | 75h |
| Sprint 4 | Weeks 10-12 | Monitoring & DR | Prometheus, Grafana, disaster recovery | 70h |

## Sprint 1: Kubernetes Deployment (Weeks 1-3, January 2026)

**Goal**: Deploy all services to Kubernetes with 99.9% uptime SLA.

### Week 1: Containerization & Helm Charts

**Tasks**:
- [ ] Create Dockerfiles for all services: API, ML serving, RAG, dashboard
- [ ] Optimize Docker images: multi-stage builds, <500MB per image
- [ ] Push images to container registry (Docker Hub, AWS ECR, or GCR)
- [ ] Design Helm chart structure: values.yaml, templates/
- [ ] Create Kubernetes manifests: Deployments, Services, ConfigMaps, Secrets
- [ ] Set resource requests/limits: CPU (0.5-2 cores), memory (1-4GB)
- [ ] Configure health checks: liveness and readiness probes

**Deliverables**:
- Dockerfiles for 5+ services
- Helm chart (ntsb-analytics) with all components
- Container registry with versioned images

**Success Metrics**:
- All images < 500MB (optimized)
- Helm chart deploys successfully on local Kubernetes (minikube)
- Health checks pass within 30 seconds

**Research Finding**: Kubernetes best practices (2024) - Use multi-stage Docker builds to reduce image size by 60-80%. Set resource requests = 70% of limits to ensure QoS. Use readiness probes to prevent traffic to unhealthy pods.

**Code Example**:
```dockerfile
# Multi-stage Dockerfile for FastAPI service
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# Helm values.yaml
api:
  replicaCount: 3
  image:
    repository: ntsb/api
    tag: "1.0.0"
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2
      memory: 4Gi
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10
  readinessProbe:
    httpGet:
      path: /ready
      port: 8000
    initialDelaySeconds: 10
    periodSeconds: 5

postgresql:
  enabled: true
  auth:
    database: ntsb
    username: ntsb_user

redis:
  enabled: true
  architecture: standalone
```

**Dependencies**: docker, helm, kubectl

### Week 2: HPA & Load Balancing

**Tasks**:
- [ ] Install Metrics Server for HPA (Horizontal Pod Autoscaling)
- [ ] Configure HPA for API service: min 2, max 10 pods, target 70% CPU
- [ ] Configure HPA for ML serving: min 1, max 5 pods, target 80% CPU
- [ ] Install NGINX Ingress Controller
- [ ] Configure Ingress: routing rules, TLS/SSL certificates (Let's Encrypt)
- [ ] Set up sticky sessions for Streamlit dashboard
- [ ] Load test with k6 or Locust: simulate 1000 concurrent users

**Deliverables**:
- HPA configured for critical services
- NGINX Ingress with TLS/SSL
- Load balancer distributing traffic across pods

**Success Metrics**:
- HPA scales up under load (70% CPU triggers scale-out)
- HPA scales down after 5 minutes of low utilization
- Load balancer latency < 10ms
- TLS certificates auto-renew

**Research Finding**: Kubernetes HPA best practices (2024) - Use HPA with custom metrics (not just CPU/memory) for better scaling decisions. Set scaleDownStabilizationWindowSeconds to 300 to prevent flapping. Use KEDA for event-driven autoscaling (e.g., scale on queue length).

**Code Example**:
```yaml
# HPA for API service
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50  # Scale down 50% at a time
        periodSeconds: 60

---
# NGINX Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ntsb-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.ntsb-analytics.com
    secretName: ntsb-tls
  rules:
  - host: api.ntsb-analytics.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 8000
```

**Dependencies**: metrics-server, nginx-ingress-controller, cert-manager

### Week 3: Deployment & Validation

**Tasks**:
- [ ] Deploy to production Kubernetes cluster (AWS EKS, GCP GKE, or Azure AKS)
- [ ] Configure persistent volumes: PostgreSQL, Neo4j, FAISS index
- [ ] Set up namespace isolation: dev, staging, prod
- [ ] Configure network policies: restrict pod-to-pod communication
- [ ] Run smoke tests: API endpoints, ML predictions, dashboard access
- [ ] Monitor metrics: CPU, memory, request rate, latency
- [ ] Calculate uptime: target 99.9% (8.76 hours downtime/year max)

**Deliverables**:
- Production Kubernetes cluster operational
- All services deployed and accessible
- Initial monitoring dashboard

**Success Metrics**:
- Zero downtime during deployment (blue-green or canary)
- API latency p95 < 200ms
- Uptime: 99.9% over first week

**Code Example**:
```bash
# Deploy to production
helm install ntsb-analytics ./helm-chart \
  --namespace prod \
  --create-namespace \
  --values values-prod.yaml

# Verify deployment
kubectl get pods -n prod
kubectl get hpa -n prod

# Run smoke tests
curl https://api.ntsb-analytics.com/health
curl https://api.ntsb-analytics.com/stats

# Monitor resource usage
kubectl top pods -n prod
kubectl top nodes
```

**Dependencies**: kubectl, helm, cloud provider CLI (awscli, gcloud, az)

**Sprint 1 Total Hours**: 80 hours

---

## Sprint 2: Public API (Weeks 4-6, January-February 2026)

**Goal**: Launch public API with authentication, rate limiting, and SDKs.

### Week 4: Authentication & Authorization

**Tasks**:
- [ ] Implement JWT authentication: login endpoint, token generation
- [ ] Add OAuth2 support: Google, GitHub OAuth providers
- [ ] Create API key system: generate, revoke, track usage
- [ ] Implement role-based access control (RBAC): admin, premium, free tiers
- [ ] Set up user database: PostgreSQL or managed auth service (Auth0, Clerk)
- [ ] Create user registration flow: email verification, password reset
- [ ] Secure sensitive endpoints: ML predictions, RAG queries

**Deliverables**:
- JWT + OAuth2 authentication
- API key management system
- RBAC with 3 tiers (free, premium, enterprise)

**Success Metrics**:
- Authentication latency < 50ms
- Token expiration: 1 hour (access), 30 days (refresh)
- 0 authentication bypasses (security audit)

**Code Example**:
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

app = FastAPI()

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_from_db(username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect credentials")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoint
@app.get("/ml/predict")
async def predict(current_user = Depends(get_current_user)):
    # Only accessible with valid token
    return {"prediction": "..."}
```

**Dependencies**: fastapi, python-jose, passlib, bcrypt

### Week 5: Rate Limiting & Quotas

**Tasks**:
- [ ] Implement rate limiting with Redis: token bucket algorithm
- [ ] Define tier limits: Free (100 req/day), Premium (10K req/day), Enterprise (unlimited)
- [ ] Add rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
- [ ] Create usage tracking dashboard: API calls per user, endpoint popularity
- [ ] Implement quota enforcement: block users exceeding limits
- [ ] Set up billing integration: Stripe for premium/enterprise tiers (optional)
- [ ] Document rate limits in API docs

**Deliverables**:
- Rate limiting system (100-10K req/day based on tier)
- Usage tracking dashboard
- Billing integration (optional)

**Success Metrics**:
- Rate limiter latency < 5ms
- 100% enforcement (no bypasses)
- Usage dashboard updates in real-time

**Code Example**:
```python
import redis
from fastapi import Request, HTTPException
import time

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class RateLimiter:
    def __init__(self, requests_per_hour=100):
        self.requests_per_hour = requests_per_hour
        self.window = 3600  # 1 hour in seconds

    async def check_rate_limit(self, request: Request, user_id: str):
        key = f"rate_limit:{user_id}"

        # Get current count
        current = redis_client.get(key)

        if current is None:
            # First request in window
            redis_client.setex(key, self.window, 1)
            remaining = self.requests_per_hour - 1
        else:
            current = int(current)
            if current >= self.requests_per_hour:
                # Rate limit exceeded
                ttl = redis_client.ttl(key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Resets in {ttl} seconds.",
                    headers={"X-RateLimit-Reset": str(int(time.time()) + ttl)}
                )

            # Increment count
            redis_client.incr(key)
            remaining = self.requests_per_hour - current - 1

        # Add rate limit headers to response
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.requests_per_hour),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time()) + self.window)
        }

# Middleware
@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    response = await call_next(request)

    if hasattr(request.state, "rate_limit_headers"):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value

    return response

# Usage
rate_limiter = RateLimiter(requests_per_hour=100)

@app.get("/api/data")
async def get_data(request: Request, user = Depends(get_current_user)):
    await rate_limiter.check_rate_limit(request, user.id)
    return {"data": "..."}
```

**Dependencies**: redis, fastapi

### Week 6: SDK Generation & Documentation

**Tasks**:
- [ ] Generate OpenAPI 3.0 specification (FastAPI auto-generates)
- [ ] Create Python SDK: requests wrapper with authentication
- [ ] Create JavaScript/TypeScript SDK: axios wrapper
- [ ] Create R SDK: httr wrapper (for data scientists)
- [ ] Publish SDKs: PyPI, npm, CRAN
- [ ] Write comprehensive API documentation: endpoints, parameters, examples
- [ ] Create interactive API explorer: Swagger UI, Postman collection
- [ ] Add code examples in 3+ languages

**Deliverables**:
- SDKs for Python, JavaScript, R
- OpenAPI specification (JSON/YAML)
- API documentation website

**Success Metrics**:
- SDKs published to package managers
- Documentation covers 100% of endpoints
- Interactive examples for all endpoints

**Code Example**:
```python
# Python SDK (ntsb-sdk)
import requests

class NTSBClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.ntsb-analytics.com"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_accidents(self, start_date=None, end_date=None, limit=100):
        """Get accidents with optional filters"""
        params = {"limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = self.session.get(f"{self.base_url}/events", params=params)
        response.raise_for_status()
        return response.json()

    def predict_severity(self, features):
        """Predict accident severity"""
        response = self.session.post(f"{self.base_url}/ml/predict", json=features)
        response.raise_for_status()
        return response.json()

    def rag_query(self, query, top_k=5):
        """Query accident narratives with RAG"""
        response = self.session.post(
            f"{self.base_url}/rag/query",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = NTSBClient(api_key="your_api_key")

accidents = client.get_accidents(start_date="2024-01-01", limit=50)
prediction = client.predict_severity({"aircraft_age": 25, "pilot_hours": 500, ...})
rag_result = client.rag_query("What causes engine failures?")
```

```javascript
// JavaScript SDK (ntsb-sdk-js)
class NTSBClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseURL = 'https://api.ntsb-analytics.com';
  }

  async getAccidents(options = {}) {
    const params = new URLSearchParams(options);
    const response = await fetch(`${this.baseURL}/events?${params}`, {
      headers: { 'Authorization': `Bearer ${this.apiKey}` }
    });
    return response.json();
  }

  async predictSeverity(features) {
    const response = await fetch(`${this.baseURL}/ml/predict`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(features)
    });
    return response.json();
  }
}

// Usage
const client = new NTSBClient('your_api_key');
const accidents = await client.getAccidents({ limit: 50 });
```

**Dependencies**: requests (Python), axios (JavaScript), httr (R)

**Sprint 2 Total Hours**: 75 hours

---

## Sprint 3: Real-time Capabilities (Weeks 7-9, February-March 2026)

**Goal**: Enable real-time data ingestion, WebSocket streaming, and <100ms p95 latency.

### Week 7: WebSocket Streaming

**Tasks**:
- [ ] Implement WebSocket endpoint: /ws for real-time updates
- [ ] Create pub/sub system with Redis: publish accident updates, dashboard notifications
- [ ] Stream ML predictions in real-time: send prediction results as they're computed
- [ ] Implement WebSocket authentication: validate JWT on connection
- [ ] Handle WebSocket reconnection: automatic retry with exponential backoff
- [ ] Create WebSocket client library (JavaScript): auto-reconnect, event handlers
- [ ] Test with 100+ concurrent WebSocket connections

**Deliverables**:
- WebSocket endpoint for real-time updates
- Redis pub/sub for event broadcasting
- WebSocket client library

**Success Metrics**:
- WebSocket latency < 50ms (p95)
- Support 1000+ concurrent connections
- Reconnection success rate > 99%

**Research Finding**: Real-time API patterns (2024) - WebSocket is ideal for bidirectional communication (client ↔ server). Use Redis pub/sub to broadcast events to multiple WebSocket connections. Implement heartbeat ping/pong to detect dead connections.

**Code Example**:
```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import redis.asyncio as aioredis

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.redis = aioredis.from_url("redis://redis:6379")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            await connection.send_text(message)

    async def listen_for_updates(self):
        """Subscribe to Redis pub/sub"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("accident_updates", "ml_predictions")

        async for message in pubsub.listen():
            if message['type'] == 'message':
                await self.broadcast(message['data'])

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()

            # Process (e.g., query, prediction request)
            response = process_websocket_request(data)

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task to listen for Redis pub/sub
@app.on_event("startup")
async def startup():
    asyncio.create_task(manager.listen_for_updates())
```

```javascript
// JavaScript WebSocket client
class NTSBWebSocketClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.ws = null;
    this.reconnectDelay = 1000;  // Start with 1 second
  }

  connect() {
    this.ws = new WebSocket(`wss://api.ntsb-analytics.com/ws?token=${this.apiKey}`);

    this.ws.onopen = () => {
      console.log('Connected to NTSB WebSocket');
      this.reconnectDelay = 1000;  // Reset delay
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed, reconnecting...');
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);  // Max 30s
    };
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  handleMessage(data) {
    // Override this method
    console.log('Received:', data);
  }
}
```

**Dependencies**: fastapi, redis, websockets

### Week 8: Apache Kafka Event Streaming

**Tasks**:
- [ ] Set up Apache Kafka cluster (3 brokers for HA)
- [ ] Create topics: accident_events, ml_predictions, rag_queries
- [ ] Implement Kafka producer: publish events from API/ML services
- [ ] Implement Kafka consumer: process events for dashboard updates, analytics
- [ ] Configure Kafka Connect: stream NTSB monthly updates to PostgreSQL
- [ ] Set retention: 7 days for events, 30 days for predictions
- [ ] Monitor Kafka: lag, throughput, consumer group status

**Deliverables**:
- Kafka cluster operational
- 3+ topics with producers/consumers
- Kafka Connect for NTSB data ingestion

**Success Metrics**:
- Event throughput: 10K+ events/second
- End-to-end latency: <100ms (publish → consume)
- Zero message loss (acks=all)

**Code Example**:
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer
producer = KafkaProducer(
    bootstrap_servers=['kafka-1:9092', 'kafka-2:9092', 'kafka-3:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',  # Wait for all replicas
    retries=3
)

def publish_prediction_event(ev_id, prediction):
    """Publish ML prediction to Kafka"""
    event = {
        'ev_id': ev_id,
        'prediction': prediction,
        'timestamp': datetime.utcnow().isoformat()
    }

    producer.send('ml_predictions', key=ev_id.encode('utf-8'), value=event)
    producer.flush()

# Consumer
consumer = KafkaConsumer(
    'ml_predictions',
    bootstrap_servers=['kafka-1:9092', 'kafka-2:9092', 'kafka-3:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='dashboard-consumer'
)

for message in consumer:
    event = message.value
    print(f"Received prediction: {event['ev_id']} -> {event['prediction']}")

    # Update dashboard in real-time
    update_dashboard(event)

    # Publish to WebSocket clients
    await manager.broadcast(json.dumps(event))
```

**Dependencies**: kafka-python, confluent-kafka

### Week 9: Real-time Dashboard & Monitoring

**Tasks**:
- [ ] Integrate WebSocket with Streamlit dashboard (st.websocket or custom)
- [ ] Create real-time accident map: updates every 5 minutes
- [ ] Create live prediction feed: stream ML predictions
- [ ] Add real-time metrics: API requests/sec, active users, error rate
- [ ] Implement server-sent events (SSE) as WebSocket alternative
- [ ] Optimize for mobile: responsive design, reduced bandwidth
- [ ] Load test: 1000 concurrent users with real-time updates

**Deliverables**:
- Real-time Streamlit dashboard
- Live accident map with auto-updates
- Real-time metrics panel

**Success Metrics**:
- Dashboard update latency: <500ms
- Support 1000+ concurrent users
- Mobile-optimized (< 1MB data transfer/minute)

**Sprint 3 Total Hours**: 75 hours

---

## Sprint 4: Monitoring & Disaster Recovery (Weeks 10-12, March 2026)

**Goal**: Comprehensive monitoring and disaster recovery with <1hr RTO, <15min RPO.

### Week 10: Prometheus & Grafana

**Tasks**:
- [ ] Install Prometheus for metrics collection
- [ ] Configure Prometheus: scrape intervals, retention (15 days)
- [ ] Instrument services: export custom metrics (requests/sec, errors, latency)
- [ ] Install Grafana for visualization
- [ ] Create dashboards: API metrics, ML performance, infrastructure health
- [ ] Set up alerts: CPU > 80%, error rate > 1%, latency > 500ms
- [ ] Integrate with PagerDuty or Slack for incident management

**Deliverables**:
- Prometheus + Grafana stack operational
- 5+ dashboards (API, ML, infrastructure, business metrics)
- Alerting rules with PagerDuty/Slack integration

**Success Metrics**:
- Metrics scrape interval: 15 seconds
- Dashboard load time: <2 seconds
- Alert latency: <1 minute (detection → notification)

**Code Example**:
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_latency = Histogram('api_request_latency_seconds', 'API request latency', ['endpoint'])
active_users = Gauge('active_users', 'Currently active users')

# Middleware to track metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time
    request_count.labels(request.method, request.url.path, response.status_code).inc()
    request_latency.labels(request.url.path).observe(latency)

    return response

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

```yaml
# Prometheus configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'ml-serving'
    static_configs:
      - targets: ['ml-serving:8001']

  - job_name: 'kubernetes'
    kubernetes_sd_configs:
      - role: pod

# Alert rules
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: api_request_latency_seconds{quantile="0.95"} > 0.5
        for: 5m
        annotations:
          summary: "API latency > 500ms"
```

**Dependencies**: prometheus-client, prometheus, grafana

### Week 11: Disaster Recovery & Backups

**Tasks**:
- [ ] Implement automated backups: PostgreSQL (pg_dump), Neo4j, Redis
- [ ] Store backups in S3/GCS: encrypted, versioned, 30-day retention
- [ ] Schedule backups: hourly (incremental), daily (full)
- [ ] Test backup restoration: restore to staging environment
- [ ] Implement database replication: PostgreSQL streaming replication (master-replica)
- [ ] Create runbooks: incident response, rollback procedures
- [ ] Define RTO/RPO: RTO <1 hour, RPO <15 minutes

**Deliverables**:
- Automated backup system (hourly + daily)
- Tested disaster recovery procedures
- Incident response runbooks

**Success Metrics**:
- Backup success rate: 100%
- Restoration test: <30 minutes
- RTO: <1 hour, RPO: <15 minutes

**Code Example**:
```bash
#!/bin/bash
# Automated PostgreSQL backup script

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgresql"
BACKUP_FILE="$BACKUP_DIR/ntsb_$TIMESTAMP.sql.gz"

# Create backup
pg_dump -h postgres -U ntsb_user -d ntsb | gzip > $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE s3://ntsb-backups/postgresql/ --server-side-encryption AES256

# Delete local backup older than 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup successful: $BACKUP_FILE"
else
    echo "Backup failed!" | mail -s "Backup Alert" admin@ntsb-analytics.com
fi
```

```yaml
# Kubernetes CronJob for backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgresql-backup
spec:
  schedule: "0 * * * *"  # Every hour
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: pg-backup
            image: postgres:15
            command: ["/backup.sh"]
            volumeMounts:
            - name: backup-script
              mountPath: /backup.sh
              subPath: backup.sh
          restartPolicy: OnFailure
          volumes:
          - name: backup-script
            configMap:
              name: backup-script
              defaultMode: 0755
```

**Dependencies**: postgresql, awscli, gcloud, azure-cli

### Week 12: Load Testing & Beta Launch

**Tasks**:
- [ ] Load test with k6: simulate 10K concurrent users
- [ ] Identify bottlenecks: database queries, API endpoints, ML serving
- [ ] Optimize performance: caching, query optimization, connection pooling
- [ ] Conduct security audit: OWASP Top 10, penetration testing
- [ ] Prepare beta launch: select 50-100 beta users
- [ ] Create beta feedback form: bugs, feature requests, UX improvements
- [ ] Document known issues and limitations
- [ ] Plan for GA launch: marketing, press release, community outreach

**Deliverables**:
- Load test report (10K users, performance metrics)
- Security audit report
- Beta launch with 50-100 users

**Success Metrics**:
- Load test: 10K concurrent users, <200ms latency (p95)
- Zero critical security vulnerabilities
- Beta user feedback: >80% satisfaction

**Code Example**:
```javascript
// k6 load test script
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 1000 },  // Ramp up to 1000 users
    { duration: '10m', target: 10000 }, // Ramp up to 10K users
    { duration: '5m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],  // 95% requests < 200ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  // Test API endpoints
  let res = http.get('https://api.ntsb-analytics.com/events?limit=50');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });

  // Simulate user behavior
  sleep(1);

  // Test ML prediction
  let features = {
    aircraft_age: 25,
    pilot_hours: 500,
    weather_condition: 'IMC',
    // ... 100+ features
  };

  res = http.post('https://api.ntsb-analytics.com/ml/predict',
                   JSON.stringify(features),
                   { headers: { 'Content-Type': 'application/json' } });

  check(res, {
    'prediction status is 200': (r) => r.status === 200,
    'prediction time < 200ms': (r) => r.timings.duration < 200,
  });

  sleep(2);
}
```

**Dependencies**: k6, owasp-zap, burp-suite

**Sprint 4 Total Hours**: 70 hours

---

## Phase 5 Deliverables Summary

1. **Kubernetes Deployment**: Helm charts, HPA (2-10 pods), 99.9% uptime
2. **Public API**: JWT + OAuth2 auth, rate limiting (100-10K req/day), SDKs (Python, JS, R)
3. **Real-time**: WebSocket streaming, Kafka event streaming (<100ms latency)
4. **Monitoring**: Prometheus + Grafana dashboards, PagerDuty alerts
5. **Disaster Recovery**: Automated backups (hourly), tested restoration (RTO <1hr, RPO <15min)
6. **Beta Launch**: 50-100 users, load tested (10K concurrent users)

## Testing Checklist

- [ ] Kubernetes cluster deployed successfully
- [ ] HPA scales correctly under load (70% CPU trigger)
- [ ] API authentication and authorization working (JWT + OAuth2)
- [ ] Rate limiting enforced (100% success)
- [ ] SDKs published to package managers (PyPI, npm, CRAN)
- [ ] WebSocket supports 1000+ concurrent connections
- [ ] Kafka processes 10K+ events/second
- [ ] Prometheus scraping metrics from all services
- [ ] Grafana dashboards operational (5+ dashboards)
- [ ] Backup restoration tested successfully (<30 min)
- [ ] Load test passes (10K users, <200ms p95 latency)
- [ ] Security audit complete (0 critical vulnerabilities)

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Uptime | 99.9% | Prometheus uptime tracking (8.76 hrs downtime/year max) |
| API latency (p95) | <200ms | Prometheus histogram metrics |
| WebSocket latency (p95) | <50ms | Custom metrics |
| Kafka throughput | 10K+ events/sec | Kafka metrics |
| HPA scale-up time | <2 minutes | Manual testing |
| Backup success rate | 100% | CronJob logs |
| RTO | <1 hour | Disaster recovery drills |
| RPO | <15 minutes | Backup frequency |
| Load test concurrency | 10K users | k6 results |
| Security vulnerabilities | 0 critical | OWASP ZAP scan |

## Resource Requirements

**Infrastructure**:
- Kubernetes cluster: 10-20 nodes (AWS EKS, GCP GKE, Azure AKS)
- PostgreSQL: 100GB storage, 8GB RAM, 4 CPUs
- Neo4j: 50GB storage, 16GB RAM, 4 CPUs
- Redis: 8GB RAM (caching + pub/sub)
- Kafka: 3 brokers, 100GB storage each
- Prometheus: 50GB storage (15 days retention)
- Grafana: 2GB RAM

**Cloud Costs** (estimated):
- Kubernetes: $500-1000/month (10-20 nodes)
- Databases: $200-400/month (managed PostgreSQL, Neo4j, Redis)
- Load balancer: $20-50/month
- Storage (S3/GCS): $50-100/month (backups, logs)
- Kafka: $200-400/month (managed Kafka or self-hosted)
- **Total**: $1000-2000/month

**External Services**:
- LLM APIs (Claude/GPT-4): $100-500/month
- PagerDuty: $0-50/month (free tier or basic plan)
- Monitoring (optional): DataDog, New Relic ($100-500/month)

**Total Estimated Budget**: $1500-3000/month

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Kubernetes deployment failures | Medium | High | Use Helm, test in staging, blue-green deployments |
| API rate limiting bypasses | Low | High | Thorough testing, security audit, Redis transaction locks |
| WebSocket connection storms | Medium | Medium | Connection limits, rate limiting, auto-scaling |
| Backup failures | Low | Critical | Monitor backup success, test restorations monthly |
| Load test reveals bottlenecks | High | Medium | Identify early, optimize before GA launch |
| Security vulnerabilities | Medium | Critical | Regular audits, dependency scanning, penetration testing |

## Dependencies on Phase 1-4

- All services containerized (Dockerfiles)
- PostgreSQL, Neo4j, Redis operational
- ML models deployed (Phase 3)
- RAG system functional (Phase 4)
- Monitoring instrumentation in code

## Post-Launch Roadmap

1. **Month 1**: Monitor beta users, fix bugs, gather feedback
2. **Month 2**: GA launch, marketing, press release, partnerships
3. **Month 3**: Scale to 1000+ users, optimize costs, add premium features
4. **Year 1**: Achieve 10K+ users, 3+ research publications, revenue $50K+

---

**Last Updated**: January 2025
**Version**: 1.0
