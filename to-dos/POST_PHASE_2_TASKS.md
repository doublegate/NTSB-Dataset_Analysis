# Post-Phase 2 Tasks & Phase 3 Preparation

**Created**: 2025-11-09
**Status**: 🔄 IN PROGRESS
**Priority**: High (prepare for Phase 3 launch)

---

## Overview

This TODO tracks immediate tasks following Phase 2 completion (v3.0.0) and preparation for Phase 3 (Advanced Machine Learning). Tasks are organized by priority and estimated effort.

---

## PRIORITY 1: Critical Fixes & Documentation (Week 1)

### 1.1 Fix Notebook Schema References ⚠️ CRITICAL

**Issue**: Exploratory notebooks reference `acft_damage` column that doesn't exist in current schema.

**Affected Notebooks**:
- `notebooks/exploratory/01_exploratory_data_analysis.ipynb`
- `notebooks/exploratory/02_temporal_trends_analysis.ipynb`
- `notebooks/exploratory/03_aircraft_safety_analysis.ipynb`
- `notebooks/exploratory/04_cause_factor_analysis.ipynb`

**Schema Fix Required**:
```python
# OLD (INCORRECT)
SELECT ev_id, acft_damage FROM events

# NEW (CORRECT)
SELECT e.ev_id, a.damage AS acft_damage
FROM events e
JOIN aircraft a ON e.ev_id = a.ev_id
```

**Tasks**:
- [ ] Read each notebook to identify all `acft_damage` references
- [ ] Replace with JOIN to `aircraft` table
- [ ] Update column references: `events.acft_damage` → `aircraft.damage`
- [ ] Test queries with LIMIT 100 first
- [ ] Re-execute notebooks with fixes
- [ ] Verify outputs and visualizations

**Estimated Effort**: 4-6 hours (1.5 hours per notebook)

**Acceptance Criteria**:
- ✅ All 4 exploratory notebooks execute without errors
- ✅ All queries return expected row counts
- ✅ All visualizations render correctly
- ✅ Generated `_executed` versions with outputs

---

### 1.2 Verify Geospatial Notebooks Schema Compatibility

**Objective**: Ensure geospatial notebooks work with current schema.

**Notebooks to Check**:
- `notebooks/geospatial/01_geospatial_exploratory_analysis.ipynb`
- `notebooks/geospatial/02_clustering_analysis.ipynb`
- `notebooks/geospatial/03_kde_heatmaps.ipynb`
- `notebooks/geospatial/04_spatial_autocorrelation.ipynb`

**Verification Steps**:
- [ ] Check coordinate column references (`dec_latitude`, `dec_longitude`)
- [ ] Verify PostGIS queries (`ST_MakePoint`, `ST_Distance`)
- [ ] Validate spatial index usage
- [ ] Test with current database state (77,495 geocoded events)
- [ ] Execute notebooks to verify outputs

**Estimated Effort**: 2-3 hours

**Acceptance Criteria**:
- ✅ All geospatial queries execute successfully
- ✅ Spatial indexes used (verify with EXPLAIN ANALYZE)
- ✅ Visualizations render correctly (maps, heatmaps, clusters)
- ✅ Performance acceptable (<10 seconds per spatial query)

---

### 1.3 Document Notebook Dependencies & Execution Order

**Objective**: Create comprehensive execution guide for reproducibility.

**File to Create**: `notebooks/README.md`

**Required Content**:

#### 1. Environment Setup
- Python version (3.13)
- Virtual environment activation
- Required packages (requirements.txt)
- Database connection setup

#### 2. Execution Order
```
Phase A: Database Setup (prerequisite)
├── scripts/setup_database.sh
├── scripts/load_with_staging.py --source avall.mdb
└── scripts/optimize_queries.sql

Phase B: Exploratory Analysis (Sprint 1-2)
├── 01_exploratory_data_analysis.ipynb (30 min)
├── 02_temporal_trends_analysis.ipynb (20 min)
├── 03_aircraft_safety_analysis.ipynb (25 min)
└── 04_cause_factor_analysis.ipynb (20 min)

Phase C: Geospatial Analysis (Sprint 8)
├── 01_geospatial_exploratory_analysis.ipynb (15 min)
├── 02_clustering_analysis.ipynb (45 min, heavy compute)
├── 03_kde_heatmaps.ipynb (30 min)
└── 04_spatial_autocorrelation.ipynb (40 min)

Phase D: Machine Learning (Sprint 6-7)
├── 01_feature_engineering.ipynb (20 min)
├── 02_logistic_regression.ipynb (30 min)
├── 03_random_forest.ipynb (60 min, heavy compute)
└── 04_model_evaluation.ipynb (15 min)

Phase E: NLP Text Mining (Sprint 9-10)
├── 01_text_preprocessing.ipynb (45 min)
├── 02_tfidf_analysis.ipynb (25 min)
├── 03_lda_topic_modeling.ipynb (90 min, heavy compute)
└── 04_word2vec_embeddings.ipynb (60 min)
```

#### 3. Data Dependencies
- Database: ntsb_aviation (179,809 events)
- Required tables: events, aircraft, flight_crew, findings, narratives
- Materialized views: mv_yearly_stats, mv_state_stats, etc.
- Geospatial data: 77,495 geocoded events

#### 4. Expected Outputs
- Figures directory: `notebooks/{category}/figures/`
- Models directory: `models/*.pkl`
- Reports directory: `notebooks/reports/`

#### 5. Troubleshooting
- Common errors and solutions
- Performance optimization tips
- Database connection issues

**Tasks**:
- [ ] Create notebooks/README.md
- [ ] Document all dependencies
- [ ] List execution order with timing estimates
- [ ] Add troubleshooting section
- [ ] Test instructions with clean environment

**Estimated Effort**: 2-3 hours

**Acceptance Criteria**:
- ✅ Complete execution guide created
- ✅ All dependencies documented
- ✅ Execution order clearly defined
- ✅ Troubleshooting section comprehensive

---

## PRIORITY 2: Optional Documentation (Week 2)

### 2.1 Create Sprint 3-4 API Summary Report

**Status**: OPTIONAL (Low Priority)
**Rationale**: API already self-documenting via OpenAPI/Swagger

**File to Create**: `notebooks/reports/sprint_3_4_api_foundation_summary.md`

**Suggested Content** (if created):

```markdown
# Sprint 3-4: REST API Foundation Summary

## Overview
- FastAPI framework selection rationale
- PostgreSQL integration via psycopg2
- 21 endpoints across 5 categories
- OpenAPI documentation auto-generated

## Endpoints Implemented

### 1. Events API (5 endpoints)
- GET /events - List events with filters
- GET /events/{ev_id} - Event details
- GET /events/search - Full-text search
- GET /events/stats - Statistics
- GET /events/timeline - Temporal aggregation

### 2. Geographic API (4 endpoints)
- GET /geographic/states - State statistics
- GET /geographic/coordinates - Geocoded events
- GET /geographic/hotspots - Spatial clusters
- GET /geographic/heatmap - Density data

### 3. Aircraft API (4 endpoints)
- GET /aircraft - Aircraft list
- GET /aircraft/{aircraft_key} - Details
- GET /aircraft/makes - Makes statistics
- GET /aircraft/models - Models statistics

### 4. Crew API (4 endpoints)
- GET /crew - Crew records
- GET /crew/certifications - Cert analysis
- GET /crew/experience - Experience stats
- GET /crew/age_analysis - Age distributions

### 5. Findings API (4 endpoints)
- GET /findings - Investigation findings
- GET /findings/causes - Top causes
- GET /findings/by_code - Code analysis
- GET /findings/fatal_rates - Fatal rates by finding

## Performance Benchmarks
- p50 latency: <50ms
- p95 latency: <200ms
- p99 latency: <500ms
- Throughput: 100+ requests/second

## Deployment
- Development: Uvicorn (localhost:8000)
- Production: Docker + systemd service
- Documentation: /docs (Swagger UI)
```

**Tasks**:
- [ ] Synthesize from api/README.md and api/ code
- [ ] Add performance benchmarks
- [ ] Document deployment procedures
- [ ] Include example requests/responses
- [ ] Cross-reference with comprehensive draft

**Estimated Effort**: 2-3 hours

**Acceptance Criteria**:
- ✅ Complete API documentation (500-800 lines)
- ✅ All 21 endpoints documented
- ✅ Performance metrics included
- ✅ Deployment guide comprehensive

**Decision**: Defer until user requests or Phase 3 planning requires it

---

### 2.2 Enhance notebooks/reports/README.md

**Status**: ✅ COMPLETED (Sub-agent created 184-line index)

**Current State**:
- Index file exists at `notebooks/reports/README.md`
- Links to all 5 sprint reports
- Coverage summary table
- Key findings section

**Optional Enhancements** (if time permits):
- [ ] Add direct links to specific sections in reports
- [ ] Create visual timeline of sprints
- [ ] Add "Quick Start" guide for researchers
- [ ] Include citation information for academic use

**Estimated Effort**: 1 hour

---

## PRIORITY 3: Phase 3 Planning (Week 3)

### 3.1 Create Detailed Phase 3 Sprint Breakdown

**Objective**: Plan 12-week Advanced Machine Learning phase (4 sprints × 3 weeks).

**File to Update**: `to-dos/PHASE_3_MACHINE_LEARNING.md`

**Required Content**:

#### Sprint 1: Advanced Feature Engineering (Weeks 1-3)
- Week 1: Text embeddings from narratives (BERT, Word2Vec)
- Week 2: Weather feature engineering (visibility, IMC, wind)
- Week 3: Interaction features (pilot experience × weather)

#### Sprint 2: XGBoost Development (Weeks 4-6)
- Week 1: Model architecture and hyperparameter tuning
- Week 2: Cross-validation and performance benchmarking
- Week 3: Feature importance analysis

#### Sprint 3: Model Explainability (SHAP) (Weeks 7-9)
- Week 1: SHAP integration and global explanations
- Week 2: Local explanations and interactive dashboards
- Week 3: Documentation and interpretation guides

#### Sprint 4: MLflow & Deployment (Weeks 10-12)
- Week 1: MLflow experiment tracking setup
- Week 2: Model serving API development
- Week 3: Automated retraining pipeline

**Tasks**:
- [ ] Read existing PHASE_3_MACHINE_LEARNING.md
- [ ] Create detailed sprint breakdown (4 sprints)
- [ ] Define success metrics for each sprint
- [ ] Estimate effort and resources
- [ ] Identify dependencies and blockers
- [ ] Create Sprint 1 Week 1 task list

**Estimated Effort**: 4-6 hours

**Acceptance Criteria**:
- ✅ Complete 12-week plan documented
- ✅ Each week has specific deliverables
- ✅ Success metrics defined (90%+ accuracy target)
- ✅ Dependencies identified
- ✅ Sprint 1 ready to start

---

### 3.2 Set Up Development Environment for Phase 3

**Objective**: Prepare tools and infrastructure for advanced ML.

**Tools to Install/Configure**:

#### Python Packages
- [ ] XGBoost: `pip install xgboost`
- [ ] SHAP: `pip install shap`
- [ ] MLflow: `pip install mlflow`
- [ ] Optuna: `pip install optuna` (hyperparameter tuning)
- [ ] Transformers: `pip install transformers` (BERT embeddings)

#### Development Tools
- [ ] MLflow tracking server setup
- [ ] Model registry configuration
- [ ] Experiment logging infrastructure
- [ ] Performance monitoring setup

#### Database Enhancements
- [ ] Create ml_experiments table for tracking
- [ ] Create ml_models table for registry
- [ ] Add indexes for feature queries
- [ ] Optimize feature extraction queries

**Tasks**:
- [ ] Install all Python packages in .venv
- [ ] Test XGBoost installation
- [ ] Set up MLflow tracking server
- [ ] Create database tables for ML tracking
- [ ] Test end-to-end ML pipeline

**Estimated Effort**: 3-4 hours

**Acceptance Criteria**:
- ✅ All packages installed and tested
- ✅ MLflow UI accessible (localhost:5000)
- ✅ Database tables created
- ✅ Sample experiment logged successfully

---

## PRIORITY 4: Security & Maintenance (Week 4)

### 4.1 Address GitHub Dependabot Security Alerts

**Current Alerts**:
- 1 Critical vulnerability
- 2 High vulnerabilities
- 2 Moderate vulnerabilities

**Tasks**:
- [ ] Review Dependabot alert details at:
  `https://github.com/doublegate/NTSB-Dataset_Analysis/security/dependabot`
- [ ] Identify affected packages and versions
- [ ] Check for available security patches
- [ ] Update vulnerable dependencies
- [ ] Test application after updates
- [ ] Verify no breaking changes introduced
- [ ] Re-run security audit: `pip-audit`

**Estimated Effort**: 2-4 hours (depends on breaking changes)

**Acceptance Criteria**:
- ✅ All critical/high vulnerabilities resolved
- ✅ Moderate vulnerabilities assessed (fix or accept risk)
- ✅ All tests passing after updates
- ✅ requirements.txt updated
- ✅ CHANGELOG.md documents security fixes

---

### 4.2 Run Comprehensive Security Audit

**Objective**: Identify and fix additional security issues.

**Tools to Use**:
- [ ] pip-audit: `pip install pip-audit && pip-audit`
- [ ] bandit: `pip install bandit && bandit -r .`
- [ ] safety: `pip install safety && safety check`
- [ ] GitHub Code Scanning (automated)

**Tasks**:
- [ ] Run pip-audit on all dependencies
- [ ] Run bandit security linter on Python code
- [ ] Review API security (authentication, rate limiting)
- [ ] Check database security (SQL injection prevention)
- [ ] Verify secrets not committed (.env, credentials)
- [ ] Document security best practices

**Estimated Effort**: 3-4 hours

**Acceptance Criteria**:
- ✅ No high/critical vulnerabilities remaining
- ✅ Security audit report created
- ✅ Best practices documented
- ✅ Automated security checks added to CI/CD

---

## PRIORITY 5: Optional Quick Wins (Ongoing)

### 5.1 Deploy Dashboard to Streamlit Cloud

**Objective**: Make interactive dashboard publicly accessible.

**Tasks**:
- [ ] Create Streamlit Cloud account
- [ ] Configure app settings (requirements.txt, secrets)
- [ ] Test database connection from cloud
- [ ] Deploy app to Streamlit Cloud
- [ ] Configure custom domain (optional)
- [ ] Monitor performance and usage
- [ ] Document deployment in README

**Estimated Effort**: 2-3 hours

**Benefits**:
- Public demo for stakeholders
- Portfolio showcase
- User feedback collection

**URL**: `https://<app-name>.streamlit.app`

---

### 5.2 Publish API Documentation to GitHub Pages

**Objective**: Host OpenAPI spec as static site.

**Tasks**:
- [ ] Generate OpenAPI JSON: `curl http://localhost:8000/openapi.json > docs/openapi.json`
- [ ] Create Swagger UI HTML page
- [ ] Configure GitHub Pages in repo settings
- [ ] Deploy to `https://<username>.github.io/<repo>/api-docs`
- [ ] Add link to README.md

**Estimated Effort**: 1-2 hours

**Benefits**:
- Easier API discovery
- Better developer experience
- Professional documentation

---

### 5.3 Investigate Random Forest UNKNOWN Finding Codes

**Issue**: 75% of finding codes classified as UNKNOWN by Random Forest model.

**Hypothesis**: Feature engineering or class imbalance issue.

**Tasks**:
- [ ] Analyze finding code distribution (class imbalance)
- [ ] Review feature importance (are finding codes used?)
- [ ] Check model training data (sufficient examples per code?)
- [ ] Test with SMOTE oversampling for minority classes
- [ ] Re-train with balanced dataset
- [ ] Compare performance metrics

**Estimated Effort**: 4-6 hours

**Expected Outcome**: Reduce UNKNOWN rate from 75% to <30%

---

## PRIORITY 6: Data Quality Improvements (Phase 3 Prep)

### 6.1 Integrate PRE1982.MDB Data (1962-1981)

**Status**: Deferred from Sprint 4

**Objective**: Add 20 years of historical data to increase coverage.

**Complexity**: High (incompatible schema, denormalized structure)

**Tasks**:
- [ ] Review PRE1982_ANALYSIS.md for schema differences
- [ ] Design ETL transformation pipeline
- [ ] Create code mapping tables (200+ coded fields)
- [ ] Implement denormalized → normalized transformation
- [ ] Test with sample data (1000 events)
- [ ] Load full dataset (~87,000 events estimated)
- [ ] Validate data quality
- [ ] Refresh materialized views

**Estimated Effort**: 8-16 hours

**Benefits**:
- Complete 64-year coverage (1962-2025)
- Richer historical trend analysis
- Better long-term forecasting

---

### 6.2 Reduce Missing Coordinates (56.7% → <30%)

**Current State**: 102,314 events missing coordinates (56.7%)

**Approach**: Geocoding pipeline using external APIs.

**Tasks**:
- [ ] Identify events with city/state but no coordinates
- [ ] Set up geocoding service (Nominatim, Google Maps API)
- [ ] Batch geocode missing locations (rate-limited)
- [ ] Validate geocoded coordinates (bounds checking)
- [ ] Update database with new coordinates
- [ ] Refresh geospatial analyses

**Estimated Effort**: 6-10 hours (mostly API wait time)

**Benefits**:
- 77,495 → ~120,000 geocoded events (+55%)
- Better spatial clustering and hotspot analysis
- Improved geographic visualizations

---

## Success Metrics

### Phase 2 Completion (✅ Achieved)
- ✅ 100% of planned deliverables completed
- ✅ 8,204 lines of comprehensive documentation
- ✅ All code quality checks passing
- ✅ Production-ready API and dashboard

### Post-Phase 2 Tasks (Target: Week 4)
- ✅ All notebooks executable (4 exploratory + 4 geospatial)
- ✅ Complete execution documentation
- ✅ Security vulnerabilities resolved
- ✅ Phase 3 sprint plan detailed

### Phase 3 Readiness (Target: Week 5)
- ✅ Development environment set up (XGBoost, SHAP, MLflow)
- ✅ Sprint 1 task list ready
- ✅ ML tracking infrastructure operational
- ✅ Data quality improvements underway

---

## Timeline Summary

| Week | Priority | Tasks | Estimated Effort |
|------|----------|-------|------------------|
| Week 1 | P1: Critical Fixes | Notebook schema fixes, geospatial validation, execution docs | 8-12 hours |
| Week 2 | P2: Optional Docs | Sprint 3-4 summary (optional), README enhancements | 3-4 hours |
| Week 3 | P3: Phase 3 Planning | Sprint breakdown, environment setup | 7-10 hours |
| Week 4 | P4: Security | Dependabot alerts, security audit | 5-8 hours |
| Ongoing | P5: Quick Wins | Dashboard deployment, API docs, model fixes | 7-11 hours |

**Total Estimated Effort**: 30-45 hours (1-2 weeks full-time, 3-4 weeks part-time)

---

## Decision Log

**2025-11-09**:
- ✅ Created POST_PHASE_2_TASKS.md (this file)
- ✅ Prioritized notebook fixes as P1 (critical for reproducibility)
- ⏭️ Deferred Sprint 3-4 API summary (low priority, self-documenting)
- ✅ Scheduled Phase 3 planning for Week 3 (allows focus on fixes first)

---

## Notes

- All tasks assume `.venv` virtual environment activated
- Database backups recommended before data quality improvements
- Security fixes may require dependency updates (test thoroughly)
- Phase 3 can start while P4-P6 tasks continue in parallel

---

**Last Updated**: 2025-11-09
**Next Review**: Week 1 completion (after notebook fixes)
