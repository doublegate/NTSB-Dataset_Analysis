# PHASE 3: MACHINE LEARNING

Production ML models for accident severity prediction, feature engineering, explainability, and model serving.

**Timeline**: Q3 2025 (12 weeks, July-September 2025)
**Prerequisites**: Phase 1-2 complete, clean data, PostgreSQL operational
**Team**: 2-3 developers (ML engineer + MLOps specialist)
**Estimated Hours**: ~320 hours total

## Overview

| Sprint | Duration | Focus Area | Key Deliverables | Hours |
|--------|----------|------------|------------------|-------|
| Sprint 1 | Weeks 1-3 | Feature Engineering | 100+ engineered features | 85h |
| Sprint 2 | Weeks 4-6 | Model Development | XGBoost, RF, Neural Networks | 80h |
| Sprint 3 | Weeks 7-9 | Tuning & Explainability | SHAP, Optuna, Cross-validation | 80h |
| Sprint 4 | Weeks 10-12 | Serving & Monitoring | MLflow, FastAPI, Evidently | 75h |

## Sprint 1: Feature Engineering (Weeks 1-3, July 2025)

**Goal**: Create 100+ high-quality features from raw aviation accident data.

### Week 1: NTSB Code Feature Extraction

**Tasks**:
- [ ] Design AviationFeatureEngineer class with modular pipelines
- [ ] Extract occurrence codes: one-hot encode 100-430 range (ABRUPT MANEUVER, ENGINE FAILURE, etc.)
- [ ] Extract phase of operation codes: 500-610 (TAXI, TAKEOFF, CRUISE, LANDING, etc.)
- [ ] Extract cause codes: Section II (30000-84200) direct causes
- [ ] Extract contributing factor codes: Section III (90000-93300) indirect causes
- [ ] Create hierarchical features: top-level category + sub-category
- [ ] Validate against ref_docs/codman.pdf coding manual

**Deliverables**:
- NTSB code extraction pipeline (handles multi-code fields)
- 50+ binary features for occurrence/phase/cause codes
- Hierarchical code aggregations (e.g., all engine-related codes → engine_failure_group)

**Success Metrics**:
- Extract codes from 95%+ of accidents
- 0% invalid codes (validated against lexicon)
- Feature sparsity < 80% (most features have >20% non-zero values)

**Code Example**:
```python
class NTSBCodeExtractor:
    def __init__(self, code_lexicon_path):
        self.lexicon = pd.read_csv(code_lexicon_path)

    def extract_occurrence_codes(self, df):
        """Extract occurrence codes (100-430)"""
        occurrence_cols = []
        for code in range(100, 431, 10):  # 100, 110, 120, ...
            col_name = f'occurrence_{code}'
            df[col_name] = df['occurrence_code'].str.contains(str(code), na=False).astype(int)
            occurrence_cols.append(col_name)
        return df, occurrence_cols

    def extract_phase_codes(self, df):
        """Extract phase of operation codes (500-610)"""
        phase_mapping = {
            500: 'standing', 510: 'taxi', 520: 'takeoff',
            550: 'cruise', 580: 'approach', 600: 'landing'
        }
        for code, phase in phase_mapping.items():
            df[f'phase_{phase}'] = (df['phase_code'] == code).astype(int)
        return df

    def create_hierarchical_codes(self, df):
        """Group codes into higher-level categories"""
        # All engine-related codes (14000-17710)
        df['engine_related'] = df['cause_code'].between(14000, 17710).astype(int)

        # All weather-related codes
        df['weather_related'] = df['cause_code'].isin([22000, 22100, 22200]).astype(int)

        return df
```

**Dependencies**: pandas, numpy, scikit-learn

### Week 2: Temporal & Spatial Features

**Tasks**:
- [ ] Cyclical encoding for temporal features (month, day_of_week, hour)
- [ ] Calculate days since last accident at same airport
- [ ] Extract flight duration from event_time fields
- [ ] Create temporal trends: accidents_last_30_days, accidents_last_year
- [ ] Spatial lag features: K-nearest neighbors (K=5, 10, 20)
- [ ] Distance to nearest major airport (using Haversine formula)
- [ ] Regional risk scores: accident rate per FAA region

**Deliverables**:
- Cyclical temporal features (sin/cos transformations)
- 15+ spatial lag features
- Regional aggregations (FAA regions: AWP, ASW, ANE, etc.)

**Success Metrics**:
- Cyclical encoding preserves temporal patterns (Dec ≈ Jan)
- Spatial features correlate with accident severity (R > 0.3)
- Regional risk scores validated against known high-risk regions

**Research Finding**: Cyclical encoding is essential for temporal features to avoid artificial ordering (December should be close to January, not far apart). Use sin/cos transformations: sin(2π × month/12), cos(2π × month/12).

**Code Example**:
```python
import numpy as np
from sklearn.neighbors import BallTree

class TemporalSpatialFeatures:
    def create_cyclical_features(self, df):
        """Encode temporal features cyclically"""
        df['month_sin'] = np.sin(2 * np.pi * df['event_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['event_month'] / 12)

        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['hour_sin'] = np.sin(2 * np.pi * df['event_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['event_hour'] / 24)

        return df

    def create_spatial_lag_features(self, df, k=10):
        """K-nearest neighbors spatial lag"""
        coords = np.radians(df[['latitude', 'longitude']].values)

        # Build BallTree for fast nearest neighbor search
        tree = BallTree(coords, metric='haversine')

        # Query K nearest neighbors
        distances, indices = tree.query(coords, k=k+1)  # +1 to exclude self

        # Calculate spatial lag: average severity of K nearest neighbors
        for i in range(1, k+1):
            neighbor_severity = df['severity'].values[indices[:, i]]
            df[f'spatial_lag_k{i}'] = neighbor_severity

        # Average distance to K nearest neighbors
        df[f'avg_distance_k{k}'] = distances[:, 1:].mean(axis=1) * 6371  # km

        return df

    def create_airport_risk_scores(self, df, airport_stats):
        """Airport-level risk aggregation"""
        # Join with pre-computed airport statistics
        df = df.merge(airport_stats, on='nearest_airport', how='left')

        # Fill missing with global average
        df['airport_accident_rate'].fillna(df['airport_accident_rate'].mean(), inplace=True)

        return df
```

**Dependencies**: scikit-learn, numpy, geopy

### Week 3: Aircraft & Crew Features

**Tasks**:
- [ ] Aircraft age: calculate from year_manufactured
- [ ] Aircraft type encoding: categorical (Airplane, Helicopter, Glider, etc.)
- [ ] Engine type: Reciprocating, Turbo Fan, Turbo Prop, Electric
- [ ] Number of engines: 1, 2, 3, 4+
- [ ] Pilot experience: total_flight_hours, hours_in_aircraft_type
- [ ] Pilot certifications: ATP, Commercial, Private, Student
- [ ] Weather conditions: VMC (visual) vs IMC (instrument), wind speed, visibility
- [ ] Flight purpose: Commercial, Personal, Instructional, Ferry

**Deliverables**:
- 25+ aircraft/crew features
- Ordinal encoding for experience levels
- One-hot encoding for categorical variables

**Success Metrics**:
- Aircraft age available for 90%+ of records
- Pilot experience extracted for 85%+ of accidents
- Weather features extracted from narratives using NLP (if not structured)

**Code Example**:
```python
class AircraftCrewFeatures:
    def create_aircraft_features(self, df):
        """Aircraft characteristics"""
        df['aircraft_age'] = df['event_year'] - df['year_manufactured']
        df['aircraft_age'] = df['aircraft_age'].clip(lower=0, upper=100)  # Cap at 100

        # Encode aircraft category
        df = pd.get_dummies(df, columns=['aircraft_category'], prefix='cat')

        # Engine count feature engineering
        df['has_multiple_engines'] = (df['num_engines'] > 1).astype(int)
        df['engine_count_cat'] = pd.cut(df['num_engines'],
                                         bins=[0, 1, 2, 4, 100],
                                         labels=['single', 'twin', 'tri_quad', 'multi'])

        return df

    def create_pilot_features(self, df):
        """Pilot experience and certifications"""
        # Log-transform flight hours (right-skewed distribution)
        df['log_flight_hours'] = np.log1p(df['total_flight_hours'])

        # Recency: hours in last 90 days (if available)
        df['recent_experience_ratio'] = df['hours_last_90_days'] / (df['total_flight_hours'] + 1)

        # Certification hierarchy (ordinal)
        cert_mapping = {'Student': 1, 'Sport': 2, 'Private': 3,
                        'Commercial': 4, 'ATP': 5}
        df['cert_level'] = df['pilot_cert'].map(cert_mapping)

        return df

    def create_weather_features(self, df):
        """Weather conditions"""
        # VMC vs IMC
        df['is_imc'] = (df['weather_condition'] == 'IMC').astype(int)

        # Wind speed categories
        df['high_wind'] = (df['wind_speed_kts'] > 25).astype(int)

        # Visibility
        df['low_visibility'] = (df['visibility_miles'] < 3).astype(int)

        return df
```

**Dependencies**: pandas, scikit-learn

**Sprint 1 Total Hours**: 85 hours

---

## Sprint 2: Model Development (Weeks 4-6, August 2025)

**Goal**: Train and evaluate XGBoost, Random Forest, and Neural Network models for severity prediction.

### Week 4: Baseline Models & XGBoost

**Tasks**:
- [ ] Define target variable: severity (binary: fatal vs non-fatal or multi-class: none/minor/serious/fatal)
- [ ] Train-test split: 80/20, stratified by severity
- [ ] Train baseline models: Logistic Regression, Decision Tree
- [ ] Train XGBoost classifier with default hyperparameters
- [ ] Evaluate with accuracy, precision, recall, F1, ROC-AUC
- [ ] Feature importance analysis: top 20 features

**Deliverables**:
- Baseline model performance (target to beat)
- XGBoost model with 90%+ accuracy
- Feature importance report

**Success Metrics**:
- XGBoost accuracy > 90%
- ROC-AUC > 0.95
- Precision/Recall balanced (F1 > 0.88)

**Research Finding**: XGBoost vs LightGBM benchmarks (2024): XGBoost generally outperforms on smaller datasets (<100K samples) with better accuracy. LightGBM is 11-15x faster on large datasets (>1M samples) and more memory-efficient. For NTSB data (~100K records), XGBoost is the better choice for accuracy.

**Code Example**:
```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Prepare data
X = df[feature_cols]
y = df['severity_class']  # 0=non-fatal, 1=fatal

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=20,
          verbose=True)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(20))
```

**Dependencies**: xgboost, scikit-learn, pandas

### Week 5: Ensemble Methods

**Tasks**:
- [ ] Train Random Forest classifier (500 trees, max_depth=15)
- [ ] Train LightGBM as comparison (for speed benchmarking)
- [ ] Create ensemble: Voting Classifier (XGBoost + RF + LightGBM)
- [ ] Train stacking ensemble: meta-learner on base model predictions
- [ ] Compare ensemble vs individual models
- [ ] Select best model for production

**Deliverables**:
- Random Forest and LightGBM models
- Voting and stacking ensembles
- Model comparison report (accuracy, speed, memory)

**Success Metrics**:
- Ensemble accuracy > 91% (1% improvement over XGBoost)
- Inference time < 10ms per prediction
- Model size < 100MB

**Research Finding**: Ensemble methods typically improve accuracy by 1-3% over individual models. Voting ensembles are faster (parallel predictions), while stacking ensembles are more accurate but slower (sequential predictions).

**Code Example**:
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Voting Ensemble (soft voting for probability averaging)
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', model),
        ('rf', rf_model),
        ('lgb', lgb_model)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Stacking Ensemble
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', model),
        ('rf', rf_model),
        ('lgb', lgb_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train, y_train)

# Compare
models = {
    'XGBoost': model,
    'Random Forest': rf_model,
    'LightGBM': lgb_model,
    'Voting': voting_clf,
    'Stacking': stacking_clf
}

for name, mdl in models.items():
    y_pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
```

**Dependencies**: scikit-learn, lightgbm, xgboost

### Week 6: Neural Network (PyTorch)

**Tasks**:
- [ ] Design feed-forward neural network: 3 hidden layers (256, 128, 64 units)
- [ ] Implement in PyTorch with BatchNorm and Dropout (0.3)
- [ ] Use Adam optimizer, learning rate 0.001 with ReduceLROnPlateau
- [ ] Train for 50 epochs with early stopping
- [ ] Implement class weighting for imbalanced data
- [ ] Compare to XGBoost/ensemble

**Deliverables**:
- PyTorch neural network model
- Training curves (loss, accuracy)
- Model comparison: NN vs tree-based models

**Success Metrics**:
- NN accuracy > 89% (may not beat XGBoost, but provides alternative)
- Training converges within 30 epochs
- No overfitting (train/val accuracy gap < 5%)

**Code Example**:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AccidentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Prepare data
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize model
model = AccidentClassifier(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Training loop
for epoch in range(50):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Loss={val_loss:.4f}")
```

**Dependencies**: torch, numpy, scikit-learn

**Sprint 2 Total Hours**: 80 hours

---

## Sprint 3: Tuning & Explainability (Weeks 7-9, August-September 2025)

**Goal**: Optimize hyperparameters, implement spatial cross-validation, and integrate SHAP explainability.

### Week 7: Hyperparameter Optimization (Optuna)

**Tasks**:
- [ ] Install Optuna for automated hyperparameter tuning
- [ ] Define search space: XGBoost (n_estimators, max_depth, learning_rate, subsample, etc.)
- [ ] Run Optuna study: 100 trials with TPE sampler
- [ ] Use spatial cross-validation (not random split) to prevent data leakage
- [ ] Visualize optimization history and parameter importance
- [ ] Retrain best model on full training set

**Deliverables**:
- Optimized XGBoost model (2-5% accuracy improvement)
- Optuna study visualization
- Best hyperparameters report

**Success Metrics**:
- Tuned model accuracy > 91%
- ROC-AUC > 0.96
- Optuna converges within 100 trials

**Research Finding**: MLflow best practices (2024): Use MLflow for experiment tracking during hyperparameter tuning. Log all trials, parameters, and metrics to enable reproducibility and comparison.

**Code Example**:
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = xgb.XGBClassifier(**params, random_state=42)

    # Spatial cross-validation (custom splitter)
    cv_scores = cross_val_score(model, X_train, y_train,
                                  cv=SpatialKFold(n_splits=5),
                                  scoring='roc_auc',
                                  n_jobs=-1)

    return cv_scores.mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best ROC-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualizations
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

**Dependencies**: optuna, xgboost, scikit-learn

### Week 8: Spatial Cross-Validation

**Tasks**:
- [ ] Implement custom SpatialKFold class (group by geographic clusters)
- [ ] Use DBSCAN clusters from Phase 2 as spatial folds
- [ ] Ensure no spatial leakage: train/test folds geographically separated
- [ ] Compare to random K-fold: spatial CV should have lower performance (more realistic)
- [ ] Calculate confidence intervals for performance metrics
- [ ] Document spatial CV methodology

**Deliverables**:
- SpatialKFold cross-validation class
- Model performance with spatial CV (expected 2-3% lower than random CV)
- Confidence intervals for accuracy, ROC-AUC

**Success Metrics**:
- Spatial CV accuracy > 88% (may be lower than random CV, more realistic)
- Performance consistent across all 5 folds (std < 2%)
- No spatial leakage detected

**Code Example**:
```python
from sklearn.model_selection import BaseCrossValidator

class SpatialKFold(BaseCrossValidator):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Split based on spatial clusters to prevent leakage.
        groups: Array of cluster IDs from DBSCAN
        """
        unique_clusters = np.unique(groups)
        np.random.shuffle(unique_clusters)

        fold_size = len(unique_clusters) // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else len(unique_clusters)

            test_clusters = unique_clusters[start:end]
            test_idx = np.isin(groups, test_clusters)

            train_idx = ~test_idx

            yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Usage
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train, y_train,
    cv=SpatialKFold(n_splits=5),
    groups=df['spatial_cluster'],  # From Phase 2 DBSCAN
    scoring=['accuracy', 'roc_auc', 'f1'],
    return_train_score=True,
    n_jobs=-1
)

print(f"Test Accuracy: {cv_results['test_accuracy'].mean():.4f} ± "
      f"{cv_results['test_accuracy'].std():.4f}")
print(f"Test ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± "
      f"{cv_results['test_roc_auc'].std():.4f}")
```

**Dependencies**: scikit-learn, numpy

### Week 9: SHAP Explainability

**Tasks**:
- [ ] Install SHAP library for model interpretability
- [ ] Generate SHAP values for XGBoost model (TreeExplainer)
- [ ] Create SHAP summary plot: global feature importance
- [ ] Create SHAP dependence plots: partial dependence for top 10 features
- [ ] Generate individual prediction explanations (waterfall plots)
- [ ] Integrate SHAP into API: return SHAP values with predictions
- [ ] Create SHAP dashboard in Streamlit

**Deliverables**:
- SHAP values for all test predictions
- SHAP visualizations (summary, dependence, waterfall)
- API endpoint: POST /predict with SHAP explanations
- SHAP dashboard page in Streamlit

**Success Metrics**:
- SHAP values computed in <100ms per prediction
- Top 10 features explain 60%+ of model variance
- SHAP explanations validated by aviation experts (sanity check)

**Research Finding**: SHAP (SHapley Additive exPlanations) is the gold standard for model explainability in 2024-2025. TreeExplainer is optimized for tree-based models (XGBoost, RF, LightGBM) with exact SHAP values computed efficiently.

**Code Example**:
```python
import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Global feature importance (summary plot)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)

# Dependence plot for top feature
shap.dependence_plot('aircraft_age', shap_values, X_test, feature_names=feature_cols)

# Waterfall plot for single prediction
shap.waterfall_plot(explainer(X_test.iloc[0]))

# Force plot (alternative visualization)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0],
                feature_names=feature_cols)

# Integrate into API
from fastapi import APIRouter

@router.post("/predict/explain")
async def predict_with_shap(features: dict):
    X = pd.DataFrame([features])
    prediction = model.predict_proba(X)[0]

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)[0]

    # Top 10 contributing features
    feature_contributions = dict(zip(feature_cols, shap_values))
    top_features = sorted(feature_contributions.items(),
                          key=lambda x: abs(x[1]),
                          reverse=True)[:10]

    return {
        "prediction": {
            "non_fatal_prob": prediction[0],
            "fatal_prob": prediction[1]
        },
        "explanation": {
            "base_value": explainer.expected_value,
            "top_features": [
                {"feature": feat, "contribution": float(contrib)}
                for feat, contrib in top_features
            ]
        }
    }
```

**Dependencies**: shap, matplotlib, fastapi

**Sprint 3 Total Hours**: 80 hours

---

## Sprint 4: Serving & Monitoring (Weeks 10-12, September 2025)

**Goal**: Deploy ML models to production with MLflow, FastAPI serving, and Evidently AI monitoring.

### Week 10: MLflow Model Registry

**Tasks**:
- [ ] Set up MLflow tracking server (local or AWS/GCP)
- [ ] Log all experiments: models, parameters, metrics, artifacts
- [ ] Register best model in MLflow Model Registry
- [ ] Tag models with versions: v1.0, v1.1, etc.
- [ ] Set model stages: Staging, Production, Archived
- [ ] Document model lineage: feature engineering → training → evaluation

**Deliverables**:
- MLflow tracking server operational
- 10+ experiments logged (baseline, XGBoost, ensemble, NN)
- Best model registered with metadata

**Success Metrics**:
- All experiments logged with reproducible results
- Model artifacts < 100MB (XGBoost model file)
- Model load time < 500ms

**Research Finding**: MLflow production best practices (2024): Use MLflow Model Registry for versioning, model serving with MLflow deployments, and integrate with CI/CD for automated retraining pipelines.

**Code Example**:
```python
import mlflow
import mlflow.xgboost

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ntsb-severity-prediction")

# Start run
with mlflow.start_run(run_name="xgboost-tuned"):
    # Log parameters
    mlflow.log_params(best_params)

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.xgboost.log_model(model, "xgboost-model")

    # Log SHAP plots as artifacts
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png")

# Register model
model_uri = f"runs:/{run.info.run_id}/xgboost-model"
mlflow.register_model(model_uri, "ntsb-severity-classifier")

# Set to Production stage
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="ntsb-severity-classifier",
    version=1,
    stage="Production"
)
```

**Dependencies**: mlflow, xgboost, scikit-learn

### Week 11: FastAPI Model Serving

**Tasks**:
- [ ] Create FastAPI endpoint: POST /ml/predict
- [ ] Load model from MLflow Model Registry
- [ ] Implement request validation with Pydantic
- [ ] Add caching with Redis for repeated predictions
- [ ] Implement batch prediction endpoint: POST /ml/predict/batch
- [ ] Set latency target: <200ms for single prediction
- [ ] Load test with Locust (1000 requests, 100 concurrent users)

**Deliverables**:
- FastAPI model serving endpoints
- Pydantic request/response schemas
- Load test results (latency, throughput)

**Success Metrics**:
- Single prediction latency: <200ms (p95)
- Batch prediction throughput: >1000 predictions/second
- Load test: 1000 concurrent users without errors

**Code Example**:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc

router = APIRouter()

# Load model from MLflow Registry
model = mlflow.pyfunc.load_model("models:/ntsb-severity-classifier/Production")

class PredictionRequest(BaseModel):
    aircraft_age: int = Field(..., ge=0, le=100)
    num_engines: int = Field(..., ge=1, le=8)
    pilot_hours: float = Field(..., ge=0)
    weather_condition: str = Field(..., regex="^(VMC|IMC)$")
    phase_of_flight: str
    # ... 100+ features

class PredictionResponse(BaseModel):
    fatal_probability: float
    non_fatal_probability: float
    predicted_class: str
    confidence: float
    shap_top_features: list

@router.post("/ml/predict", response_model=PredictionResponse)
async def predict_severity(request: PredictionRequest):
    try:
        # Convert to DataFrame
        X = pd.DataFrame([request.dict()])

        # Feature engineering (reuse AviationFeatureEngineer)
        X = feature_engineer.transform(X)

        # Predict
        proba = model.predict(X)[0]

        # SHAP explanation
        shap_values = explainer.shap_values(X)[0]
        top_features = [
            {"feature": feat, "contribution": float(shap_values[i])}
            for i, feat in enumerate(feature_cols)
        ]
        top_features = sorted(top_features, key=lambda x: abs(x['contribution']), reverse=True)[:5]

        return PredictionResponse(
            fatal_probability=proba[1],
            non_fatal_probability=proba[0],
            predicted_class="fatal" if proba[1] > 0.5 else "non-fatal",
            confidence=max(proba),
            shap_top_features=top_features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction for efficiency"""
    X = pd.DataFrame([req.dict() for req in requests])
    X = feature_engineer.transform(X)

    probas = model.predict(X)

    return [
        {
            "fatal_probability": proba[1],
            "predicted_class": "fatal" if proba[1] > 0.5 else "non-fatal"
        }
        for proba in probas
    ]
```

**Dependencies**: fastapi, mlflow, pydantic, pandas

### Week 12: A/B Testing & Drift Monitoring

**Tasks**:
- [ ] Implement A/B testing framework: compare v1 vs v2 models
- [ ] Route 90% traffic to v1 (current), 10% to v2 (challenger)
- [ ] Track performance metrics for both models
- [ ] Install Evidently AI for data drift detection
- [ ] Monitor feature drift: compare training vs production distributions
- [ ] Monitor prediction drift: track prediction distribution shifts
- [ ] Set up alerts: Slack/email if drift detected (p-value < 0.05)
- [ ] Create drift monitoring dashboard

**Deliverables**:
- A/B testing infrastructure
- Evidently AI drift monitoring
- Drift detection dashboard
- Alerting system for model degradation

**Success Metrics**:
- A/B test: statistical significance within 1000 predictions
- Drift detection: identify shifts within 24 hours
- False positive rate < 5%

**Research Finding**: Data drift and model drift are critical in production ML. Evidently AI (2024) provides comprehensive drift detection with KS test, PSI (Population Stability Index), and custom metrics. Monitor both input features (data drift) and predictions (prediction drift).

**Code Example**:
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# A/B Testing
import random

@router.post("/ml/predict/ab")
async def predict_ab(request: PredictionRequest):
    # Route traffic
    model_version = "v2" if random.random() < 0.1 else "v1"

    model = mlflow.pyfunc.load_model(f"models:/ntsb-severity-classifier/{model_version}")

    # ... prediction logic ...

    # Log to analytics
    log_prediction(model_version=model_version, input=request, output=prediction)

    return prediction

# Drift Monitoring with Evidently
from evidently.pipeline.column_mapping import ColumnMapping

def check_drift(reference_data, current_data):
    """Check for data drift"""
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    drift_results = report.as_dict()

    # Check if drift detected
    drift_detected = drift_results['metrics'][0]['result']['dataset_drift']

    if drift_detected:
        send_alert(f"⚠️ Data drift detected! "
                   f"{drift_results['metrics'][0]['result']['number_of_drifted_columns']} "
                   f"features drifted.")

    return report

# Schedule drift monitoring (Airflow DAG)
from airflow import DAG
from airflow.operators.python import PythonOperator

def monitor_drift():
    # Load reference data (training set)
    reference_data = pd.read_parquet("data/train_features.parquet")

    # Load production data (last 7 days)
    current_data = fetch_production_data(days=7)

    # Check drift
    report = check_drift(reference_data, current_data)

    # Save report
    report.save_html("reports/drift_report.html")

dag = DAG('ml_drift_monitoring', schedule_interval='@daily')

monitor_task = PythonOperator(
    task_id='check_drift',
    python_callable=monitor_drift,
    dag=dag
)
```

**Dependencies**: evidently, airflow, mlflow, slack_sdk

**Sprint 4 Total Hours**: 75 hours

---

## Phase 3 Deliverables Summary

1. **Feature Engineering Pipeline**: 100+ features (NTSB codes, temporal, spatial, aircraft, crew)
2. **ML Models**: XGBoost (91%+ accuracy), Random Forest, Neural Network, Ensemble
3. **Hyperparameter Optimization**: Optuna-tuned XGBoost with spatial cross-validation
4. **Explainability**: SHAP values for all predictions, API integration
5. **Model Serving**: FastAPI endpoints with <200ms latency
6. **MLOps**: MLflow registry, A/B testing, Evidently drift monitoring

## Testing Checklist

- [ ] Feature engineering runs on full dataset without errors
- [ ] XGBoost accuracy > 91% on holdout test set
- [ ] Spatial cross-validation implemented correctly (no leakage)
- [ ] SHAP values computed for 1000+ predictions (validation)
- [ ] MLflow tracks all experiments with reproducible results
- [ ] API load test: 1000 concurrent users, <200ms latency
- [ ] Drift monitoring detects synthetic data shifts (validation test)

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Model accuracy | >91% | Holdout test set (20% of data) |
| ROC-AUC | >0.96 | Holdout test set |
| Feature count | 100+ | Feature engineering output |
| SHAP computation time | <100ms | API timing |
| API latency (p95) | <200ms | Load testing with Locust |
| Model size | <100MB | MLflow artifact |
| Drift detection latency | <24hrs | Evidently monitoring |

## Resource Requirements

**Infrastructure**:
- PostgreSQL with clean data (from Phase 1)
- MLflow tracking server (local or cloud)
- Redis for API caching
- FastAPI server (Docker container)

**Compute**:
- 32GB RAM for feature engineering
- Optional: GPU for neural network training (not required for XGBoost)
- 4-8 CPU cores for parallel training

**Python Libraries**:
- **ML**: xgboost, lightgbm, scikit-learn, torch, optuna
- **Explainability**: shap, matplotlib, seaborn
- **MLOps**: mlflow, evidently, fastapi, pydantic
- **Data**: pandas, numpy, polars

**Estimated Budget**: $50-200/month (MLflow cloud hosting optional, GPU training optional)

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Feature engineering bugs | Medium | High | Extensive unit testing, validate on sample data |
| Model overfitting | Medium | High | Spatial cross-validation, regularization, dropout |
| API performance issues | Medium | Medium | Caching, batch prediction, load balancing |
| Drift detection false positives | Medium | Low | Tune thresholds, manual review of alerts |
| MLflow downtime | Low | Medium | Use local tracking as fallback |

## Dependencies on Phase 1-2

- Clean PostgreSQL data with >95% quality (Phase 1)
- DBSCAN clusters for spatial CV (Phase 2)
- Airflow for scheduling drift monitoring (Phase 1)
- FastAPI infrastructure (Phase 1)

## Next Phase

Upon completion, proceed to [PHASE_4_AI_INTEGRATION.md](PHASE_4_AI_INTEGRATION.md) for NLP, RAG, knowledge graphs, and LLM integration.

---

**Last Updated**: January 2025
**Version**: 1.0
