# MACHINE LEARNING APPLICATIONS

Comprehensive guide to machine learning techniques for NTSB aviation accident analysis. Based on academic research and industry best practices.

## Table of Contents

- [Overview](#overview)
- [Supervised Learning](#supervised-learning)
- [Ensemble Methods](#ensemble-methods)
- [Deep Learning](#deep-learning)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Model Explainability](#model-explainability)
- [Production Deployment](#production-deployment)

## Overview

### Problem Formulations

**Classification Tasks**:
1. **Accident Severity Prediction** - 4-class (Fatal/Serious/Minor/None) - **Primary Use Case**
2. **Occurrence Type Classification** - Multi-class (100-430 codes)
3. **Probable Cause Identification** - Multi-label (multiple causes)
4. **Human Factors Detection** - Binary (human error Y/N)

**Regression Tasks**:
1. **Fatality Count Prediction** - Count regression
2. **Injury Severity Score** - Continuous scale
3. **Damage Cost Estimation** - Financial impact

**Time Series Tasks**:
1. **Accident Rate Forecasting** - Monthly/yearly predictions
2. **Seasonal Pattern Detection** - Cyclical trends
3. **Anomaly Detection** - Unusual accident clusters

### Performance Benchmarks (from Research)

| Model | Accuracy | F1-Score | Dataset | Reference |
|-------|----------|----------|---------|-----------|
| XGBoost | **91.2%** | 0.89 | NTSB 1990-2020 | IEEE 2023 |
| Random Forest | 88.7% | 0.86 | NTSB General Aviation | Embry-Riddle 2021 |
| LSTM + Attention | 87.9% | 0.85 | NTSB Reports (text) | ScienceDirect 2021 |
| Ensemble (RF+XGB+SVM) | 92.4% | 0.91 | Aviation accidents worldwide | Aviation Journal 2024 |
| Deep Neural Network | 85.3% | 0.82 | NTSB 1982-2019 | IRJET 2023 |

**Baseline**: Random guess for 4-class = 25% accuracy

## Supervised Learning

### Classification Algorithms

#### 1. Logistic Regression
**Best For**: Baseline model, interpretable coefficients

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().transform(X_test)

# Multi-class logistic regression
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    C=1.0,  # Regularization strength
    class_weight='balanced'  # Handle imbalanced classes
)
model.fit(X_train_scaled, y_train)

# Coefficient interpretation
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]  # For binary classification
}).sort_values('coefficient', key=abs, ascending=False)
```

**Hyperparameters**:
- `C`: Inverse regularization (0.001 - 10)
- `penalty`: 'l1', 'l2', 'elasticnet'
- `solver`: 'liblinear' (l1), 'lbfgs' (l2)

**Pros**: Fast, interpretable, probabilistic outputs
**Cons**: Assumes linear relationships, poor with non-linear patterns

#### 2. Support Vector Machines (SVM)
**Best For**: High-dimensional data, clear decision boundaries

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Grid search for hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm_model = SVC(
    probability=True,  # Enable probability estimates
    class_weight='balanced',
    random_state=42
)

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
```

**Hyperparameters**:
- `C`: Regularization (0.1 - 100)
- `kernel`: RBF (default), polynomial, sigmoid
- `gamma`: Kernel coefficient ('scale', 'auto', 0.001-1)

**Pros**: Effective in high dimensions, memory efficient
**Cons**: Slow for large datasets (>10K samples), sensitive to scaling

#### 3. K-Nearest Neighbors (KNN)
**Best For**: Small datasets, instance-based learning

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=5,  # Start with 5, tune with cross-validation
    weights='distance',  # Weight by inverse distance
    metric='minkowski',  # Euclidean distance
    p=2
)
knn_model.fit(X_train_scaled, y_train)

# Find optimal k
from sklearn.model_selection import cross_val_score

k_range = range(1, 31, 2)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
```

**Hyperparameters**:
- `n_neighbors`: Number of neighbors (3-30, typically 5-11)
- `weights`: 'uniform' or 'distance'
- `metric`: 'euclidean', 'manhattan', 'minkowski'

**Pros**: Simple, no training phase, effective for small datasets
**Cons**: Slow prediction, memory intensive, curse of dimensionality

### Naive Bayes
**Best For**: Text classification, categorical features

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# For count/frequency features (e.g., occurrence codes)
nb_model = MultinomialNB(alpha=1.0)  # Laplace smoothing

# For continuous features (e.g., lat/lon, temperatures)
nb_gaussian = GaussianNB(var_smoothing=1e-9)

nb_model.fit(X_train, y_train)
```

**Applications**:
- Text-based classification (narratives → severity)
- Quick baseline for categorical data
- Real-time prediction (fast inference)

**Pros**: Fast training/prediction, works with small data
**Cons**: Strong independence assumption, not for complex patterns

## Ensemble Methods

### Random Forest
**Best For**: Baseline ensemble, feature importance, robust predictions

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=500,  # More trees = better performance (diminishing returns after 500)
    max_depth=None,  # Grow trees fully (or limit to prevent overfitting)
    min_samples_split=5,  # Minimum samples to split node
    min_samples_leaf=2,  # Minimum samples per leaf
    max_features='sqrt',  # Features to consider at each split
    bootstrap=True,  # Bootstrap sampling
    oob_score=True,  # Out-of-bag score estimation
    class_weight='balanced',  # Handle imbalanced classes
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Out-of-bag score (validation without separate set)
print(f"OOB Score: {rf_model.oob_score_:.4f}")
```

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Number of combinations to try
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

**Pros**: High accuracy (95%+ in aviation studies), robust to overfitting, parallel training
**Cons**: Large model size, slower than single trees, less interpretable than single trees

### XGBoost (Extreme Gradient Boosting)
**Best For**: Highest accuracy, competition-winning performance

```python
import xgboost as xgb
from sklearn.metrics import log_loss

# Convert to DMatrix (XGBoost optimized format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Hyperparameters (tuned for aviation accidents)
params = {
    'objective': 'multi:softprob',  # Multi-class classification with probabilities
    'num_class': 4,  # Fatal, Serious, Minor, None
    'eval_metric': ['mlogloss', 'merror'],
    'max_depth': 6,  # Tree depth (3-10 typical)
    'learning_rate': 0.1,  # Eta (0.01-0.3)
    'subsample': 0.8,  # Row sampling per tree
    'colsample_bytree': 0.8,  # Column sampling per tree
    'min_child_weight': 3,  # Minimum sum of instance weight in child
    'gamma': 0.1,  # Minimum loss reduction for split
    'reg_alpha': 0.01,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'scale_pos_weight': 1,  # Balance of positive/negative weights
    'seed': 42
}

# Training with early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  # Maximum iterations
    evals=evals,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose_eval=50  # Print every 50 rounds
)

# Prediction
y_pred_proba = xgb_model.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)
```

**Advanced Hyperparameter Tuning**:
```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

def objective(params):
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])

    model = xgb.train(params, dtrain, num_boost_round=500)
    pred = model.predict(dtest)
    loss = log_loss(y_test, pred)
    return {'loss': loss, 'status': STATUS_OK}

search_space = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(1)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(10)),
    'objective': 'multi:softprob',
    'num_class': 4
}

trials = Trials()
best_params = fmin(objective, search_space, algo=tpe.suggest, max_evals=100, trials=trials)
```

**Pros**: Best accuracy (91%+ in studies), handles missing data, built-in regularization
**Cons**: Overfitting risk (requires tuning), slower than Random Forest, sensitive to hyperparameters

### LightGBM
**Best For**: Large datasets (>100K rows), faster than XGBoost

```python
import lightgbm as lgb

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': ['multi_logloss', 'multi_error'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # <2^max_depth to prevent overfitting
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.01,
    'lambda_l2': 1.0,
    'verbose': -1
}

# Train with early stopping
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    early_stopping_rounds=50,
    verbose_eval=50
)
```

**Pros**: Faster than XGBoost, lower memory usage, handles categorical features natively
**Cons**: Can overfit on small datasets, sensitive to hyperparameters

### AdaBoost
**Best For**: Combining weak learners, interpretable boosting

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R',  # Real AdaBoost (probability estimates)
    random_state=42
)
ada_model.fit(X_train, y_train)
```

**Pros**: Simple, interpretable, works well with weak learners
**Cons**: Sensitive to outliers, slower than other boosting methods

## Deep Learning

### Multi-Layer Perceptron (MLP)
**Best For**: Tabular data with complex non-linear relationships

```python
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,  # L2 regularization
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    verbose=True,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
```

**PyTorch Implementation** (more control):
```python
import torch
import torch.nn as nn
import torch.optim as optim

class AccidentSeverityNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super(AccidentSeverityNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize model
model = AccidentSeverityNN(
    input_dim=X_train.shape[1],
    hidden_dims=[256, 128, 64],
    num_classes=4,
    dropout=0.3
)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

**Hyperparameters**:
- `hidden_layer_sizes`: Architecture (e.g., (256,128,64))
- `activation`: 'relu', 'tanh', 'logistic'
- `alpha`: L2 regularization (0.0001-0.01)
- `learning_rate_init`: 0.0001-0.01
- `dropout`: 0.2-0.5 (if using PyTorch)

**Pros**: Captures complex non-linear patterns, flexible architecture
**Cons**: Requires careful tuning, prone to overfitting, black box

### LSTM for Time Series
**Best For**: Accident rate forecasting, temporal sequence analysis

```python
import torch
import torch.nn as nn

class LSTMAccidentPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMAccidentPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

# Initialize
model = LSTMAccidentPredictor(
    input_size=10,  # Number of features
    hidden_size=128,
    num_layers=2,
    output_size=1,  # Accident count prediction
    dropout=0.2
)
```

**Bidirectional LSTM** (best for aviation accident sequences):
```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True  # Bidirectional
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

**Research Finding**: Bidirectional LSTM achieves 87.9%+ accuracy on NTSB narrative classification.

### Transformer Models (BERT) for Text
**Best For**: Narrative analysis, causal extraction

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4  # Fatal, Serious, Minor, None
)

# Tokenize narratives
def tokenize_function(examples):
    return tokenizer(examples['narrative'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()
```

**SafeAeroBERT** (aviation-specific):
- Pretrained on ASRS and NTSB reports
- Achieves 87.9%+ accuracy on phase of flight classification
- Available via Hugging Face (community models)

## Feature Engineering

### Categorical Encoding

**One-Hot Encoding** (for low cardinality):
```python
from sklearn.preprocessing import OneHotEncoder

# Example: Aircraft category (Airplane, Helicopter, Glider)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
acft_category_encoded = encoder.fit_transform(df[['acft_category']])
```

**Label Encoding** (for ordinal):
```python
from sklearn.preprocessing import LabelEncoder

# Example: Injury severity (None < Minor < Serious < Fatal)
severity_order = {'NONE': 0, 'MINR': 1, 'SERS': 2, 'FATL': 3}
df['injury_encoded'] = df['ev_highest_injury'].map(severity_order)
```

**Target Encoding** (for high cardinality):
```python
from category_encoders import TargetEncoder

# Example: Aircraft make (1000+ unique values)
encoder = TargetEncoder(cols=['acft_make'])
df['acft_make_encoded'] = encoder.fit_transform(df['acft_make'], df['severity'])
```

### Numerical Features

**Scaling**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (mean=0, std=1) - default choice
scaler = StandardScaler()
df[['altitude', 'temperature', 'wind_speed']] = scaler.fit_transform(df[['altitude', 'temperature', 'wind_speed']])

# RobustScaler (median-based, robust to outliers)
robust_scaler = RobustScaler()
df[['pilot_tot_time']] = robust_scaler.fit_transform(df[['pilot_tot_time']])
```

**Binning**:
```python
# Age groups
df['pilot_age_group'] = pd.cut(df['pilot_age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])

# Experience levels
df['experience_level'] = pd.cut(df['pilot_tot_time'], bins=[0, 100, 500, 1500, float('inf')], labels=['novice', 'intermediate', 'experienced', 'professional'])
```

### Temporal Features

```python
# Extract from ev_date
df['ev_year'] = pd.to_datetime(df['ev_date']).dt.year
df['ev_month'] = pd.to_datetime(df['ev_date']).dt.month
df['ev_dayofweek'] = pd.to_datetime(df['ev_date']).dt.dayofweek
df['ev_quarter'] = pd.to_datetime(df['ev_date']).dt.quarter
df['is_weekend'] = (df['ev_dayofweek'] >= 5).astype(int)

# Cyclical encoding (month)
df['month_sin'] = np.sin(2 * np.pi * df['ev_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['ev_month'] / 12)
```

### Geospatial Features

```python
# Distance to nearest airport
from scipy.spatial.distance import cdist

accident_coords = df[['dec_latitude', 'dec_longitude']].values
airport_coords = airports[['lat', 'lon']].values

distances = cdist(accident_coords, airport_coords, metric='euclidean')
df['nearest_airport_dist'] = distances.min(axis=1)

# Density (accidents per region)
from scipy.stats import gaussian_kde

kde = gaussian_kde(df[['dec_latitude', 'dec_longitude']].T)
df['accident_density'] = kde(df[['dec_latitude', 'dec_longitude']].T)
```

### Aviation-Specific Features

```python
# Power-to-weight ratio
df['power_to_weight'] = df['eng_hp_or_lbs'] / df['cert_max_gr_wt']

# Experience ratio (type-specific / total)
df['type_experience_ratio'] = df['pilot_make_time'] / df['pilot_tot_time'].replace(0, 1)

# Recent activity ratio
df['recent_activity'] = df['pilot_30_days'] / df['pilot_90_days'].replace(0, 1)

# Multi-occurrence flag
occurrence_counts = df.groupby('ev_id')['occurrence_code'].transform('count')
df['multi_occurrence'] = (occurrence_counts > 1).astype(int)

# VMC/IMC binary
df['is_imc'] = (df['wx_cond_basic'] == 'IMC').astype(int)
```

### Text Features (from Narratives)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = vectorizer.fit_transform(df['narrative'])

# Word embeddings (Word2Vec)
from gensim.models import Word2Vec

sentences = [text.split() for text in df['narrative']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2)

# Average word vectors per narrative
def get_avg_vector(text):
    vectors = [w2v_model.wv[word] for word in text.split() if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

df['narrative_embedding'] = df['narrative'].apply(get_avg_vector).tolist()
```

## Model Evaluation

### Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Fatal', 'Serious', 'Minor', 'None']))

# ROC-AUC (multi-class)
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
```

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
print(f"CV F1 Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1_weighted'
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('F1 Score')
plt.legend()
```

## Model Explainability

### SHAP (SHapley Additive exPlanations)
**Best For**: Model-agnostic feature importance

```python
import shap

# Tree-based models (XGBoost, Random Forest)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot (single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Dependence plot (interaction effects)
shap.dependence_plot('pilot_tot_time', shap_values, X_test)
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Fatal', 'Serious', 'Minor', 'None'],
    mode='classification'
)

# Explain single prediction
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)
```

## Production Deployment

### Model Serialization

```python
import joblib
import pickle

# scikit-learn models
joblib.dump(model, 'accident_severity_model.pkl')
loaded_model = joblib.load('accident_severity_model.pkl')

# XGBoost models
xgb_model.save_model('xgb_model.json')
loaded_xgb = xgb.Booster()
loaded_xgb.load_model('xgb_model.json')

# PyTorch models
torch.save(model.state_dict(), 'pytorch_model.pth')
model.load_state_dict(torch.load('pytorch_model.pth'))
```

### Inference Pipeline

```python
class AccidentPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = [...]  # Load from config

    def preprocess(self, raw_data):
        # Feature engineering
        features = self.extract_features(raw_data)
        # Scaling
        features_scaled = self.scaler.transform(features)
        return features_scaled

    def predict(self, raw_data):
        features = self.preprocess(raw_data)
        pred_proba = self.model.predict_proba(features)
        pred_class = self.model.predict(features)
        return {
            'severity': ['Fatal', 'Serious', 'Minor', 'None'][pred_class[0]],
            'probabilities': {
                'Fatal': pred_proba[0][0],
                'Serious': pred_proba[0][1],
                'Minor': pred_proba[0][2],
                'None': pred_proba[0][3]
            }
        }
```

### Model Monitoring

```python
import mlflow

# Track experiments
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({'f1_score': f1, 'accuracy': accuracy})
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifacts('feature_importance.png')
```

---

**References**:
- Research papers: See `docs/RESEARCH_OPPORTUNITIES.md`
- Code examples: `examples/advanced_analysis.py`
- Next: See `AI_POWERED_ANALYSIS.md` for LLM integration

**Last Updated**: January 2025
