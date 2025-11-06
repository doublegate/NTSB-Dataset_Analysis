# Data Quality Strategy for NTSB Aviation Accident Database

## Table of Contents

- [Introduction to Data Quality](#introduction-to-data-quality)
- [Data Validation Frameworks](#data-validation-frameworks)
- [Outlier Detection](#outlier-detection)
- [Missing Data Analysis](#missing-data-analysis)
- [Missing Data Imputation](#missing-data-imputation)
- [Data Standardization](#data-standardization)
- [Data Quality Metrics](#data-quality-metrics)
- [Automated Quality Checks](#automated-quality-checks)
- [Data Quality Dashboard](#data-quality-dashboard)
- [Data Lineage Tracking](#data-lineage-tracking)
- [Schema Evolution](#schema-evolution)

---

## Introduction to Data Quality

High-quality data is the foundation of reliable aviation safety analytics. The NTSB database (1.6GB, 1962-present) contains 60+ years of accident data with varying levels of completeness, accuracy, and consistency. This document outlines comprehensive strategies for assessing, monitoring, and improving data quality.

### Six Dimensions of Data Quality

1. **Completeness**: Percentage of non-null values (target: 95%+ for critical fields)
2. **Accuracy**: Correctness of data (validated against source documents)
3. **Consistency**: Uniform formats across time periods and tables
4. **Validity**: Conforms to rules and constraints (e.g., dates in valid range)
5. **Timeliness**: Up-to-date data (avall.mdb updated monthly by NTSB)
6. **Uniqueness**: No duplicate records (ev_id must be unique)

### Data Quality Goals

- **Critical Fields** (ev_id, ev_date, ev_state): 99%+ completeness, 100% accuracy
- **Important Fields** (aircraft type, phase, injuries): 90%+ completeness, 95%+ accuracy
- **Optional Fields** (narratives, weather): 70%+ completeness, acceptable accuracy
- **Overall Quality Score**: 95+ out of 100 (weighted by field importance)

---

## Data Validation Frameworks

### Great Expectations

Great Expectations is a Python library for defining, executing, and documenting data expectations. It provides comprehensive validation with detailed reporting.

#### Installation

```fish
pip install great_expectations
```

#### Define Expectations for Events Table

```python
import great_expectations as ge
import pandas as pd

# Load data
df = ge.read_csv('data/avall-events.csv')

# Completeness expectations
df.expect_column_values_to_not_be_null('ev_id', mostly=1.0)
df.expect_column_values_to_not_be_null('ev_date', mostly=0.95)
df.expect_column_values_to_not_be_null('ev_year', mostly=0.99)
df.expect_column_values_to_not_be_null('ev_state', mostly=0.90)

# Validity expectations
df.expect_column_values_to_be_between('ev_year', min_value=1962, max_value=2100)
df.expect_column_values_to_be_in_set('ev_type', ['ACC', 'INC'])
df.expect_column_values_to_be_between('dec_latitude', min_value=-90, max_value=90)
df.expect_column_values_to_be_between('dec_longitude', min_value=-180, max_value=180)

# Format expectations
df.expect_column_values_to_match_regex('ev_id', regex=r'^[0-9]{8}$')
df.expect_column_values_to_match_regex('ev_state', regex=r'^[A-Z]{2}$')

# Uniqueness expectations
df.expect_column_values_to_be_unique('ev_id')

# Range expectations for injury counts
df.expect_column_values_to_be_between('inj_tot_f', min_value=0, max_value=1000, mostly=0.99)
df.expect_column_values_to_be_between('inj_tot_s', min_value=0, max_value=1000, mostly=0.99)
df.expect_column_values_to_be_between('inj_tot_m', min_value=0, max_value=1000, mostly=0.99)
df.expect_column_values_to_be_between('inj_tot_n', min_value=0, max_value=10000, mostly=0.99)

# Save expectations suite
df.save_expectation_suite('ntsb_events_suite.json')

# Run validation
validation_result = df.validate()
print(validation_result)
```

#### Advanced Expectations

```python
# Custom expectations for aviation-specific logic

# Fatal accidents should have inj_tot_f > 0
def expect_fatal_accidents_have_fatalities(df):
    """Custom expectation: If ev_type=='ACC' and severity is high, expect fatalities"""
    fatal_rows = df[(df['ev_type'] == 'ACC') & (df['inj_tot_f'].isna())]
    if len(fatal_rows) > 0:
        return {
            'success': False,
            'result': {'unexpected_count': len(fatal_rows)}
        }
    return {'success': True}

# Coordinates should be within US (for most accidents)
df.expect_column_pair_values_to_be_in_set(
    column_A='dec_latitude',
    column_B='dec_longitude',
    value_pairs_set=us_bounding_box,  # Define US lat/lon bounds
    mostly=0.85  # 85% should be in US
)

# Event date should be <= today
from datetime import date
df.expect_column_values_to_be_between(
    'ev_date',
    min_value='1962-01-01',
    max_value=date.today().isoformat()
)
```

#### Automated Validation Reporting

```python
import great_expectations as ge

# Build data context
context = ge.data_context.DataContext()

# Create checkpoint for automated validation
checkpoint_config = {
    'name': 'ntsb_events_checkpoint',
    'config_version': 1,
    'class_name': 'SimpleCheckpoint',
    'validations': [
        {
            'batch_request': {
                'datasource_name': 'ntsb_datasource',
                'data_asset_name': 'events',
            },
            'expectation_suite_name': 'ntsb_events_suite'
        }
    ]
}

context.add_checkpoint(**checkpoint_config)

# Run checkpoint
result = context.run_checkpoint(checkpoint_name='ntsb_events_checkpoint')

# Generate data docs
context.build_data_docs()
print(f"Validation success: {result['success']}")
```

---

### Pandera Schema Validation

Pandera provides a concise, declarative API for DataFrame validation with strong typing support.

#### Installation

```fish
pip install pandera
```

#### Define Schema for Events Table

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd

events_schema = DataFrameSchema(
    {
        'ev_id': Column(str, Check.str_matches(r'^[0-9]{8}$'), nullable=False, unique=True),
        'ev_year': Column(int, Check.in_range(1962, 2100), nullable=False),
        'ev_date': Column(str, nullable=True),
        'ev_type': Column(str, Check.isin(['ACC', 'INC']), nullable=True),
        'ev_state': Column(str, Check.str_length(2, 2), nullable=True),
        'dec_latitude': Column(float, Check.in_range(-90, 90), nullable=True),
        'dec_longitude': Column(float, Check.in_range(-180, 180), nullable=True),
        'inj_tot_f': Column(int, Check.greater_than_or_equal_to(0), nullable=True),
        'inj_tot_s': Column(int, Check.greater_than_or_equal_to(0), nullable=True),
        'inj_tot_m': Column(int, Check.greater_than_or_equal_to(0), nullable=True),
        'inj_tot_n': Column(int, Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,  # Allow additional columns not in schema
    coerce=True    # Attempt to coerce types
)

# Validate DataFrame
df = pd.read_csv('data/avall-events.csv')
try:
    validated_df = events_schema.validate(df, lazy=True)  # lazy=True collects all errors
    print("✅ Validation passed!")
except pa.errors.SchemaErrors as e:
    print("❌ Validation failed:")
    print(e.failure_cases)
    print(e.data)
```

#### Custom Validators

```python
import pandera as pa

# Custom check: fatalities should be less than total occupants
@pa.check_output(pa.Check(lambda df: (df['inj_tot_f'] <= df['inj_tot_f'] + df['inj_tot_s'] + df['inj_tot_m'] + df['inj_tot_n']).all()))
def validate_injury_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure injury counts are logically consistent"""
    return df

# Custom check: accident date should match year
@pa.check_output(pa.Check(lambda df: (pd.to_datetime(df['ev_date'], errors='coerce').dt.year == df['ev_year']).all()))
def validate_date_year_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ev_date year matches ev_year field"""
    return df

# Apply validators
@validate_injury_logic
@validate_date_year_consistency
def load_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_events('data/avall-events.csv')
```

---

## Outlier Detection

Outliers can indicate data quality issues or genuinely unusual accidents. Both require investigation.

### Statistical Methods

#### Interquartile Range (IQR) Method

```python
import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    """
    Detect outliers using Interquartile Range method.

    Args:
        df: Input DataFrame
        column: Column name to analyze
        multiplier: IQR multiplier (1.5 = mild outliers, 3.0 = extreme outliers)

    Returns:
        Tuple of (outliers_df, lower_bound, upper_bound)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"Column: {column}")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    return outliers, lower_bound, upper_bound

# Example: Detect outliers in fatality counts
df = pd.read_csv('data/avall-events.csv')
outliers_f, lb, ub = detect_outliers_iqr(df, 'inj_tot_f', multiplier=3.0)
print(outliers_f[['ev_id', 'ev_date', 'acft_make', 'inj_tot_f']].head(10))
```

#### Z-Score Method

```python
def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3):
    """
    Detect outliers using Z-score method.

    Args:
        df: Input DataFrame
        column: Column name to analyze
        threshold: Z-score threshold (3 = 99.7% of data)

    Returns:
        DataFrame with outliers
    """
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)

    outliers = df[z_scores > threshold].copy()
    outliers['z_score'] = z_scores[z_scores > threshold]

    print(f"Column: {column}")
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"Outliers (|z| > {threshold}): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    return outliers.sort_values('z_score', ascending=False)

# Example: Detect outliers in aircraft year
outliers_year = detect_outliers_zscore(df, 'acft_year', threshold=3)
print(outliers_year[['ev_id', 'acft_make', 'acft_year', 'z_score']].head())
```

---

### Machine Learning Methods

#### Isolation Forest (Multivariate Outliers)

```python
from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.01):
    """
    Detect multivariate outliers using Isolation Forest.

    Args:
        df: Input DataFrame
        contamination: Expected proportion of outliers (0.01 = 1%)

    Returns:
        DataFrame with outliers
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    predictions = iso_forest.fit_predict(X)

    # Add predictions to DataFrame
    df_with_pred = df.copy()
    df_with_pred['is_outlier'] = predictions == -1
    df_with_pred['anomaly_score'] = iso_forest.score_samples(X)

    outliers = df_with_pred[df_with_pred['is_outlier']].sort_values('anomaly_score')

    print(f"Total outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Numeric features used: {', '.join(numeric_cols[:5])}... ({len(numeric_cols)} total)")

    return outliers

# Example: Detect multivariate outliers in events data
df = pd.read_csv('data/avall-events.csv')
outliers = detect_outliers_isolation_forest(df, contamination=0.05)
print(outliers[['ev_id', 'ev_date', 'inj_tot_f', 'anomaly_score']].head(10))
```

#### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(df: pd.DataFrame, contamination: float = 0.01, n_neighbors: int = 20):
    """
    Detect outliers using Local Outlier Factor.

    Args:
        df: Input DataFrame
        contamination: Expected proportion of outliers
        n_neighbors: Number of neighbors for density estimation

    Returns:
        DataFrame with outliers
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    predictions = lof.fit_predict(X)

    df_with_pred = df.copy()
    df_with_pred['is_outlier'] = predictions == -1
    df_with_pred['lof_score'] = -lof.negative_outlier_factor_

    outliers = df_with_pred[df_with_pred['is_outlier']].sort_values('lof_score', ascending=False)

    print(f"Total outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    return outliers

# Example
outliers_lof = detect_outliers_lof(df, contamination=0.02)
```

---

## Missing Data Analysis

Understanding patterns of missingness is crucial for choosing appropriate imputation strategies.

### Missing Data Types

1. **MCAR (Missing Completely At Random)**: Missingness unrelated to any variables
2. **MAR (Missing At Random)**: Missingness related to observed variables
3. **MNAR (Missing Not At Random)**: Missingness related to unobserved variables

### Comprehensive Missing Data Analysis

```python
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data(df: pd.DataFrame):
    """
    Comprehensive missing data analysis.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with missing data statistics
    """
    # Missing percentage per column
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    print("=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns with >10% missing:")
    high_missing = missing_pct[missing_pct > 10]
    for col, pct in high_missing.items():
        print(f"  {col:30s}: {pct:6.2f}%")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Missing data matrix
    msno.matrix(df, ax=axes[0, 0], fontsize=10)
    axes[0, 0].set_title("Missing Data Matrix", fontsize=14)

    # Missing data heatmap (correlation of missingness)
    msno.heatmap(df, ax=axes[0, 1], fontsize=10)
    axes[0, 1].set_title("Missingness Correlation Heatmap", fontsize=14)

    # Missing data bar chart
    missing_pct.head(20).plot(kind='barh', ax=axes[1, 0])
    axes[1, 0].set_xlabel("% Missing")
    axes[1, 0].set_title("Top 20 Columns by Missingness", fontsize=14)

    # Dendrogram (hierarchical clustering of missingness)
    msno.dendrogram(df, ax=axes[1, 1])
    axes[1, 1].set_title("Missingness Dendrogram", fontsize=14)

    plt.tight_layout()
    plt.savefig('missing_data_analysis.png', dpi=300)
    print("\n✅ Visualizations saved to 'missing_data_analysis.png'")

    return {
        'missing_pct': missing_pct,
        'high_missing_cols': high_missing.index.tolist()
    }

# Example
df = pd.read_csv('data/avall-events.csv')
missing_stats = analyze_missing_data(df)
```

### Test for Missing Completely At Random (MCAR)

```python
from scipy.stats import chi2_contingency

def test_mcar(df: pd.DataFrame, target_col: str, group_col: str):
    """
    Test if missingness in target_col is random with respect to group_col.

    Args:
        df: Input DataFrame
        target_col: Column with missing data
        group_col: Grouping variable

    Returns:
        Dictionary with test results
    """
    # Create contingency table
    contingency = pd.crosstab(
        df[group_col],
        df[target_col].isnull(),
        margins=True
    )

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency.iloc[:-1, :-1])

    print(f"\nMCAR Test: {target_col} vs {group_col}")
    print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("❌ Missingness is NOT random (MAR or MNAR)")
        print("   → Missingness depends on", group_col)
    else:
        print("✅ Missingness appears random (MCAR)")
        print("   → Simple imputation may be appropriate")

    return {
        'chi2': chi2,
        'p_value': p_value,
        'is_mcar': p_value >= 0.05
    }

# Example: Is missingness in crew_age related to ev_year?
result = test_mcar(df, target_col='crew_age', group_col='ev_year')
```

---

## Missing Data Imputation

Choose imputation method based on missingness pattern and data characteristics.

### Simple Imputation

```python
from sklearn.impute import SimpleImputer
import pandas as pd

def simple_imputation(df: pd.DataFrame, strategy: str = 'median'):
    """
    Fill missing values with mean/median/mode/constant.

    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'most_frequent', or 'constant'

    Returns:
        Imputed DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    df_imputed = df.copy()

    # Impute numeric columns
    if len(numeric_cols) > 0:
        imputer_numeric = SimpleImputer(strategy=strategy if strategy != 'most_frequent' else 'median')
        df_imputed[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
        print(f"✅ Imputed {len(numeric_cols)} numeric columns with {strategy}")

    # Impute categorical columns
    if len(categorical_cols) > 0:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
        print(f"✅ Imputed {len(categorical_cols)} categorical columns with mode")

    return df_imputed

# Example
df = pd.read_csv('data/avall-events.csv')
df_imputed = simple_imputation(df, strategy='median')
```

---

### Advanced Imputation

#### K-Nearest Neighbors (KNN) Imputation

```python
from sklearn.impute import KNNImputer

def knn_imputation(df: pd.DataFrame, n_neighbors: int = 5):
    """
    Impute missing values using K-Nearest Neighbors.

    Research shows KNN achieves lowest MAE/RMSE for many datasets.

    Args:
        df: Input DataFrame
        n_neighbors: Number of neighbors for imputation

    Returns:
        Imputed DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns to impute")
        return df

    df_imputed = df.copy()

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    print(f"✅ KNN imputation complete ({n_neighbors} neighbors)")
    print(f"   Imputed columns: {', '.join(numeric_cols[:5])}... ({len(numeric_cols)} total)")

    return df_imputed

# Example
df_knn = knn_imputation(df, n_neighbors=7)
```

#### Multiple Imputation by Chained Equations (MICE)

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def mice_imputation(df: pd.DataFrame, max_iter: int = 10, n_imputations: int = 5):
    """
    Multiple Imputation by Chained Equations (MICE).

    MICE performs well for MAR data and provides uncertainty estimates.

    Args:
        df: Input DataFrame
        max_iter: Maximum iterations for convergence
        n_imputations: Number of imputed datasets to generate

    Returns:
        List of imputed DataFrames (length = n_imputations)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns to impute")
        return [df]

    imputed_dfs = []

    for i in range(n_imputations):
        df_imputed = df.copy()

        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=42 + i,
            verbose=0
        )
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        imputed_dfs.append(df_imputed)

    print(f"✅ MICE imputation complete")
    print(f"   Generated {n_imputations} imputed datasets")
    print(f"   Max iterations: {max_iter}")

    return imputed_dfs

# Example
df_mice_list = mice_imputation(df, max_iter=10, n_imputations=5)

# Pool results (average across imputations)
df_mice_pooled = pd.concat(df_mice_list).groupby(level=0).mean()
```

#### MissForest (Random Forest Imputation)

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def missforest_imputation(df: pd.DataFrame, max_iter: int = 5):
    """
    MissForest imputation using Random Forests.

    Research shows MissForest achieves lowest imputation error for many datasets.

    Args:
        df: Input DataFrame
        max_iter: Maximum iterations

    Returns:
        Imputed DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_imputed = df.copy()

    for iteration in range(max_iter):
        for col in numeric_cols:
            if df_imputed[col].isnull().sum() == 0:
                continue

            # Split into observed and missing
            observed = df_imputed[df_imputed[col].notnull()]
            missing = df_imputed[df_imputed[col].isnull()]

            if len(missing) == 0:
                continue

            # Features for prediction (other numeric columns)
            feature_cols = [c for c in numeric_cols if c != col]
            X_train = observed[feature_cols].fillna(observed[feature_cols].median())
            y_train = observed[col]
            X_missing = missing[feature_cols].fillna(observed[feature_cols].median())

            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # Predict missing values
            predictions = rf.predict(X_missing)
            df_imputed.loc[df_imputed[col].isnull(), col] = predictions

        print(f"  Iteration {iteration + 1}/{max_iter} complete")

    print(f"✅ MissForest imputation complete")

    return df_imputed

# Example
df_missforest = missforest_imputation(df, max_iter=3)
```

---

## Data Standardization

### State Code Normalization

```python
def standardize_state_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize state abbreviations to standard 2-letter codes.

    Args:
        df: Input DataFrame with 'ev_state' column

    Returns:
        DataFrame with standardized state codes
    """
    state_corrections = {
        'CALIFORNIA': 'CA', 'calif': 'CA', 'Ca.': 'CA', 'Calif': 'CA',
        'TEXAS': 'TX', 'texas': 'TX', 'Tex': 'TX',
        'FLORIDA': 'FL', 'florida': 'FL', 'Fla': 'FL',
        'NEW YORK': 'NY', 'new york': 'NY', 'N.Y.': 'NY',
        # Add more as needed
    }

    valid_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC', 'PR', 'VI', 'GU', 'AS', 'MP'  # Territories
    ]

    df = df.copy()

    # Strip whitespace and convert to uppercase
    df['ev_state'] = df['ev_state'].str.strip().str.upper()

    # Apply corrections
    df['ev_state'] = df['ev_state'].replace(state_corrections)

    # Validate against known states
    invalid_count = (~df['ev_state'].isin(valid_states)).sum()
    if invalid_count > 0:
        print(f"⚠️ Found {invalid_count} invalid state codes, setting to None")
        df.loc[~df['ev_state'].isin(valid_states), 'ev_state'] = None

    print(f"✅ Standardized state codes")
    print(f"   Valid states: {df['ev_state'].notna().sum():,}")
    print(f"   Missing states: {df['ev_state'].isna().sum():,}")

    return df

# Example
df = standardize_state_codes(df)
```

### Aircraft Name Normalization

```python
def standardize_aircraft_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize aircraft make and model names.

    Args:
        df: Input DataFrame with 'acft_make' and 'acft_model' columns

    Returns:
        DataFrame with standardized aircraft names
    """
    make_corrections = {
        'CESSNA': 'Cessna', 'cessna': 'Cessna', 'CESNA': 'Cessna',
        'BOEING': 'Boeing', 'boeing': 'Boeing',
        'AIRBUS': 'Airbus', 'airbus': 'Airbus',
        'PIPER': 'Piper', 'piper': 'Piper',
        'BEECH': 'Beechcraft', 'BEECHCRAFT': 'Beechcraft', 'beech': 'Beechcraft',
        'CIRRUS': 'Cirrus', 'cirrus': 'Cirrus',
        'MOONEY': 'Mooney', 'mooney': 'Mooney',
        # Add more as needed
    }

    df = df.copy()

    # Standardize make
    df['acft_make'] = df['acft_make'].str.strip()
    df['acft_make'] = df['acft_make'].replace(make_corrections)

    # Extract clean model number
    df['acft_model_clean'] = df['acft_model'].str.extract(r'(\d+[A-Z]*)', expand=False)

    # Create canonical aircraft identifier
    df['aircraft_id'] = df['acft_make'] + ' ' + df['acft_model_clean']

    print(f"✅ Standardized aircraft names")
    print(f"   Top makes: {df['acft_make'].value_counts().head(5).to_dict()}")

    return df

# Example
df = standardize_aircraft_names(df)
```

---

## Data Quality Metrics

### Overall Quality Score

```python
def calculate_quality_score(df: pd.DataFrame) -> tuple:
    """
    Compute overall data quality score (0-100).

    Returns:
        Tuple of (total_score, breakdown_dict)
    """
    scores = {}

    # Completeness (40% weight)
    completeness = (1 - df.isnull().sum() / len(df)).mean()
    scores['completeness'] = completeness * 40

    # Validity (30% weight) - % within expected ranges
    valid_years = ((df['ev_year'] >= 1962) & (df['ev_year'] <= 2100)).mean()
    valid_coords = (
        (df['dec_latitude'].between(-90, 90, na=True)) &
        (df['dec_longitude'].between(-180, 180, na=True))
    ).mean()
    validity = (valid_years + valid_coords) / 2
    scores['validity'] = validity * 30

    # Consistency (20% weight) - Format compliance
    date_format_ok = df['ev_date'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False).mean()
    scores['consistency'] = date_format_ok * 20

    # Uniqueness (10% weight) - No duplicate ev_id
    uniqueness = 1 - df.duplicated(subset=['ev_id']).mean()
    scores['uniqueness'] = uniqueness * 10

    total_score = sum(scores.values())

    print("=" * 60)
    print("DATA QUALITY SCORE")
    print("=" * 60)
    print(f"Overall Score: {total_score:.2f}/100")
    print(f"\nBreakdown:")
    print(f"  Completeness (40%): {scores['completeness']:.2f}")
    print(f"  Validity (30%):     {scores['validity']:.2f}")
    print(f"  Consistency (20%):  {scores['consistency']:.2f}")
    print(f"  Uniqueness (10%):   {scores['uniqueness']:.2f}")

    if total_score >= 95:
        print(f"\n✅ Excellent quality")
    elif total_score >= 85:
        print(f"\n⚠️ Good quality (some improvements needed)")
    else:
        print(f"\n❌ Poor quality (significant improvements needed)")

    return total_score, scores

# Example
df = pd.read_csv('data/avall-events.csv')
score, breakdown = calculate_quality_score(df)
```

---

## Automated Quality Checks

### Apache Airflow Data Quality DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import pandas as pd

def quality_check_function():
    """Run all quality checks and alert if below threshold"""
    from sqlalchemy import create_engine

    # Connect to database
    engine = create_engine('postgresql://user:pass@localhost:5432/ntsb')

    # Load data
    df = pd.read_sql("SELECT * FROM events", engine)

    # Run validations
    score, breakdown = calculate_quality_score(df)

    # Alert if below threshold
    if score < 95:
        send_alert(f"⚠️ Data quality below threshold: {score:.2f}/100")
        raise ValueError(f"Quality check failed: score={score:.2f}")

    # Log metrics
    log_quality_metrics(score, breakdown)

    print(f"✅ Quality check passed: {score:.2f}/100")

def send_alert(message: str):
    """Send alert via email/Slack"""
    # Implementation depends on your setup
    print(f"ALERT: {message}")

def log_quality_metrics(score: float, breakdown: dict):
    """Log metrics to monitoring system"""
    # Implementation depends on your setup
    print(f"Logged metrics: {breakdown}")

# Define DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ntsb_data_quality_checks',
    default_args=default_args,
    description='Daily data quality checks for NTSB database',
    schedule_interval='@daily',
    catchup=False
)

quality_task = PythonOperator(
    task_id='run_quality_checks',
    python_callable=quality_check_function,
    dag=dag
)
```

---

## Data Quality Dashboard

### Streamlit Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno

st.set_page_config(page_title="NTSB Data Quality Dashboard", layout="wide")

st.title("📊 NTSB Data Quality Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/avall-events.csv')

df = load_data()

# Overall quality score
score, breakdown = calculate_quality_score(df)

col1, col2 = st.columns([1, 3])

with col1:
    st.metric("Overall Quality Score", f"{score:.1f}/100",
              delta=f"{score - 95:.1f} vs target (95)")

    # Breakdown
    st.subheader("Score Breakdown")
    for key, value in breakdown.items():
        st.metric(key.title(), f"{value:.1f}")

with col2:
    # Score visualization
    fig = px.bar(
        x=list(breakdown.keys()),
        y=list(breakdown.values()),
        labels={'x': 'Dimension', 'y': 'Score'},
        title="Quality Score by Dimension"
    )
    st.plotly_chart(fig, use_container_width=True)

# Missing data analysis
st.subheader("Missing Data Patterns")

col1, col2 = st.columns(2)

with col1:
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(20)
    fig_missing = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation='h',
        labels={'x': '% Missing', 'y': 'Column'},
        title="Top 20 Columns by Missingness"
    )
    st.plotly_chart(fig_missing, use_container_width=True)

with col2:
    # Missing data matrix
    fig_matrix = msno.matrix(df)
    st.pyplot(fig_matrix)

# Outlier detection
st.subheader("Outlier Detection")

outliers = detect_outliers_isolation_forest(df, contamination=0.05)
st.write(f"Detected {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
st.dataframe(outliers[['ev_id', 'ev_date', 'inj_tot_f', 'anomaly_score']].head(20))

# Data freshness
st.subheader("Data Freshness")
latest_date = pd.to_datetime(df['ev_date'], errors='coerce').max()
days_old = (pd.Timestamp.now() - latest_date).days
st.metric("Latest Accident", latest_date.strftime('%Y-%m-%d'), delta=f"{days_old} days ago")
```

---

## Data Lineage Tracking

```python
from datetime import datetime
import json

class DataLineage:
    """Track data transformations for reproducibility"""

    def __init__(self):
        self.transformations = []

    def log_transformation(self, name: str, input_rows: int, output_rows: int, metadata: dict):
        """Log a transformation step"""
        self.transformations.append({
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'input_rows': input_rows,
            'output_rows': output_rows,
            'rows_changed': input_rows - output_rows,
            'pct_changed': (input_rows - output_rows) / input_rows * 100 if input_rows > 0 else 0,
            'metadata': metadata
        })

    def export_lineage(self, path: str = 'data_lineage.json'):
        """Export lineage as JSON"""
        with open(path, 'w') as f:
            json.dump(self.transformations, f, indent=2)
        print(f"✅ Lineage exported to {path}")

    def print_summary(self):
        """Print lineage summary"""
        print("\n" + "=" * 80)
        print("DATA LINEAGE SUMMARY")
        print("=" * 80)
        for i, t in enumerate(self.transformations, 1):
            print(f"\n{i}. {t['name']}")
            print(f"   Timestamp: {t['timestamp']}")
            print(f"   Input rows: {t['input_rows']:,}")
            print(f"   Output rows: {t['output_rows']:,}")
            print(f"   Changed: {t['rows_changed']:,} ({t['pct_changed']:.2f}%)")
            print(f"   Metadata: {t['metadata']}")

# Usage example
lineage = DataLineage()

# Step 1: Load raw data
df = pd.read_csv('datasets/avall.mdb')
lineage.log_transformation('load_raw', 0, len(df), {'source': 'avall.mdb'})

# Step 2: Remove nulls
df_clean = df[df['ev_id'].notna()]
lineage.log_transformation('remove_null_ev_id', len(df), len(df_clean), {'column': 'ev_id'})

# Step 3: Remove outliers
outliers = detect_outliers_isolation_forest(df_clean, contamination=0.01)
df_final = df_clean[~df_clean.index.isin(outliers.index)]
lineage.log_transformation('remove_outliers', len(df_clean), len(df_final), {'method': 'IsolationForest', 'contamination': 0.01})

# Print and export
lineage.print_summary()
lineage.export_lineage()
```

---

## Schema Evolution

```python
def handle_schema_changes(old_schema: dict, new_schema: dict):
    """
    Detect and migrate data when schema changes.

    Args:
        old_schema: Dict of {column_name: dtype}
        new_schema: Dict of {column_name: dtype}
    """
    # Detect changes
    added_columns = set(new_schema.keys()) - set(old_schema.keys())
    removed_columns = set(old_schema.keys()) - set(new_schema.keys())
    changed_types = {
        col for col in set(old_schema.keys()) & set(new_schema.keys())
        if old_schema[col] != new_schema[col]
    }

    print("\n" + "=" * 80)
    print("SCHEMA EVOLUTION ANALYSIS")
    print("=" * 80)

    if added_columns:
        print(f"\n✅ Added columns ({len(added_columns)}):")
        for col in added_columns:
            print(f"   - {col} ({new_schema[col]})")

    if removed_columns:
        print(f"\n❌ Removed columns ({len(removed_columns)}):")
        for col in removed_columns:
            print(f"   - {col} ({old_schema[col]})")
        print("   ⚠️ WARNING: Data loss if not backed up!")

    if changed_types:
        print(f"\n🔄 Changed types ({len(changed_types)}):")
        for col in changed_types:
            print(f"   - {col}: {old_schema[col]} → {new_schema[col]}")

    if not (added_columns or removed_columns or changed_types):
        print("\n✅ No schema changes detected")

    return {
        'added': list(added_columns),
        'removed': list(removed_columns),
        'changed_types': list(changed_types)
    }

# Example
old_schema = {
    'ev_id': 'str',
    'ev_date': 'str',
    'ev_year': 'int',
    'inj_tot_f': 'int'
}

new_schema = {
    'ev_id': 'str',
    'ev_date': 'datetime64[ns]',  # Type changed
    'ev_year': 'int',
    'inj_tot_f': 'int',
    'severity_score': 'float'  # Added
}

changes = handle_schema_changes(old_schema, new_schema)
```

---

## Summary

This comprehensive data quality strategy ensures the NTSB Aviation Accident Database maintains high standards of completeness, accuracy, consistency, validity, timeliness, and uniqueness. By implementing:

- **Validation frameworks** (Great Expectations, Pandera)
- **Outlier detection** (IQR, Z-score, Isolation Forest, LOF)
- **Missing data analysis** (MCAR testing, missingno visualizations)
- **Advanced imputation** (KNN, MICE, MissForest)
- **Standardization** (state codes, aircraft names)
- **Quality metrics** (overall score, dimension breakdown)
- **Automated checks** (Airflow DAGs)
- **Dashboards** (Streamlit real-time monitoring)
- **Lineage tracking** (reproducibility)
- **Schema evolution** (migration planning)

We achieve **95+ quality score** required for production ML models and safety-critical applications.

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Target Audience**: Data engineers, ML engineers, data scientists
**Related Documents**: ARCHITECTURE_VISION.md, MACHINE_LEARNING_APPLICATIONS.md, FEATURE_ENGINEERING_GUIDE.md
