# FEATURE ENGINEERING GUIDE

Comprehensive guide to feature engineering for aviation accident prediction using NTSB data. This document provides production-ready code examples for extracting, transforming, and selecting features from aviation accident databases.

## Table of Contents

- [Introduction](#introduction)
- [Domain-Specific Features from NTSB Codes](#domain-specific-features-from-ntsb-codes)
- [Temporal Features Engineering](#temporal-features-engineering)
- [Geospatial Features Engineering](#geospatial-features-engineering)
- [Aircraft Features Engineering](#aircraft-features-engineering)
- [Crew Features Engineering](#crew-features-engineering)
- [Text Features from Narratives](#text-features-from-narratives)
- [Interaction Features](#interaction-features)
- [Feature Selection Techniques](#feature-selection-techniques)
- [Feature Store Design](#feature-store-design)
- [Complete Feature Engineering Pipeline](#complete-feature-engineering-pipeline)
- [Performance Optimization](#performance-optimization)

## Introduction

### Why Feature Engineering is Critical for Aviation ML

Aviation accident prediction requires domain expertise to extract meaningful patterns from complex data. Raw NTSB database fields alone are insufficient—effective models require engineered features that capture:

- **Hierarchical patterns** in NTSB coding system (100-93300 range)
- **Temporal dynamics** (seasonal patterns, time-of-day effects, historical trends)
- **Geospatial relationships** (proximity to airports, terrain characteristics, regional patterns)
- **Complex interactions** between pilot experience, aircraft age, weather conditions
- **Derived aviation metrics** (power-to-weight ratios, currency, recent activity)

### Feature Engineering Best Practices

1. **Start with domain knowledge**: Leverage aviation safety expertise
2. **Validate features individually**: Check distributions and outlier patterns
3. **Test feature importance**: Use SHAP, permutation importance before deployment
4. **Handle missing values carefully**: Aviation data has systematic missingness patterns
5. **Version features alongside models**: Track feature definitions in code
6. **Document aviation-specific logic**: Why features were created, not just how

### Tools and Libraries

```python
# Core libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Feature engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from category_encoders import TargetEncoder

# Geospatial
import geopandas as gpd
from geopy.distance import geodesic

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Performance
import polars as pl  # 10x faster than pandas for large datasets
```

## Domain-Specific Features from NTSB Codes

### Occurrence Features (Codes 100-430)

Extract hierarchical features from occurrence codes to capture **what happened** during the accident.

```python
def extract_occurrence_features(df):
    """
    Extract features from NTSB occurrence codes (100-430).

    Args:
        df: DataFrame with 'occurrence_code' column

    Returns:
        DataFrame with additional occurrence-based features
    """

    # Binary features for major occurrence categories
    df['has_engine_failure'] = df['occurrence_code'].between(100, 119).astype(int)
    df['has_powerplant_failure'] = df['occurrence_code'].between(120, 139).astype(int)
    df['has_propeller_failure'] = df['occurrence_code'].between(140, 159).astype(int)
    df['has_engine_fire'] = df['occurrence_code'].between(160, 179).astype(int)

    df['has_flight_control_failure'] = df['occurrence_code'].between(200, 219).astype(int)
    df['has_control_separation'] = df['occurrence_code'].between(220, 239).astype(int)
    df['has_control_jamming'] = df['occurrence_code'].between(240, 259).astype(int)

    df['has_structural_failure'] = df['occurrence_code'].between(260, 279).astype(int)
    df['has_airframe_icing'] = (df['occurrence_code'] == 290).astype(int)

    # Collision types (critical for severity)
    df['is_midair_collision'] = (df['occurrence_code'] == 300).astype(int)
    df['is_cfit'] = (df['occurrence_code'] == 310).astype(int)  # Controlled Flight Into Terrain
    df['is_obstacle_collision'] = df['occurrence_code'].between(320, 339).astype(int)
    df['is_water_collision'] = (df['occurrence_code'] == 330).astype(int)
    df['is_ground_collision'] = df['occurrence_code'].between(340, 359).astype(int)

    # Fuel issues (highly preventable)
    df['is_fuel_exhaustion'] = (df['occurrence_code'] == 380).astype(int)
    df['is_fuel_starvation'] = (df['occurrence_code'] == 390).astype(int)
    df['has_fuel_issue'] = df[['is_fuel_exhaustion', 'is_fuel_starvation']].max(axis=1)

    # Fire and smoke
    df['has_fire_inflight'] = (df['occurrence_code'] == 400).astype(int)
    df['has_fire_postcrash'] = (df['occurrence_code'] == 405).astype(int)
    df['has_smoke'] = (df['occurrence_code'] == 410).astype(int)

    # Critical occurrences (high fatality correlation)
    df['is_hard_landing'] = (df['occurrence_code'] == 420).astype(int)
    df['is_abrupt_maneuver'] = (df['occurrence_code'] == 425).astype(int)
    df['is_loss_of_control'] = (df['occurrence_code'] == 430).astype(int)

    # Occurrence category (hierarchical grouping)
    def categorize_occurrence(code):
        if pd.isna(code):
            return 'unknown'
        elif 100 <= code <= 179:
            return 'powerplant'
        elif 200 <= code <= 259:
            return 'flight_control'
        elif 260 <= code <= 290:
            return 'structural'
        elif 300 <= code <= 370:
            return 'collision'
        elif 380 <= code <= 390:
            return 'fuel'
        elif 400 <= code <= 410:
            return 'fire'
        elif 420 <= code <= 430:
            return 'loss_of_control'
        else:
            return 'other'

    df['occurrence_category'] = df['occurrence_code'].apply(categorize_occurrence)

    return df
```

### Phase of Flight Features (Codes 500-620)

Extract features indicating **when** the accident occurred in the flight profile.

```python
def extract_phase_features(df):
    """
    Extract features from phase of operation codes (500-620).

    Critical phases (takeoff, approach, landing) account for 70%+ of accidents.
    """

    # Binary phase indicators
    df['phase_standing'] = (df['phase_code'] == 500).astype(int)
    df['phase_taxi'] = df['phase_code'].between(510, 539).astype(int)
    df['phase_takeoff'] = df['phase_code'].between(550, 570).astype(int)
    df['phase_climb'] = (df['phase_code'] == 580).astype(int)
    df['phase_cruise'] = (df['phase_code'] == 582).astype(int)
    df['phase_descent'] = (df['phase_code'] == 585).astype(int)
    df['phase_approach'] = (df['phase_code'] == 590).astype(int)
    df['phase_goaround'] = (df['phase_code'] == 595).astype(int)
    df['phase_landing'] = df['phase_code'].between(600, 610).astype(int)
    df['phase_maneuvering'] = (df['phase_code'] == 620).astype(int)

    # Critical phases (high accident rate)
    df['phase_critical'] = (
        df['phase_takeoff'] |
        df['phase_approach'] |
        df['phase_landing']
    ).astype(int)

    # Phase category
    def categorize_phase(code):
        if pd.isna(code):
            return 'unknown'
        elif 500 <= code <= 540:
            return 'ground'
        elif 550 <= code <= 570:
            return 'takeoff'
        elif 580 <= code <= 585:
            return 'enroute'
        elif 590 <= code <= 610:
            return 'approach_landing'
        elif code == 620:
            return 'maneuvering'
        else:
            return 'other'

    df['phase_category'] = df['phase_code'].apply(categorize_phase)

    return df
```

### Finding/Cause Code Features (Codes 10000-93300)

Extract features from investigation findings to capture **why** the accident happened.

```python
def extract_cause_features(df_events, df_findings):
    """
    Aggregate finding codes per event into features.

    Args:
        df_events: Events table
        df_findings: Findings table (many-to-one with events)

    Returns:
        df_events with aggregated cause features
    """

    # Filter to probable cause findings only (Release 3.0+)
    df_pc = df_findings[df_findings['cm_inPC'] == True].copy()

    # Categorize finding codes
    def categorize_finding(code):
        if pd.isna(code):
            return 'unknown'
        elif 10000 <= code <= 11999:
            return 'airframe'
        elif 12000 <= code <= 13999:
            return 'systems'
        elif 14000 <= code <= 17999:
            return 'powerplant'
        elif 22000 <= code <= 23999:
            return 'performance'
        elif 24000 <= code <= 25999:
            return 'operations'
        elif 30000 <= code <= 89999:
            return 'mechanical_failure'
        elif 90000 <= code <= 93999:
            return 'organizational'
        else:
            return 'other'

    df_pc['finding_category'] = df_pc['finding_code'].apply(categorize_finding)

    # Aggregate counts per event
    cause_counts = df_pc.groupby(['ev_id', 'finding_category']).size().unstack(fill_value=0)
    cause_counts.columns = [f'cause_count_{col}' for col in cause_counts.columns]

    # Total number of probable cause findings
    cause_total = df_pc.groupby('ev_id').size().rename('cause_count_total')

    # Most common finding code per event
    most_common_finding = df_pc.groupby('ev_id')['finding_code'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
    ).rename('primary_finding_code')

    # Human factors flag (codes 24000-25000, 90000-93000)
    human_factors = df_pc[
        (df_pc['finding_code'].between(24000, 25999)) |
        (df_pc['finding_code'].between(90000, 93999))
    ].groupby('ev_id').size().rename('has_human_factors')

    # Merge back to events
    df_events = df_events.merge(cause_counts, left_on='ev_id', right_index=True, how='left')
    df_events = df_events.merge(cause_total, left_on='ev_id', right_index=True, how='left')
    df_events = df_events.merge(most_common_finding, left_on='ev_id', right_index=True, how='left')
    df_events = df_events.merge(human_factors, left_on='ev_id', right_index=True, how='left')

    # Fill NaN (events with no findings)
    cause_cols = [col for col in df_events.columns if col.startswith('cause_count_')]
    df_events[cause_cols] = df_events[cause_cols].fillna(0).astype(int)
    df_events['has_human_factors'] = df_events['has_human_factors'].fillna(0).astype(int)

    return df_events
```

## Temporal Features Engineering

### Date/Time Features

```python
def create_temporal_features(df):
    """
    Extract temporal patterns from ev_date field.

    Cyclical encoding ensures model understands temporal continuity
    (e.g., December and January are adjacent months).
    """

    df['ev_date'] = pd.to_datetime(df['ev_date'])

    # Basic temporal components
    df['year'] = df['ev_date'].dt.year
    df['month'] = df['ev_date'].dt.month
    df['day'] = df['ev_date'].dt.day
    df['day_of_week'] = df['ev_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['ev_date'].dt.dayofyear
    df['week_of_year'] = df['ev_date'].dt.isocalendar().week
    df['quarter'] = df['ev_date'].dt.quarter

    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding (preserves periodicity)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Season (meteorological)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    df['season'] = df['month'].apply(get_season)

    # Holiday season (high GA activity)
    df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
    df['is_summer_travel'] = df['month'].isin([6, 7, 8]).astype(int)

    # Years since start of dataset (trend feature)
    min_year = df['year'].min()
    df['years_since_start'] = df['year'] - min_year

    return df
```

### Historical Aggregation Features

```python
def create_historical_features(df, window_days=[30, 90, 365]):
    """
    Create rolling window aggregations for temporal patterns.

    WARNING: This can be slow on large datasets. Consider using Polars.
    """

    df = df.sort_values('ev_date')

    for window in window_days:
        # Accidents in rolling window (by state)
        df[f'accidents_last_{window}d'] = df.groupby('ev_state')['ev_id'].transform(
            lambda x: x.rolling(window=f'{window}D', on=df.loc[x.index, 'ev_date'], min_periods=1).count()
        )

        # Fatalities in rolling window
        df[f'fatalities_last_{window}d'] = df.groupby('ev_state')['inj_tot_f'].transform(
            lambda x: x.rolling(window=f'{window}D', on=df.loc[x.index, 'ev_date'], min_periods=1).sum()
        )

    # Trend: recent vs historical accident rate
    if 'accidents_last_30d' in df.columns and 'accidents_last_365d' in df.columns:
        df['accident_rate_trend'] = df['accidents_last_30d'] / (df['accidents_last_365d'] + 1)

    return df
```

### Lag Features for Time Series

```python
def create_lag_features(df, group_cols=['acft_make', 'acft_model'], lags=[1, 3, 6, 12]):
    """
    Create lag features for time series prediction.

    Useful for forecasting accident rates by aircraft type.
    """

    df = df.sort_values('ev_date')

    for lag in lags:
        # Previous accident severity
        df[f'severity_lag_{lag}'] = df.groupby(group_cols)['severity_encoded'].shift(lag)

        # Previous fatality count
        df[f'fatalities_lag_{lag}'] = df.groupby(group_cols)['inj_tot_f'].shift(lag)

        # Days since last accident
        df[f'days_since_last_{lag}'] = (
            df.groupby(group_cols)['ev_date'].diff(lag).dt.days
        )

    return df
```

## Geospatial Features Engineering

### Location-Based Features

```python
def create_geospatial_features(df):
    """
    Extract geographic patterns from coordinates and location data.
    """

    # State-level accident statistics
    state_stats = df.groupby('ev_state').agg({
        'ev_id': 'count',
        'inj_tot_f': 'sum',
        'inj_tot_s': 'sum'
    }).rename(columns={
        'ev_id': 'state_accident_count',
        'inj_tot_f': 'state_total_fatalities',
        'inj_tot_s': 'state_total_serious'
    })

    # Fatality rate per state
    state_stats['state_fatality_rate'] = (
        state_stats['state_total_fatalities'] / state_stats['state_accident_count']
    )

    df = df.merge(state_stats, left_on='ev_state', right_index=True, how='left')

    # Geographic regions (FAA regions)
    regions = {
        'Western': ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'AK', 'HI'],
        'Central': ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
        'Southern': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'WV', 'VA', 'NC', 'SC', 'GA', 'FL'],
        'Eastern': ['PA', 'NY', 'VT', 'NH', 'ME', 'MA', 'RI', 'CT', 'NJ', 'DE', 'MD', 'DC']
    }

    def get_region(state):
        for region, states in regions.items():
            if state in states:
                return region
        return 'Other'

    df['region'] = df['ev_state'].apply(get_region)

    # Latitude/longitude derived features
    df['abs_latitude'] = df['dec_latitude'].abs()
    df['abs_longitude'] = df['dec_longitude'].abs()

    # Rough climate zones (based on latitude)
    def get_climate_zone(lat):
        if pd.isna(lat):
            return 'unknown'
        lat = abs(lat)
        if lat < 23.5:
            return 'tropical'
        elif lat < 35:
            return 'subtropical'
        elif lat < 50:
            return 'temperate'
        else:
            return 'subarctic'

    df['climate_zone'] = df['dec_latitude'].apply(get_climate_zone)

    return df
```

### Distance to Nearest Airport

```python
def create_distance_features(df, airports_df):
    """
    Calculate distance to nearest airport for each accident.

    Args:
        df: Accident DataFrame with dec_latitude, dec_longitude
        airports_df: DataFrame with airport lat, lon columns

    Returns:
        df with distance features added
    """
    from scipy.spatial import cKDTree

    # Remove missing coordinates
    df_with_coords = df.dropna(subset=['dec_latitude', 'dec_longitude']).copy()
    airports_clean = airports_df.dropna(subset=['latitude', 'longitude']).copy()

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(airports_clean[['latitude', 'longitude']].values)

    # Query nearest airport for each accident
    accident_coords = df_with_coords[['dec_latitude', 'dec_longitude']].values
    distances, indices = tree.query(accident_coords, k=1)

    # Convert to kilometers (approximate)
    df_with_coords['dist_to_nearest_airport_km'] = distances * 111  # 1 degree ≈ 111 km

    # Nearest airport ID
    df_with_coords['nearest_airport_id'] = airports_clean.iloc[indices]['airport_id'].values

    # Distance categories
    df_with_coords['airport_proximity'] = pd.cut(
        df_with_coords['dist_to_nearest_airport_km'],
        bins=[0, 5, 20, 50, 100, 500, float('inf')],
        labels=['very_close', 'close', 'medium', 'far', 'very_far', 'remote']
    )

    # Merge back to original DataFrame
    df = df.merge(
        df_with_coords[['ev_id', 'dist_to_nearest_airport_km', 'nearest_airport_id', 'airport_proximity']],
        on='ev_id',
        how='left'
    )

    return df
```

## Aircraft Features Engineering

### Aircraft Age and Type Features

```python
def create_aircraft_features(df):
    """
    Extract aircraft-specific characteristics.
    """

    # Aircraft age at time of accident
    df['aircraft_age'] = df['ev_year'] - df['acft_year']
    df['aircraft_age'] = df['aircraft_age'].clip(lower=0)  # Handle data errors

    # Age categories
    df['aircraft_age_category'] = pd.cut(
        df['aircraft_age'],
        bins=[0, 5, 15, 30, 50, 100],
        labels=['new', 'moderate', 'old', 'very_old', 'vintage']
    )

    # Aircraft category risk (based on historical data)
    category_risk = {
        'Airplane': 0.50,
        'Helicopter': 0.72,
        'Glider': 0.28,
        'Balloon': 0.15,
        'Ultralight': 0.85,
        'Weight-Shift': 0.78,
        'Powered Parachute': 0.65,
        'Gyroplane': 0.68
    }
    df['aircraft_category_risk'] = df['acft_category'].map(category_risk).fillna(0.50)

    # Common aircraft models (high accident frequency)
    common_models = ['172', '152', '182', 'PA-28', 'PA-18', 'C-150', 'C-140']
    df['is_common_model'] = df['acft_model'].apply(
        lambda x: any(model in str(x).upper() for model in common_models) if pd.notna(x) else False
    ).astype(int)

    # Certified vs experimental
    df['is_experimental'] = (df['acft_category'].str.contains('EXP', case=False, na=False)).astype(int)

    # Multi-engine aircraft
    df['is_multi_engine'] = (df['num_eng'] > 1).astype(int)

    # Turbine vs reciprocating
    df['has_turbine'] = (df['eng_type'].str.contains('TURB', case=False, na=False)).astype(int)

    # Power-to-weight ratio (performance metric)
    df['power_to_weight'] = df['eng_hp_or_lbs'] / (df['cert_max_gr_wt'] + 1)

    return df
```

### Manufacturer Safety Statistics

```python
def create_manufacturer_features(df):
    """
    Aggregate historical accident rates by manufacturer.
    """

    # Manufacturer accident statistics
    mfr_stats = df.groupby('acft_make').agg({
        'ev_id': 'count',
        'inj_tot_f': ['sum', 'mean'],
        'severity_encoded': 'mean'
    })

    mfr_stats.columns = ['_'.join(col).strip('_') for col in mfr_stats.columns]
    mfr_stats = mfr_stats.rename(columns={
        'ev_id_count': 'mfr_accident_count',
        'inj_tot_f_sum': 'mfr_total_fatalities',
        'inj_tot_f_mean': 'mfr_avg_fatalities',
        'severity_encoded_mean': 'mfr_avg_severity'
    })

    # Fatality rate
    mfr_stats['mfr_fatality_rate'] = mfr_stats['mfr_total_fatalities'] / mfr_stats['mfr_accident_count']

    # Merge back
    df = df.merge(mfr_stats, left_on='acft_make', right_index=True, how='left')

    return df
```

## Crew Features Engineering

### Pilot Experience and Certification

```python
def create_crew_features(df_events, df_crew):
    """
    Extract pilot experience and certification features.

    Args:
        df_events: Events table
        df_crew: Flight_Crew table (many-to-one with events)
    """

    # Filter to pilot-in-command (primary pilot)
    df_pic = df_crew[df_crew['crew_category'].isin(['PLT', 'COPL'])].copy()

    # Experience levels
    df_pic['pilot_experience_level'] = pd.cut(
        df_pic['pilot_tot_time'],
        bins=[0, 100, 500, 1500, 5000, 20000],
        labels=['student', 'low', 'medium', 'high', 'professional']
    )

    # Recent flight activity (currency)
    df_pic['recent_flight_ratio'] = df_pic['pilot_90_days'] / (df_pic['pilot_tot_time'] + 1)
    df_pic['is_current'] = (df_pic['pilot_90_days'] >= 10).astype(int)  # FAA currency

    # Type-specific experience
    df_pic['type_experience_ratio'] = df_pic['pilot_make_time'] / (df_pic['pilot_tot_time'] + 1)
    df_pic['is_type_proficient'] = (df_pic['pilot_make_time'] >= 100).astype(int)

    # Age categories
    df_pic['crew_age_group'] = pd.cut(
        df_pic['crew_age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['very_young', 'young', 'mid', 'senior', 'retirement', 'post_retirement']
    )

    # Medical certificate level (higher number = more restrictions)
    medical_map = {'1': 1, '2': 2, '3': 3, 'None': 4, 'Unknown': 3}
    df_pic['medical_cert_level'] = df_pic['pilot_med_class'].map(medical_map).fillna(3)

    # Certificate level (numeric encoding)
    cert_map = {
        'ATP': 5,
        'Commercial': 4,
        'Private': 3,
        'Recreational': 2,
        'Sport': 1,
        'Student': 0,
        'None': 0
    }
    df_pic['pilot_cert_level'] = df_pic['pilot_cert'].map(cert_map).fillna(2)

    # Aggregate to event level (take PIC values)
    crew_features = df_pic.groupby('ev_id').first()[
        ['pilot_tot_time', 'pilot_90_days', 'pilot_experience_level', 'recent_flight_ratio',
         'is_current', 'type_experience_ratio', 'is_type_proficient', 'crew_age',
         'crew_age_group', 'medical_cert_level', 'pilot_cert_level']
    ]

    # Merge to events
    df_events = df_events.merge(crew_features, left_on='ev_id', right_index=True, how='left')

    return df_events
```

## Text Features from Narratives

### TF-IDF Vectorization

```python
def create_tfidf_features(df, text_column='full_narrative', max_features=100):
    """
    Extract TF-IDF features from accident narratives.

    Args:
        df: DataFrame with narrative text
        text_column: Column containing text
        max_features: Number of TF-IDF features to extract
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Combine narrative fields
    if 'narr_accp' in df.columns:
        df['full_narrative'] = (
            df['narr_accp'].fillna('') + ' ' +
            df['narr_accf'].fillna('') + ' ' +
            df['narr_cause'].fillna('')
        )
        text_column = 'full_narrative'

    # TF-IDF vectorizer with aviation-specific settings
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,  # Ignore terms in fewer than 5 documents
        max_df=0.8,  # Ignore terms in more than 80% of documents
        sublinear_tf=True  # Use log scaling for TF
    )

    tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(''))

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{i}' for i in range(max_features)],
        index=df.index
    )

    # Concatenate with original DataFrame
    df = pd.concat([df, tfidf_df], axis=1)

    # Store vectorizer for production use
    return df, vectorizer
```

### Sentence Embeddings

```python
def create_embedding_features(df, text_column='full_narrative', n_components=50):
    """
    Create dense embeddings from narratives using sentence transformers.

    More powerful than TF-IDF but slower. Use for smaller datasets (<50K).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings

    # Encode narratives
    narratives = df[text_column].fillna('').tolist()
    embeddings = model.encode(narratives, show_progress_bar=True, batch_size=32)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Create DataFrame
    embed_df = pd.DataFrame(
        embeddings_reduced,
        columns=[f'embed_{i}' for i in range(n_components)],
        index=df.index
    )

    df = pd.concat([df, embed_df], axis=1)

    return df, model, pca
```

## Interaction Features

### Domain-Specific Interactions

```python
def create_interaction_features(df):
    """
    Create interaction features between important aviation variables.
    """

    # Weather × Phase interactions (critical risk factor)
    if 'is_imc' in df.columns and 'phase_critical' in df.columns:
        df['imc_critical_phase'] = df['is_imc'] * df['phase_critical']

    # Experience × Aircraft age (inexperienced pilots in old aircraft)
    if 'pilot_tot_time' in df.columns and 'aircraft_age' in df.columns:
        df['low_exp_old_aircraft'] = (
            (df['pilot_tot_time'] < 500).astype(int) *
            (df['aircraft_age'] > 30).astype(int)
        )

    # Night × Weather (compounding risk)
    # Note: Requires time-of-day feature (not in basic NTSB data)

    # Season × Region (regional weather patterns)
    if 'season' in df.columns and 'region' in df.columns:
        df['season_region'] = df['season'] + '_' + df['region']

    # Pilot currency × Type experience
    if 'is_current' in df.columns and 'type_experience_ratio' in df.columns:
        df['current_and_proficient'] = df['is_current'] * (df['type_experience_ratio'] > 0.1).astype(int)

    # Multi-engine × Engine failure (asymmetric thrust risk)
    if 'is_multi_engine' in df.columns and 'has_engine_failure' in df.columns:
        df['multi_eng_failure'] = df['is_multi_engine'] * df['has_engine_failure']

    return df
```

## Feature Selection Techniques

### Correlation-Based Selection

```python
def select_features_correlation(df, target_col, threshold=0.85):
    """
    Remove highly correlated features to reduce multicollinearity.
    """

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()

    # Select upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} highly correlated features (threshold={threshold})")

    return df.drop(columns=to_drop), to_drop
```

### Recursive Feature Elimination (RFE)

```python
def select_features_rfe(X, y, n_features=50):
    """
    Select top N features using Recursive Feature Elimination with Random Forest.
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    # Use Random Forest as estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # RFE selector
    selector = RFE(estimator, n_features_to_select=n_features, step=5)
    selector.fit(X, y)

    # Get selected features
    selected_features = X.columns[selector.support_].tolist()

    # Feature ranking
    feature_ranking = pd.DataFrame({
        'feature': X.columns,
        'ranking': selector.ranking_,
        'selected': selector.support_
    }).sort_values('ranking')

    print(f"Selected {len(selected_features)} features using RFE")
    print(f"\nTop 10 features:")
    print(feature_ranking.head(10))

    return selected_features, feature_ranking
```

### SHAP-Based Feature Selection

```python
def select_features_shap(model, X, y, top_k=50):
    """
    Select features based on SHAP importance values.

    Args:
        model: Trained tree-based model (XGBoost, RandomForest)
        X: Feature matrix
        y: Target variable
        top_k: Number of features to select
    """
    import shap

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values
    if isinstance(shap_values, list):  # Multi-class
        shap_importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

    # Create importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)

    # Select top-k features
    top_features = feature_importance.head(top_k)['feature'].tolist()

    print(f"Selected {len(top_features)} features based on SHAP importance")
    print(f"\nTop 10 features:")
    print(feature_importance.head(10))

    return top_features, feature_importance
```

## Feature Store Design

### Feast Feature Store Setup

```python
# feature_store/features.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Int64, Float32, Bool
from datetime import timedelta

# Define entity (primary key)
event = Entity(
    name="event",
    join_keys=["ev_id"],
    description="Aviation accident event"
)

# Define feature view
accident_features = FeatureView(
    name="accident_features",
    entities=[event],
    ttl=timedelta(days=365 * 10),  # 10 years retention
    schema=[
        # Temporal
        Field(name="year", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="season", dtype=String),
        Field(name="is_weekend", dtype=Bool),

        # Geospatial
        Field(name="dec_latitude", dtype=Float32),
        Field(name="dec_longitude", dtype=Float32),
        Field(name="region", dtype=String),
        Field(name="dist_to_nearest_airport_km", dtype=Float32),

        # Aircraft
        Field(name="aircraft_age", dtype=Int64),
        Field(name="aircraft_category_risk", dtype=Float32),
        Field(name="is_multi_engine", dtype=Bool),

        # Crew
        Field(name="pilot_tot_time", dtype=Int64),
        Field(name="pilot_experience_level", dtype=String),
        Field(name="is_current", dtype=Bool),

        # Occurrences
        Field(name="has_engine_failure", dtype=Bool),
        Field(name="is_cfit", dtype=Bool),
        Field(name="phase_critical", dtype=Bool),

        # Target
        Field(name="severity", dtype=String),
    ],
    source=FileSource(path="data/features.parquet"),
)

# Usage in feature_store/feature_store.yaml
# project: ntsb_aviation
# registry: data/registry.db
# provider: local
# online_store:
#   type: sqlite
#   path: data/online_store.db
```

## Complete Feature Engineering Pipeline

### Production Pipeline Class

```python
class AviationFeatureEngineer:
    """
    Complete feature engineering pipeline for NTSB aviation data.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.fitted = False
        self.scaler = None
        self.encoders = {}

    def fit_transform(self, df_events, df_aircraft=None, df_crew=None,
                     df_findings=None, airports_df=None):
        """
        Apply all feature engineering steps.

        Returns:
            DataFrame with engineered features
        """

        print("Starting feature engineering pipeline...")

        # 1. Domain-specific NTSB codes
        print("  [1/8] Extracting NTSB code features...")
        if 'occurrence_code' in df_events.columns:
            df_events = extract_occurrence_features(df_events)
        if 'phase_code' in df_events.columns:
            df_events = extract_phase_features(df_events)

        # 2. Temporal features
        print("  [2/8] Creating temporal features...")
        if 'ev_date' in df_events.columns:
            df_events = create_temporal_features(df_events)

        # 3. Geospatial features
        print("  [3/8] Creating geospatial features...")
        if all(col in df_events.columns for col in ['dec_latitude', 'dec_longitude']):
            df_events = create_geospatial_features(df_events)

            if airports_df is not None:
                df_events = create_distance_features(df_events, airports_df)

        # 4. Aircraft features
        print("  [4/8] Creating aircraft features...")
        if df_aircraft is not None:
            # Merge aircraft data
            df_events = df_events.merge(
                df_aircraft[['ev_id', 'acft_make', 'acft_model', 'acft_year', 'acft_category',
                            'num_eng', 'eng_type', 'cert_max_gr_wt', 'eng_hp_or_lbs']],
                on='ev_id',
                how='left'
            )

        if 'acft_year' in df_events.columns:
            df_events = create_aircraft_features(df_events)
            df_events = create_manufacturer_features(df_events)

        # 5. Crew features
        print("  [5/8] Creating crew features...")
        if df_crew is not None:
            df_events = create_crew_features(df_events, df_crew)

        # 6. Cause features
        print("  [6/8] Aggregating cause features...")
        if df_findings is not None:
            df_events = extract_cause_features(df_events, df_findings)

        # 7. Interaction features
        print("  [7/8] Creating interaction features...")
        df_events = create_interaction_features(df_events)

        # 8. Feature selection (correlation-based)
        print("  [8/8] Removing highly correlated features...")
        if 'severity' in df_events.columns:
            df_events, dropped = select_features_correlation(df_events, 'severity', threshold=0.9)

        self.fitted = True
        print(f"Feature engineering complete. Final shape: {df_events.shape}")

        return df_events

    def transform(self, df):
        """Transform new data using fitted pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform()")

        # Apply same transformations (without fitting)
        # Implementation details omitted for brevity
        pass
```

## Performance Optimization

### Using Polars for Speed

```python
import polars as pl

def extract_features_polars(df_path):
    """
    Use Polars for 10x faster feature engineering on large datasets.
    """

    # Read with Polars (lazy evaluation)
    df = pl.scan_parquet(df_path)

    # Temporal features (vectorized)
    df = df.with_columns([
        pl.col('ev_date').dt.year().alias('year'),
        pl.col('ev_date').dt.month().alias('month'),
        pl.col('ev_date').dt.day().alias('day'),
        (pl.col('ev_date').dt.month().is_in([12, 1, 2])).alias('is_winter'),
        (pl.col('ev_date').dt.weekday() >= 5).alias('is_weekend'),
    ])

    # Occurrence features
    df = df.with_columns([
        pl.col('occurrence_code').is_between(100, 119).alias('has_engine_failure'),
        pl.col('occurrence_code').is_between(300, 370).alias('is_collision'),
        (pl.col('occurrence_code') == 430).alias('is_loss_of_control'),
    ])

    # Collect (execute lazy operations)
    result = df.collect()

    return result.to_pandas()  # Convert back to pandas if needed
```

---

## References

**Research Papers:**
- Rose et al. (2024). "NLP in Aviation Safety: Systematic Review." MDPI Aerospace.
- Liu et al. (2025). "ML Anomaly Detection in Commercial Aircraft." PLOS ONE.
- Baugh (2021). "Predicting General Aviation Accidents Using ML." ERAU Dissertation.

**Documentation:**
- NTSB Coding Manual: `ref_docs/codman.pdf`
- Database Schema: `DATA_DICTIONARY.md`
- Aviation Codes: `AVIATION_CODING_LEXICON.md`

**Tools:**
- Scikit-learn feature engineering: https://scikit-learn.org/stable/modules/preprocessing.html
- Polars documentation: https://pola-rs.github.io/polars/
- Feast feature store: https://feast.dev/

---

**Last Updated:** January 2025
**Version:** 1.0.0
**Next:** See `MODEL_DEPLOYMENT_GUIDE.md` for deploying models with engineered features
