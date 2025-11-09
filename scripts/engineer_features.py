#!/usr/bin/env python3
"""
Feature Engineering Script for NTSB Aviation Accident ML Models

This script transforms raw database features into ML-ready features for:
- Binary classification (fatal vs non-fatal outcome)
- Multi-class classification (injury severity levels)
- Cause prediction (finding codes)

Phase 2 Sprint 6-7: Statistical Modeling & ML Preparation
"""

import warnings
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_raw_features() -> pd.DataFrame:
    """Load raw features from temporary parquet file."""
    print("Loading raw features...")
    df = pd.read_parquet("data/raw_features_temp.parquet")
    print(f"  Loaded {len(df):,} events with {len(df.columns)} features")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or flag missing values."""
    print("\nHandling missing values...")
    df = df.copy()

    # Categorical features - fill with 'UNKNOWN'
    categorical_cols = [
        "ev_state",
        "acft_make",
        "acft_model",
        "acft_category",
        "acft_damage",
        "flight_phase",
        "wx_cond_basic",
        "pilot_cert",
        "far_part",
        "flight_plan_filed",
        "flight_activity",
        "ev_dow",
        "season",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # Numeric features - fill with median or 0
    numeric_impute = {
        "crew_age": df["crew_age"].median(),
        "pilot_tot_time": 0,
        "pilot_90_days": 0,
        "num_eng": 1,
        "wx_temp": df["wx_temp"].median(),
        "wx_wind_speed": df["wx_wind_speed"].median(),
        "wx_vis": df["wx_vis"].median(),
    }

    for col, value in numeric_impute.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Geographic - create missing flag
    df["has_coordinates"] = (
        df["dec_latitude"].notna() & df["dec_longitude"].notna()
    ).astype(int)

    df["dec_latitude"] = df["dec_latitude"].fillna(0)
    df["dec_longitude"] = df["dec_longitude"].fillna(0)

    # Finding code - fill with 'UNKNOWN'
    df["primary_finding_code"] = df["primary_finding_code"].fillna("99999")

    print(f"  Categorical features: {len(categorical_cols)} filled with 'UNKNOWN'")
    print(f"  Numeric features: {len(numeric_impute)} imputed")
    print("  Geographic flags: has_coordinates created")

    return df


def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binned versions of continuous features."""
    print("\nCreating binned features...")
    df = df.copy()

    # Age bins
    df["age_group"] = pd.cut(
        df["crew_age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
    ).astype(str)

    # Experience bins (total hours)
    df["experience_level"] = pd.cut(
        df["pilot_tot_time"],
        bins=[-1, 100, 500, 1000, 5000, np.inf],
        labels=["<100hrs", "100-500hrs", "500-1000hrs", "1000-5000hrs", "5000+hrs"],
    ).astype(str)

    # Recent flight hours (90-day)
    df["recent_activity"] = pd.cut(
        df["pilot_90_days"],
        bins=[-1, 10, 50, 100, np.inf],
        labels=["<10hrs", "10-50hrs", "50-100hrs", "100+hrs"],
    ).astype(str)

    # Temperature bins
    df["temp_category"] = pd.cut(
        df["wx_temp"],
        bins=[-np.inf, 32, 60, 80, np.inf],
        labels=["Cold", "Cool", "Moderate", "Hot"],
    ).astype(str)

    # Visibility bins
    df["visibility_category"] = pd.cut(
        df["wx_vis"],
        bins=[-1, 1, 3, 10, np.inf],
        labels=["Low", "Moderate", "Good", "Excellent"],
    ).astype(str)

    print("  Created 5 binned features")
    return df


def encode_aircraft_make(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Encode aircraft make as top N + 'Other'."""
    print(f"\nEncoding aircraft make (top {top_n})...")
    df = df.copy()

    top_makes = df["acft_make"].value_counts().head(top_n).index.tolist()
    df["acft_make_grouped"] = df["acft_make"].apply(
        lambda x: x if x in top_makes else "OTHER"
    )

    other_count = len(df[df["acft_make_grouped"] == "OTHER"])
    print(f"  {other_count:,} events grouped as 'OTHER'")

    return df


def encode_finding_codes(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Encode primary finding codes as top N + 'Other'."""
    print(f"\nEncoding finding codes (top {top_n})...")
    df = df.copy()

    top_codes = df["primary_finding_code"].value_counts().head(top_n).index.tolist()
    df["finding_code_grouped"] = df["primary_finding_code"].apply(
        lambda x: x if x in top_codes else "OTHER"
    )

    other_count = len(df[df["finding_code_grouped"] == "OTHER"])
    print(f"  {other_count:,} events grouped as 'OTHER'")

    return df


def encode_damage_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Encode aircraft damage as ordinal severity."""
    print("\nEncoding damage severity...")
    df = df.copy()

    damage_map = {"DEST": 4, "SUBS": 3, "MINR": 2, "NONE": 1, "UNKNOWN": 0}

    df["damage_severity"] = df["acft_damage"].map(damage_map).fillna(0).astype(int)
    print("  Damage encoded (0=Unknown, 1=None, 2=Minor, 3=Substantial, 4=Destroyed)")

    return df


def assign_region(df: pd.DataFrame) -> pd.DataFrame:
    """Assign US census region based on state."""
    print("\nAssigning geographic regions...")
    df = df.copy()

    regions = {
        "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
        "Midwest": [
            "IL",
            "IN",
            "MI",
            "OH",
            "WI",
            "IA",
            "KS",
            "MN",
            "MO",
            "NE",
            "ND",
            "SD",
        ],
        "South": [
            "DE",
            "FL",
            "GA",
            "MD",
            "NC",
            "SC",
            "VA",
            "WV",
            "AL",
            "KY",
            "MS",
            "TN",
            "AR",
            "LA",
            "OK",
            "TX",
        ],
        "West": [
            "AZ",
            "CO",
            "ID",
            "MT",
            "NV",
            "NM",
            "UT",
            "WY",
            "AK",
            "CA",
            "HI",
            "OR",
            "WA",
        ],
    }

    state_to_region = {}
    for region, states in regions.items():
        for state in states:
            state_to_region[state] = region

    df["region"] = df["ev_state"].map(state_to_region).fillna("Other")
    print(f"  Regions assigned: {df['region'].nunique()} unique regions")

    return df


def create_severity_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-class injury severity target variable."""
    print("\nCreating severity levels...")
    df = df.copy()

    def classify_severity(row):
        if row["total_fatalities"] > 0:
            return "FATAL"
        elif row["total_serious_injuries"] > 0:
            return "SERIOUS"
        elif row["total_minor_injuries"] > 0:
            return "MINOR"
        else:
            return "NONE"

    df["severity_level"] = df.apply(classify_severity, axis=1)

    fatal_rate = (df["severity_level"] == "FATAL").mean()
    print(f"  Severity levels created (Fatal rate: {fatal_rate:.2%})")

    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature set for modeling."""
    print("\nSelecting final features...")

    feature_groups = {
        "temporal": ["ev_year", "ev_month", "day_of_week", "season"],
        "geographic": [
            "ev_state",
            "region",
            "dec_latitude",
            "dec_longitude",
            "has_coordinates",
        ],
        "aircraft": [
            "acft_make_grouped",
            "acft_category",
            "damage_severity",
            "num_eng",
            "far_part",
        ],
        "operational": [
            "flight_phase",
            "wx_cond_basic",
            "temp_category",
            "visibility_category",
            "flight_plan_filed",
            "flight_activity",
        ],
        "crew": ["age_group", "pilot_cert", "experience_level", "recent_activity"],
        "targets": ["fatal_outcome", "severity_level", "finding_code_grouped"],
        "identifiers": ["ev_id", "ntsb_no", "ev_date"],
    }

    all_features = []
    for group, features in feature_groups.items():
        all_features.extend(features)

    ml_df = df[all_features].copy()

    print(
        f"  Selected {len(all_features)} features across {len(feature_groups)} groups"
    )
    print(f"  Dataset shape: {ml_df.shape}")
    print(f"  Memory usage: {ml_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return ml_df, feature_groups


def create_visualizations(df: pd.DataFrame) -> None:
    """Create feature statistics visualizations."""
    print("\nCreating visualizations...")

    figures_dir = Path("notebooks/modeling/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Target variable distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fatal outcome
    df["fatal_outcome"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"]
    )
    axes[0].set_title("Fatal Outcome Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Fatal Outcome (0=No, 1=Yes)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Severity level
    df["severity_level"].value_counts().plot(
        kind="bar", ax=axes[1], color=sns.color_palette("Set2")
    )
    axes[1].set_title("Injury Severity Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Severity Level")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "01_target_variable_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("  Saved: 01_target_variable_distribution.png")

    # Feature correlation with fatal outcome
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Damage severity vs fatal outcome
    fatal_by_damage = df.groupby("damage_severity")["fatal_outcome"].mean() * 100
    fatal_by_damage.plot(kind="bar", ax=axes[0, 0], color="#e74c3c")
    axes[0, 0].set_title(
        "Fatal Rate by Damage Severity", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Damage Severity")
    axes[0, 0].set_ylabel("Fatal Rate (%)")
    axes[0, 0].tick_params(axis="x", rotation=0)

    # Weather condition vs fatal outcome
    fatal_by_wx = df.groupby("wx_cond_basic")["fatal_outcome"].mean() * 100
    fatal_by_wx.plot(kind="bar", ax=axes[0, 1], color="#3498db")
    axes[0, 1].set_title(
        "Fatal Rate by Weather Condition", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Weather Condition")
    axes[0, 1].set_ylabel("Fatal Rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Flight phase vs fatal outcome (top 10)
    top_phases = df["flight_phase"].value_counts().head(10).index
    fatal_by_phase = (
        df[df["flight_phase"].isin(top_phases)]
        .groupby("flight_phase")["fatal_outcome"]
        .mean()
        * 100
    )
    fatal_by_phase.sort_values().plot(kind="barh", ax=axes[1, 0], color="#9b59b6")
    axes[1, 0].set_title(
        "Fatal Rate by Flight Phase (Top 10)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Fatal Rate (%)")
    axes[1, 0].set_ylabel("Flight Phase")

    # Region vs fatal outcome
    fatal_by_region = df.groupby("region")["fatal_outcome"].mean() * 100
    fatal_by_region.plot(kind="bar", ax=axes[1, 1], color="#e67e22")
    axes[1, 1].set_title("Fatal Rate by Region", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Region")
    axes[1, 1].set_ylabel("Fatal Rate (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "02_fatal_rate_by_features.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("  Saved: 02_fatal_rate_by_features.png")


def save_features(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
    """Save engineered features and metadata."""
    print("\nSaving features...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save to parquet
    output_path = data_dir / "ml_features.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"  Features saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")

    # Save metadata
    import json

    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_samples": len(df),
        "num_features": len(df.columns),
        "date_range": {
            "start": df["ev_date"].min().isoformat(),
            "end": df["ev_date"].max().isoformat(),
        },
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
        "target_distributions": {
            "fatal_outcome": df["fatal_outcome"].value_counts().to_dict(),
            "severity_level": df["severity_level"].value_counts().to_dict(),
        },
        "missing_values": df.isnull().sum().to_dict(),
    }

    metadata_path = data_dir / "ml_features_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Metadata saved to: {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Feature Engineering for Aviation Accident Prediction")
    print("Phase 2 Sprint 6-7: Statistical Modeling & ML Preparation")
    print("=" * 80)

    # Load raw features
    df = load_raw_features()

    # Apply transformations
    df = handle_missing_values(df)
    df = create_binned_features(df)
    df = encode_aircraft_make(df, top_n=20)
    df = encode_finding_codes(df, top_n=30)
    df = encode_damage_severity(df)
    df = assign_region(df)
    df = create_severity_levels(df)

    # Select final features
    ml_df, feature_groups = select_final_features(df)

    # Create visualizations
    create_visualizations(ml_df)

    # Save features
    save_features(ml_df, feature_groups)

    print("\n" + "=" * 80)
    print("Feature engineering complete!")
    print("=" * 80)
    print("\nDataset Summary:")
    print(f"  Events: {len(ml_df):,}")
    print(f"  Features: {len(ml_df.columns)}")
    print(f"  Date range: {ml_df['ev_date'].min()} to {ml_df['ev_date'].max()}")
    print(f"  Fatal rate: {ml_df['fatal_outcome'].mean():.2%}")
    print("\nNext steps:")
    print("  1. Logistic Regression (notebooks/modeling/01_logistic_regression.ipynb)")
    print(
        "  2. Cox Proportional Hazards (notebooks/modeling/02_cox_proportional_hazards.ipynb)"
    )
    print(
        "  3. Random Forest (notebooks/modeling/03_random_forest_classification.ipynb)"
    )


if __name__ == "__main__":
    main()
