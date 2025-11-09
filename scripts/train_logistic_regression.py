#!/usr/bin/env python3
"""
Logistic Regression Model for Fatal Outcome Prediction

Binary classification to predict fatal vs non-fatal aviation accidents.

Phase 2 Sprint 6-7: Statistical Modeling & ML Preparation
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_features() -> pd.DataFrame:
    """Load engineered features."""
    print("Loading features...")
    df = pd.read_parquet("data/ml_features.parquet")
    print(f"  Loaded {len(df):,} events with {len(df.columns)} features")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple:
    """Prepare data for logistic regression."""
    print("\nPreparing data...")

    # Separate features and target
    target = "fatal_outcome"
    identifiers = ["ev_id", "ntsb_no", "ev_date"]
    other_targets = ["severity_level", "finding_code_grouped"]

    # Features to exclude
    exclude = identifiers + other_targets + [target]
    feature_cols = [col for col in df.columns if col not in exclude]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Convert to numeric
    X = X.astype(float)

    print(f"  Features: {X.shape[1]}")
    print(f"  Categorical features encoded: {len(categorical_cols)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print(f"  Fatal rate: {y.mean():.2%}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoders,
        feature_cols,
    )


def train_model(X_train, y_train) -> Tuple:
    """Train logistic regression with hyperparameter tuning."""
    print("\nTraining logistic regression...")

    # Hyperparameter grid
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "class_weight": ["balanced"],
        "max_iter": [1000],
    }

    # Base model
    base_model = LogisticRegression(random_state=RANDOM_STATE)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    print("  Running grid search (5-fold CV)...")
    grid_search.fit(X_train, y_train)

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best ROC-AUC (CV): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols) -> Dict:
    """Evaluate model performance."""
    print("\nEvaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred),
            "roc_auc": roc_auc_score(y_train, y_train_proba),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
        },
    }

    print("\n  Training Metrics:")
    for metric, value in metrics["train"].items():
        print(f"    {metric}: {value:.4f}")

    print("\n  Test Metrics:")
    for metric, value in metrics["test"].items():
        print(f"    {metric}: {value:.4f}")

    # Classification report
    print("\n  Classification Report (Test Set):")
    print(
        classification_report(y_test, y_test_pred, target_names=["Non-Fatal", "Fatal"])
    )

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)

    print("\n  Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    return metrics, y_test_pred, y_test_proba, feature_importance


def create_visualizations(
    y_test, y_test_pred, y_test_proba, feature_importance
) -> None:
    """Create model evaluation visualizations."""
    print("\nCreating visualizations...")

    figures_dir = Path("notebooks/modeling/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    axes[0, 0].plot(
        fpr, tpr, color="#e74c3c", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    axes[0, 0].plot(
        [0, 1], [0, 1], color="#7f8c8d", lw=2, linestyle="--", label="Random"
    )
    axes[0, 0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0, 0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0, 0].set_title(
        "ROC Curve - Logistic Regression", fontsize=14, fontweight="bold"
    )
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 1],
        xticklabels=["Non-Fatal", "Fatal"],
        yticklabels=["Non-Fatal", "Fatal"],
    )
    axes[0, 1].set_xlabel("Predicted", fontsize=12)
    axes[0, 1].set_ylabel("Actual", fontsize=12)
    axes[0, 1].set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Feature Importance (Top 20)
    top_20_features = feature_importance.head(20)
    colors = ["#e74c3c" if x > 0 else "#3498db" for x in top_20_features["coefficient"]]

    axes[1, 0].barh(
        range(len(top_20_features)), top_20_features["coefficient"], color=colors
    )
    axes[1, 0].set_yticks(range(len(top_20_features)))
    axes[1, 0].set_yticklabels(top_20_features["feature"], fontsize=9)
    axes[1, 0].set_xlabel("Coefficient", fontsize=12)
    axes[1, 0].set_title("Feature Importance (Top 20)", fontsize=14, fontweight="bold")
    axes[1, 0].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # Prediction Distribution
    axes[1, 1].hist(
        y_test_proba[y_test == 0],
        bins=50,
        alpha=0.6,
        label="Non-Fatal",
        color="#2ecc71",
    )
    axes[1, 1].hist(
        y_test_proba[y_test == 1], bins=50, alpha=0.6, label="Fatal", color="#e74c3c"
    )
    axes[1, 1].set_xlabel("Predicted Probability", fontsize=12)
    axes[1, 1].set_ylabel("Frequency", fontsize=12)
    axes[1, 1].set_title(
        "Prediction Probability Distribution", fontsize=14, fontweight="bold"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "03_logistic_regression_evaluation.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("  Saved: 03_logistic_regression_evaluation.png")


def save_model(
    model, scaler, label_encoders, feature_cols, metrics, best_params
) -> None:
    """Save trained model and metadata."""
    print("\nSaving model...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model artifacts
    model_artifacts = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_cols,
    }

    model_path = models_dir / "logistic_regression.pkl"
    joblib.dump(model_artifacts, model_path)
    print(f"  Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "LogisticRegression",
        "trained_at": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "hyperparameters": best_params,
        "performance": metrics,
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
    }

    metadata_path = models_dir / "logistic_regression_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Logistic Regression for Fatal Outcome Prediction")
    print("Phase 2 Sprint 6-7: Statistical Modeling & ML Preparation")
    print("=" * 80)

    # Load features
    df = load_features()

    # Prepare data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoders,
        feature_cols,
    ) = prepare_data(df)

    # Train model
    model, best_params = train_model(X_train, y_train)

    # Evaluate model
    metrics, y_test_pred, y_test_proba, feature_importance = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_cols
    )

    # Create visualizations
    create_visualizations(y_test, y_test_pred, y_test_proba, feature_importance)

    # Save model
    save_model(model, scaler, label_encoders, feature_cols, metrics, best_params)

    print("\n" + "=" * 80)
    print("Logistic Regression Training Complete!")
    print("=" * 80)
    print("\nModel Performance Summary:")
    print(f"  Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"  Test F1-Score: {metrics['test']['f1']:.4f}")
    print("\nModel artifacts saved to models/")
    print("Visualizations saved to notebooks/modeling/figures/")


if __name__ == "__main__":
    main()
