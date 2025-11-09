#!/usr/bin/env python3
"""
Random Forest Classifier for Cause Prediction

Multi-class classification to predict primary finding codes (accident causes).

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
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    """Prepare data for random forest."""
    print("\nPreparing data...")

    # Target variable
    target = "finding_code_grouped"
    identifiers = ["ev_id", "ntsb_no", "ev_date"]
    other_targets = ["severity_level", "fatal_outcome"]

    # Exclude finding-related features to avoid data leakage
    exclude = identifiers + other_targets + [target, "primary_finding_code"]

    # Features to use
    feature_cols = [col for col in df.columns if col not in exclude]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    X = X.astype(float)

    print(f"  Features: {X.shape[1]}")
    print(f"  Categorical features encoded: {len(categorical_cols)}")
    print(f"  Target classes: {len(target_encoder.classes_)}")
    print("  Class distribution (top 5):")
    top_5_classes = pd.Series(y).value_counts().head()
    for cls, count in top_5_classes.items():
        print(f"    {cls}: {count:,} ({count/len(y)*100:.2f}%)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"\n  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoders,
        target_encoder,
        feature_cols,
    )


def train_model(X_train, y_train) -> Tuple:
    """Train random forest with hyperparameter tuning."""
    print("\nTraining random forest...")
    print("  This may take 5-10 minutes with hyperparameter tuning...")

    # Hyperparameter grid
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
        "bootstrap": [True],
    }

    # Base model
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Randomized search (faster than grid search)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=20,  # 20 random combinations
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    print("  Running randomized search (3-fold CV, 20 iterations)...")
    random_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {random_search.best_params_}")
    print(f"  Best F1-Macro (CV): {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def evaluate_model(
    model, X_train, X_test, y_train, y_test, target_encoder, feature_cols
) -> Dict:
    """Evaluate model performance."""
    print("\nEvaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision_macro": precision_score(
                y_train, y_train_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_train, y_train_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_train, y_train_pred, average="macro", zero_division=0
            ),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision_macro": precision_score(
                y_test, y_test_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, y_test_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        },
    }

    print("\n  Training Metrics:")
    for metric, value in metrics["train"].items():
        print(f"    {metric}: {value:.4f}")

    print("\n  Test Metrics:")
    for metric, value in metrics["test"].items():
        print(f"    {metric}: {value:.4f}")

    # Classification report (top 10 classes)
    print("\n  Classification Report (Top 10 Classes, Test Set):")
    top_10_classes = pd.Series(y_test).value_counts().head(10).index.tolist()
    y_test_top10 = [y if y in top_10_classes else -1 for y in y_test]
    y_test_pred_top10 = [y if y in top_10_classes else -1 for y in y_test_pred]

    # Get class names
    class_names_full = target_encoder.classes_
    top_10_class_names = [class_names_full[i] for i in top_10_classes]

    # Filter for report
    mask = np.array(y_test_top10) != -1
    if mask.sum() > 0:
        print(
            classification_report(
                np.array(y_test_top10)[mask],
                np.array(y_test_pred_top10)[mask],
                labels=top_10_classes,
                target_names=top_10_class_names,
                zero_division=0,
            )
        )

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n  Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    return metrics, y_test_pred, feature_importance, top_10_classes


def create_visualizations(
    y_test, y_test_pred, feature_importance, top_10_classes, target_encoder
) -> None:
    """Create model evaluation visualizations."""
    print("\nCreating visualizations...")

    figures_dir = Path("notebooks/modeling/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Confusion Matrix (Top 10 classes)
    top_10_mask = np.isin(y_test, top_10_classes)
    y_test_top10 = y_test[top_10_mask]
    y_test_pred_top10 = y_test_pred[top_10_mask]

    cm = confusion_matrix(y_test_top10, y_test_pred_top10, labels=top_10_classes)

    class_names = target_encoder.classes_
    top_10_labels = [class_names[i][:10] for i in top_10_classes]  # Truncate labels

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=top_10_labels,
        yticklabels=top_10_labels,
        cbar_kws={"label": "Count"},
    )
    axes[0, 0].set_xlabel("Predicted", fontsize=12)
    axes[0, 0].set_ylabel("Actual", fontsize=12)
    axes[0, 0].set_title(
        "Confusion Matrix (Top 10 Classes)", fontsize=14, fontweight="bold"
    )
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].tick_params(axis="y", rotation=0)

    # Feature Importance (Top 20)
    top_20_features = feature_importance.head(20)
    axes[0, 1].barh(
        range(len(top_20_features)), top_20_features["importance"], color="#3498db"
    )
    axes[0, 1].set_yticks(range(len(top_20_features)))
    axes[0, 1].set_yticklabels(top_20_features["feature"], fontsize=9)
    axes[0, 1].set_xlabel("Importance", fontsize=12)
    axes[0, 1].set_title("Feature Importance (Top 20)", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # Class Distribution
    class_dist = pd.Series(y_test).value_counts().head(15)
    class_labels = [class_names[i][:15] for i in class_dist.index]

    axes[1, 0].bar(
        range(len(class_dist)), class_dist.values, color=sns.color_palette("Set2")
    )
    axes[1, 0].set_xticks(range(len(class_dist)))
    axes[1, 0].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_xlabel("Finding Code", fontsize=12)
    axes[1, 0].set_ylabel("Count", fontsize=12)
    axes[1, 0].set_title(
        "Test Set Class Distribution (Top 15)", fontsize=14, fontweight="bold"
    )
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Prediction Accuracy by Class (Top 10)
    accuracy_by_class = []
    for cls in top_10_classes:
        mask = y_test == cls
        if mask.sum() > 0:
            acc = (y_test_pred[mask] == cls).mean()
            accuracy_by_class.append(acc)
        else:
            accuracy_by_class.append(0)

    axes[1, 1].bar(range(len(top_10_classes)), accuracy_by_class, color="#e74c3c")
    axes[1, 1].set_xticks(range(len(top_10_classes)))
    axes[1, 1].set_xticklabels(
        [class_names[i][:10] for i in top_10_classes],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    axes[1, 1].set_xlabel("Finding Code", fontsize=12)
    axes[1, 1].set_ylabel("Accuracy", fontsize=12)
    axes[1, 1].set_title(
        "Prediction Accuracy by Class (Top 10)", fontsize=14, fontweight="bold"
    )
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "04_random_forest_evaluation.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("  Saved: 04_random_forest_evaluation.png")


def save_model(
    model, label_encoders, target_encoder, feature_cols, metrics, best_params
) -> None:
    """Save trained model and metadata."""
    print("\nSaving model...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model artifacts
    model_artifacts = {
        "model": model,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "feature_names": feature_cols,
    }

    model_path = models_dir / "random_forest.pkl"
    joblib.dump(model_artifacts, model_path)
    print(f"  Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "trained_at": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "hyperparameters": best_params,
        "performance": metrics,
        "num_features": len(feature_cols),
        "num_classes": len(target_encoder.classes_),
        "feature_names": feature_cols,
        "class_names": target_encoder.classes_.tolist(),
    }

    metadata_path = models_dir / "random_forest_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Random Forest for Cause Prediction (Finding Codes)")
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
        label_encoders,
        target_encoder,
        feature_cols,
    ) = prepare_data(df)

    # Train model
    model, best_params = train_model(X_train, y_train)

    # Evaluate model
    metrics, y_test_pred, feature_importance, top_10_classes = evaluate_model(
        model, X_train, X_test, y_train, y_test, target_encoder, feature_cols
    )

    # Create visualizations
    create_visualizations(
        y_test, y_test_pred, feature_importance, top_10_classes, target_encoder
    )

    # Save model
    save_model(
        model, label_encoders, target_encoder, feature_cols, metrics, best_params
    )

    print("\n" + "=" * 80)
    print("Random Forest Training Complete!")
    print("=" * 80)
    print("\nModel Performance Summary:")
    print(f"  Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"  Test F1-Macro: {metrics['test']['f1_macro']:.4f}")
    print(f"  Test Precision-Macro: {metrics['test']['precision_macro']:.4f}")
    print(f"  Test Recall-Macro: {metrics['test']['recall_macro']:.4f}")
    print("\nModel artifacts saved to models/")
    print("Visualizations saved to notebooks/modeling/figures/")


if __name__ == "__main__":
    main()
