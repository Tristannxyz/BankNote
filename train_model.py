"""
train_model.py — Training pipeline for NoteGuard Banknote Authenticity Classifier.
Trains Decision Tree and Random Forest models on data/banknote_authenticity.csv
(derived directly from the real banknote images in dataset/).

Features:
  - 4 Wavelet features: variance, skewness, curtosis, entropy (Haar DWT 'db1')
  - 6 Color statistics: r_mean, g_mean, b_mean, r_std, g_std, b_std

Target:
  - class: 0 = Genuine, 1 = Forged
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "banknote_authenticity.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_NAMES = [
    "variance", "skewness", "curtosis", "entropy",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"
]
WAVELET_ONLY_FEATURES = ["variance", "skewness", "curtosis", "entropy"]
TARGET_NAME = "class"


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_CSV):
        raise RuntimeError(f"Authenticity dataset not found at {DATA_CSV}. Run build_authenticity_dataset.py first.")

    df = pd.read_csv(DATA_CSV)
    print("\n" + "=" * 60)
    print("1. DATA VALIDATION & EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"Dataset path: {DATA_CSV}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Missing values
    missing = int(df.isnull().sum().sum())
    print(f"Missing values: {missing}")

    # Class balance
    class_counts = df[TARGET_NAME].value_counts().to_dict()
    genuine_count = int(class_counts.get(0, 0))
    forged_count = int(class_counts.get(1, 0))
    print(f"Class distribution: Genuine (0) = {genuine_count} ({genuine_count/len(df)*100:.1f}%), "
          f"Forged (1) = {forged_count} ({forged_count/len(df)*100:.1f}%)")

    dataset_summary = {
        "dataset_source": "dataset/ (Philippine Banknotes YOLO dataset crops + realistic counterfeit simulations)",
        "total_records": len(df),
        "genuine_count": genuine_count,
        "forged_count": forged_count,
        "missing_values": missing,
        "target_mapping": {"0": "Genuine", "1": "Forged"}
    }

    # Features and Target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # Grouped 80/20 train/test split — prevents data leakage between
    # crop and full-photo versions of the same source image
    if "source_image_id" in df.columns:
        groups = df["source_image_id"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"\n[GroupShuffleSplit] Split by source_image_id to prevent leakage.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"\n[Fallback] Using stratified random split (no source_image_id column).")
    print(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")
    print(f"Train class dist: {y_train.value_counts().to_dict()}")
    print(f"Test  class dist: {y_test.value_counts().to_dict()}")

    # Feature statistics on training set for out-of-distribution checks
    feature_stats = {}
    for col in FEATURE_NAMES:
        feature_stats[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std())
        }
    print("\nTraining set feature statistics:")
    for col, st in feature_stats.items():
        print(f"  {col:<10}: min={st['min']:7.4f}, max={st['max']:7.4f}, "
              f"mean={st['mean']:7.4f}, std={st['std']:7.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Decision Tree Training with GridSearchCV
    print("\n" + "=" * 60)
    print("2. TRAINING DECISION TREE CLASSIFIER")
    print("=" * 60)
    dt_params = {
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    dt_grid = GridSearchCV(
        # class_weight='balanced' compensates for genuine >> counterfeit samples
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        dt_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    print(f"Best Decision Tree params: {dt_grid.best_params_}")
    print(f"Best CV F1-score: {dt_grid.best_score_:.4f}")

    # 3. Random Forest Training with GridSearchCV
    print("\n" + "=" * 60)
    print("3. TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    rf_params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [6, 10, 15, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
    }
    rf_grid = GridSearchCV(
        # class_weight='balanced' compensates for genuine >> counterfeit samples
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        rf_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best Random Forest params: {rf_grid.best_params_}")
    print(f"Best CV F1-score: {rf_grid.best_score_:.4f}")

    # 4. Evaluation on Test Set (Forged = positive class 1)
    models = {
        "decision_tree": (best_dt, "Decision Tree"),
        "random_forest": (best_rf, "Random Forest")
    }

    metrics_results = {}
    print("\n" + "=" * 60)
    print("4. TEST SET EVALUATION (Positive class = 1 [Forged])")
    print("=" * 60)

    for key, (model, display_name) in models.items():
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        importances = [float(val) for val in model.feature_importances_]
        feat_imp_dict = dict(zip(FEATURE_NAMES, importances))

        metrics_results[key] = {
            "model_name": display_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "feature_importances": feat_imp_dict,
            "best_params": dt_grid.best_params_ if key == "decision_tree" else rf_grid.best_params_
        }

        print(f"\nModel: {display_name}")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  F1-Score : {f1 * 100:.2f}%")
        print(f"  Confusion Matrix (Actual [0:Gen, 1:Forg] vs Pred [0:Gen, 1:Forg]):")
        print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
        print(f"  Feature Importances:")
        for feat, imp in feat_imp_dict.items():
            print(f"    {feat:<10}: {imp:.4f}")

    # 5. Pick Best Model
    best_key = max(metrics_results.keys(), key=lambda k: metrics_results[k]["f1_score"])
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {models[best_key][1]} (F1-score: {metrics_results[best_key]['f1_score']:.4f})")
    print("=" * 60)

    # 6. Save models and metrics
    dt_path = os.path.join(MODELS_DIR, "decision_tree.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    joblib.dump(best_dt, dt_path)
    joblib.dump(best_rf, rf_path)
    print(f"\n[OK] Saved models:\n  {dt_path}\n  {rf_path}")

    full_payload = {
        "best_model": best_key,
        "feature_names": FEATURE_NAMES,
        "wavelet_features": WAVELET_ONLY_FEATURES,
        "models": metrics_results,
        "dataset_summary": dataset_summary,
        "training_feature_ranges": feature_stats,
        "test_set_size": len(X_test),
        "train_set_size": len(X_train),
        "positive_class": "1 (Forged)",
        "negative_class": "0 (Genuine)"
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[OK] Saved metrics to {METRICS_JSON}")

    return full_payload


if __name__ == "__main__":
    train_and_evaluate()
