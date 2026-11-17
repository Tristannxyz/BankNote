"""
train_denomination_model.py — Multi-class denomination classifier for Philippine Peso banknotes.

Trains two models on the YOLO-derived peso_notes.csv:
  - RandomForestClassifier  (saved as denomination_rf.joblib)
  - DecisionTreeClassifier  (saved as denomination_dt.joblib)

Features used (10 total):
  variance, skewness, curtosis, entropy   — from Haar DWT (same as UCI pipeline)
  r_mean, g_mean, b_mean, r_std, g_std, b_std — from RGB pixel statistics

Data splits are pre-labelled in the CSV ('split' column: train/valid/test).
"""

import os
import json
import colorsys
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(BASE_DIR, "data", "peso_notes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_JSON = os.path.join(MODELS_DIR, "denomination_metrics.json")

FEATURE_COLS = [
    "variance", "skewness", "curtosis", "entropy",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "r_minus_g", "r_minus_b", "g_minus_b",
    "r_ratio", "g_ratio", "b_ratio",
    "hue", "sat"
]

# Canonical ordered class list
CLASS_ORDER = [
    "20_Pesos", "50_Pesos", "100_Pesos", "200_Pesos",
    "500_Pesos", "1000_Pesos", "Polymer_1000_Pesos",
]

# Display labels for the UI
CLASS_DISPLAY = {
    "20_Pesos":           "₱20 Pesos (Orange/Brown)",
    "50_Pesos":           "₱50 Pesos (Red)",
    "100_Pesos":          "₱100 Pesos (Violet/Purple)",
    "200_Pesos":          "₱200 Pesos (Green)",
    "500_Pesos":          "₱500 Pesos (Yellow/Gold)",
    "1000_Pesos":         "₱1,000 Pesos (Blue)",
    "Polymer_1000_Pesos": "₱1,000 Polymer (2022 Philippine Eagle)",
}


def add_chromatic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute chromatic contrast and HSV features from RGB channels."""
    r = np.clip(df["r_mean"].values, 0.0, 1.0)
    g = np.clip(df["g_mean"].values, 0.0, 1.0)
    b = np.clip(df["b_mean"].values, 0.0, 1.0)
    tot = r + g + b + 1e-6
    df["r_ratio"] = r / tot
    df["g_ratio"] = g / tot
    df["b_ratio"] = b / tot
    df["r_minus_g"] = r - g
    df["r_minus_b"] = r - b
    df["g_minus_b"] = g - b

    hues, sats, vals = [], [], []
    for ri, gi, bi in zip(r, g, b):
        h, s, v = colorsys.rgb_to_hsv(ri, gi, bi)
        hues.append(h * 360.0)
        sats.append(s)
        vals.append(v)
    df["hue"] = hues
    df["sat"] = sats
    df["val"] = vals
    return df


def load_splits(csv_path: str):
    """Load CSV and split into train/valid/test DataFrames with chromatic features."""
    df = pd.read_csv(csv_path)
    # Keep only the 7 bill denominations
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    df = add_chromatic_features(df)

    missing = df[FEATURE_COLS].isnull().sum().sum()
    if missing > 0:
        print(f"[WARN] Dropping {missing} rows with missing feature values.")
        df = df.dropna(subset=FEATURE_COLS)

    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df  = df[df["split"] == "test"]

    return train_df, valid_df, test_df


def compute_metrics(model, X, y, class_names):
    """Compute per-class and macro metrics; return dict."""
    y_pred = model.predict(X)
    acc    = float(accuracy_score(y, y_pred))
    prec   = float(precision_score(y, y_pred, average="macro", zero_division=0))
    rec    = float(recall_score(y, y_pred, average="macro", zero_division=0))
    f1     = float(f1_score(y, y_pred, average="macro", zero_division=0))
    cm     = confusion_matrix(y, y_pred, labels=class_names).tolist()

    report = classification_report(
        y, y_pred, labels=class_names, output_dict=True, zero_division=0
    )
    per_class = {}
    for cls in class_names:
        if cls in report:
            per_class[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1_score":  round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }

    return {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class,
    }


def train_model_grid(EstimatorClass, param_grid, X_train, y_train, label):
    """Run GridSearchCV and return the best estimator."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        EstimatorClass(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    print(f"\n[INFO] Grid-searching {label}...")
    grid.fit(X_train, y_train)
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_CSV):
        raise SystemExit(
            f"[ERROR] Dataset not found at {DATA_CSV}.\n"
            "Run build_dataset.py first to generate it from the dataset/ folder."
        )

    print("=" * 60)
    print("DENOMINATION CLASSIFIER — Training Pipeline")
    print("=" * 60)
    print(f"Loading: {DATA_CSV}")

    train_df, valid_df, test_df = load_splits(DATA_CSV)

    # Combine train + valid for final training (test stays held-out)
    train_all_df = pd.concat([train_df, valid_df], ignore_index=True)

    print(f"\nTrain+Valid rows : {len(train_all_df)}")
    print(f"Test rows        : {len(test_df)}")
    print(f"Classes          : {CLASS_ORDER}")

    class_counts = train_all_df["class"].value_counts().to_dict()
    print("\nClass distribution (Train+Valid):")
    for cls in CLASS_ORDER:
        print(f"  {cls:<22}: {class_counts.get(cls, 0)}")

    X_train = train_all_df[FEATURE_COLS].values
    y_train = train_all_df["class"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["class"].values

    # Feature statistics for OOD checks
    feature_stats = {}
    for i, col in enumerate(FEATURE_COLS):
        col_vals = train_all_df[col].values
        feature_stats[col] = {
            "min":  float(col_vals.min()),
            "max":  float(col_vals.max()),
            "mean": float(col_vals.mean()),
            "std":  float(col_vals.std()),
        }

    # -------------------------------------------------------------------------
    # 1. Random Forest
    # -------------------------------------------------------------------------
    print("\n[INFO] Training Random Forest...")
    rf_best_params = {"n_estimators": 150, "max_depth": 14, "max_features": "sqrt"}
    best_rf = RandomForestClassifier(
        n_estimators=150, max_depth=14, max_features="sqrt", random_state=42
    )
    best_rf.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 2. Decision Tree
    # -------------------------------------------------------------------------
    print("[INFO] Training Decision Tree...")
    dt_best_params = {"max_depth": 10, "min_samples_split": 5, "criterion": "gini"}
    best_dt = DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, criterion="gini", random_state=42
    )
    best_dt.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Evaluation on held-out test set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    models_map = {
        "denomination_rf": (best_rf, "Random Forest",  rf_best_params),
        "denomination_dt": (best_dt, "Decision Tree",  dt_best_params),
    }

    model_metrics = {}
    for key, (model, display_name, best_params) in models_map.items():
        m = compute_metrics(model, X_test, y_test, CLASS_ORDER)
        m["model_name"]  = display_name
        m["best_params"] = best_params

        feat_imp = dict(zip(FEATURE_COLS, [float(v) for v in model.feature_importances_]))
        m["feature_importances"] = feat_imp

        model_metrics[key] = m

        print(f"\n{display_name}:")
        print(f"  Accuracy  : {m['accuracy']*100:.2f}%")
        print(f"  Precision : {m['precision']*100:.2f}%")
        print(f"  Recall    : {m['recall']*100:.2f}%")
        print(f"  F1 (macro): {m['f1_score']*100:.2f}%")
        print("  Per-class F1:")
        for cls, cm_stats in m["per_class_metrics"].items():
            print(f"    {cls:<22}: F1={cm_stats['f1_score']:.4f}")

    # Pick best by macro F1
    best_key = max(model_metrics, key=lambda k: model_metrics[k]["f1_score"])
    print(f"\n[BEST] {model_metrics[best_key]['model_name']} "
          f"(F1={model_metrics[best_key]['f1_score']:.4f})")

    # -------------------------------------------------------------------------
    # Save models
    # -------------------------------------------------------------------------
    rf_path = os.path.join(MODELS_DIR, "denomination_rf.joblib")
    dt_path = os.path.join(MODELS_DIR, "denomination_dt.joblib")
    joblib.dump(best_rf, rf_path)
    joblib.dump(best_dt, dt_path)
    print(f"\n[OK] Saved {rf_path}")
    print(f"[OK] Saved {dt_path}")

    # -------------------------------------------------------------------------
    # Save metrics JSON
    # -------------------------------------------------------------------------
    payload = {
        "best_model":           best_key,
        "feature_names":        FEATURE_COLS,
        "class_names":          CLASS_ORDER,
        "class_display_names":  CLASS_DISPLAY,
        "models":               model_metrics,
        "training_feature_ranges": feature_stats,
        "train_rows":           int(len(train_all_df)),
        "test_rows":            int(len(test_df)),
        "dataset_source":       "dataset/ (Philippine Banknotes Roboflow YOLO v8)",
        "dataset_summary": {
            "total_records":    int(len(train_all_df) + len(test_df)),
            "class_distribution": {
                cls: int(train_all_df["class"].value_counts().get(cls, 0))
                for cls in CLASS_ORDER
            },
        },
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved {METRICS_JSON}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

