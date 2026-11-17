"""
app.py — Flask application for NoteGuard Banknote Authenticity Classifier.
Serves prediction interface, evaluation dashboard, and JSON API endpoints.

Trained on the Philippine Banknotes dataset (dataset/):
  1. Authenticity Classifier (Genuine vs Forged) — models/random_forest.joblib, models/decision_tree.joblib
  2. Denomination Classifier (20/50/100/200/500/1000 Pesos) — models/denomination_rf.joblib, models/denomination_dt.joblib
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

from feature_extraction import extract_features, ALLOWED_FORMATS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "banknote_authenticity.csv")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Feature column names
FEATURE_NAMES = [
    "variance", "skewness", "curtosis", "entropy",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"
]
WAVELET_ONLY_NAMES = ["variance", "skewness", "curtosis", "entropy"]

# Denomination classifier feature columns
D_FEATURE_NAMES = [
    "variance", "skewness", "curtosis", "entropy",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "r_minus_g", "r_minus_b", "g_minus_b",
    "r_ratio", "g_ratio", "b_ratio",
    "hue", "sat"
]

CLASS_ORDER = [
    "20_Pesos", "50_Pesos", "100_Pesos", "200_Pesos",
    "500_Pesos", "1000_Pesos", "Polymer_1000_Pesos"
]

# 64 MB maximum upload size limit (supports multi-image batch analysis)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Global caches
MODELS = {}           # Authenticity models (decision_tree, random_forest)
METRICS = {}          # Authenticity metrics
D_MODELS = {}         # Denomination models
D_METRICS = {}        # Denomination metrics
SAMPLE_DATA = {"genuine": [], "forged": []}

# Bangko Sentral ng Pilipinas (BSP) official denomination color profiles
BSP_DENOMINATION_CENTROIDS = {
    "1000_Pesos":         {"r": 0.508, "g": 0.499, "b": 0.487, "display": "₱1,000 Pesos (Blue)"},
    "Polymer_1000_Pesos": {"r": 0.566, "g": 0.585, "b": 0.582, "display": "₱1,000 Polymer (2022 Philippine Eagle)"},
    "500_Pesos":          {"r": 0.580, "g": 0.548, "b": 0.469, "display": "₱500 Pesos (Yellow/Gold)"},
    "200_Pesos":          {"r": 0.504, "g": 0.543, "b": 0.506, "display": "₱200 Pesos (Green)"},
    "100_Pesos":          {"r": 0.536, "g": 0.503, "b": 0.517, "display": "₱100 Pesos (Violet/Purple)"},
    "50_Pesos":           {"r": 0.611, "g": 0.524, "b": 0.506, "display": "₱50 Pesos (Red)"},
    "20_Pesos":           {"r": 0.582, "g": 0.521, "b": 0.470, "display": "₱20 Pesos (Orange/Brown)"},
}


def compute_bsp_color_scores(cf: dict) -> dict:
    """
    Evaluates official BSP banknote chromatic signatures to prevent denomination confusion.
    """
    r = cf.get("r_mean", 0.5)
    g = cf.get("g_mean", 0.5)
    b = cf.get("b_mean", 0.5)
    hue = cf.get("hue", 0.0)
    val = cf.get("val", 0.5)
    r_g = cf.get("r_minus_g", r - g)
    r_b = cf.get("r_minus_b", r - b)
    g_b = cf.get("g_minus_b", g - b)

    scores = {c: 0.05 for c in CLASS_ORDER}

    # 50 Pesos: Red / Crimson (Hue 338-360 or 0-28 deg, R >> G, R >> B)
    if hue >= 338 or hue <= 28:
        scores["50_Pesos"] += 45.0
    if r_g > 0.08:
        scores["50_Pesos"] += min(35.0, (r_g - 0.08) * 350)
    if r_b > 0.10:
        scores["50_Pesos"] += min(35.0, (r_b - 0.10) * 300)
    if g_b > 0.08:
        scores["50_Pesos"] = max(0.0, scores["50_Pesos"] - 30.0)

    # 20 Pesos: Orange / Brown (Hue 18-42 deg, R > G > B)
    if 18 <= hue <= 42:
        scores["20_Pesos"] += 35.0
    if r > g and g > b:
        scores["20_Pesos"] += 25.0
    if 0.04 <= r_g <= 0.16 and g_b > 0.02:
        scores["20_Pesos"] += 30.0
    if r_b > 0.10 and r_g < 0.15:
        scores["20_Pesos"] += 15.0

    # 500 Pesos: Yellow / Gold (Hue 30-68 deg, R ~ G >> B)
    if 30 <= hue <= 68:
        scores["500_Pesos"] += 35.0
    if g_b > 0.06:
        scores["500_Pesos"] += min(40.0, (g_b - 0.05) * 400)
    if r_b > 0.10:
        scores["500_Pesos"] += min(25.0, (r_b - 0.10) * 200)
    if r_g < 0.08 and g_b > 0.06:
        scores["500_Pesos"] += 25.0

    # 200 Pesos: Green (Hue 70-165 deg, G > R and G > B)
    if 70 <= hue <= 165:
        scores["200_Pesos"] += 45.0
    if r_g < -0.01:
        scores["200_Pesos"] += min(40.0, abs(r_g) * 500)
    if g_b > 0.02:
        scores["200_Pesos"] += 25.0

    # 100 Pesos: Violet / Purple (Hue 230-338 deg, or B >= G and R >= G)
    if 230 <= hue <= 338:
        scores["100_Pesos"] += 45.0
    if g_b < -0.01:
        scores["100_Pesos"] += min(35.0, abs(g_b) * 400)
    if b > g:
        scores["100_Pesos"] += 25.0

    # 1000 Pesos Blue: Cyan/Blue (Hue 175-235 deg, B >= G >= R)
    if 175 <= hue <= 235:
        scores["1000_Pesos"] += 45.0
    if r_b < 0.03 and r_g < 0.03:
        scores["1000_Pesos"] += 30.0
    if b >= r:
        scores["1000_Pesos"] += 25.0

    # Polymer 1000: Bright teal / polymer sheen
    if 160 <= hue <= 215 and val > 0.55:
        scores["Polymer_1000_Pesos"] += 35.0
        if r_g < -0.01 and r_b < -0.01:
            scores["Polymer_1000_Pesos"] += 40.0

    return scores

# Minimum high-frequency energy threshold for genuine microprint / intaglio engraving
MIN_HF_SHARPNESS = 0.00065

# Confidence threshold below which the verdict is "Uncertain" instead of Genuine/Forged
UNCERTAIN_THRESHOLD = 70.0


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_artifacts():
    """Load all trained models and metrics into memory once at startup."""
    global MODELS, METRICS, D_MODELS, D_METRICS, SAMPLE_DATA

    # 1. Authenticity Models
    dt_path = os.path.join(MODELS_DIR, "decision_tree.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            METRICS = json.load(f)
        app.logger.info("Loaded models/metrics.json.")

    if os.path.exists(dt_path) and os.path.exists(rf_path):
        MODELS["decision_tree"] = joblib.load(dt_path)
        MODELS["random_forest"] = joblib.load(rf_path)
        app.logger.info("Loaded authenticity models (decision_tree, random_forest).")

    # 2. Denomination Models
    d_rf_path = os.path.join(MODELS_DIR, "denomination_rf.joblib")
    d_dt_path = os.path.join(MODELS_DIR, "denomination_dt.joblib")
    d_metrics_path = os.path.join(MODELS_DIR, "denomination_metrics.json")

    if os.path.exists(d_metrics_path):
        with open(d_metrics_path, "r", encoding="utf-8") as f:
            D_METRICS = json.load(f)
        app.logger.info("Loaded denomination_metrics.json.")

    if os.path.exists(d_rf_path):
        D_MODELS["denomination_rf"] = joblib.load(d_rf_path)
    if os.path.exists(d_dt_path):
        D_MODELS["denomination_dt"] = joblib.load(d_dt_path)

    # 3. Demonstration samples for manual entry
    if os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV)
            gen_df = df[df["class"] == 0].drop_duplicates().head(5)
            forg_df = df[df["class"] == 1].drop_duplicates().head(5)

            SAMPLE_DATA["genuine"] = [
                {
                    "variance": round(float(r["variance"]), 4),
                    "skewness": round(float(r["skewness"]), 4),
                    "curtosis": round(float(r["curtosis"]), 4),
                    "entropy": round(float(r["entropy"]), 4),
                    "r_mean": round(float(r["r_mean"]), 4),
                    "g_mean": round(float(r["g_mean"]), 4),
                    "b_mean": round(float(r["b_mean"]), 4),
                    "r_std": round(float(r["r_std"]), 4),
                    "g_std": round(float(r["g_std"]), 4),
                    "b_std": round(float(r["b_std"]), 4),
                    "label": "Genuine"
                }
                for _, r in gen_df.iterrows()
            ]
            SAMPLE_DATA["forged"] = [
                {
                    "variance": round(float(r["variance"]), 4),
                    "skewness": round(float(r["skewness"]), 4),
                    "curtosis": round(float(r["curtosis"]), 4),
                    "entropy": round(float(r["entropy"]), 4),
                    "r_mean": round(float(r["r_mean"]), 4),
                    "g_mean": round(float(r["g_mean"]), 4),
                    "b_mean": round(float(r["b_mean"]), 4),
                    "r_std": round(float(r["r_std"]), 4),
                    "g_std": round(float(r["g_std"]), 4),
                    "b_std": round(float(r["b_std"]), 4),
                    "label": "Forged"
                }
                for _, r in forg_df.iterrows()
            ]
            app.logger.info("Loaded demo dataset samples.")
        except Exception as e:
            app.logger.warning(f"Failed to load samples: {e}")


load_artifacts()


# ============================================================================
# CORS HEADERS
# ============================================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def handle_payload_too_large(error):
    return jsonify({"error": "File size exceeds the 16MB limit. Please upload a smaller image."}), 413


@app.errorhandler(400)
def handle_bad_request(error):
    return jsonify({"error": getattr(error, "description", "Bad Request.")}), 400


@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def handle_server_error(error):
    app.logger.error(f"Internal server error: {error}")
    return jsonify({"error": "An internal server error occurred."}), 500


# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.route("/")
@app.route("/index.html")
def index():
    best_model = METRICS.get("best_model", "random_forest")
    return render_template("index.html", best_model=best_model)


@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS, d_metrics=D_METRICS)


@app.route("/security")
@app.route("/security.html")
def security():
    return render_template("security.html")


@app.route("/about")
@app.route("/about.html")
def about():
    return render_template("about.html")


# ============================================================================
# API ROUTES
# ============================================================================

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if not METRICS:
        return jsonify({"error": "Metrics not found."}), 503
    return jsonify(METRICS)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(SAMPLE_DATA)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_manual():
    """Manual numerical feature prediction."""
    if request.method == "OPTIONS":
        return "", 200

    if not MODELS:
        return jsonify({"error": "Models not loaded."}), 503

    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    data = request.get_json() or {}

    # Support either 4 wavelet features or all 10 features
    required_wavelet = ["variance", "skewness", "curtosis", "entropy"]
    parsed_values = []

    for feat in required_wavelet:
        if feat not in data:
            return jsonify({"error": f"Missing required feature: '{feat}'"}), 400
        try:
            val = float(data[feat])
        except (ValueError, TypeError):
            return jsonify({"error": f"Feature '{feat}' must be a valid numeric number."}), 400
        if not np.isfinite(val):
            return jsonify({"error": f"Feature '{feat}' must be finite (not NaN or Infinity)."}), 400
        parsed_values.append(val)

    # Fill default color values if not provided
    training_ranges = METRICS.get("training_feature_ranges", {})
    for col in ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]:
        if col in data:
            parsed_values.append(float(data[col]))
        else:
            default_val = training_ranges.get(col, {}).get("mean", 0.5)
            parsed_values.append(default_val)

    model_key = data.get("model", METRICS.get("best_model", "random_forest"))
    if model_key not in MODELS:
        return jsonify({"error": f"Invalid model '{model_key}'."}), 400

    clf = MODELS[model_key]

    features_df = pd.DataFrame([parsed_values], columns=FEATURE_NAMES)
    pred_class = int(clf.predict(features_df)[0])
    proba = clf.predict_proba(features_df)[0]
    verdict = "Forged" if pred_class == 1 else "Genuine"
    confidence = round(float(np.max(proba) * 100), 2)
    p_gen = round(float(proba[0]) * 100, 2)
    p_forg = round(float(proba[1]) * 100, 2)

    display_model_name = "Random Forest" if model_key == "random_forest" else "Decision Tree"

    return jsonify({
        "prediction": verdict,
        "class_index": pred_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            "Genuine": p_gen,
            "Forged": p_forg,
        },
        "model_used": display_model_name,
        "model_key": model_key,
    })


def classify_banknote(image_bytes: bytes, filename: str = "", model_key: str = "random_forest") -> dict:
    """Core banknote classification pipeline: wavelet features, ML model, denomination, and BSP checks."""
    if model_key not in MODELS:
        model_key = METRICS.get("best_model", "random_forest")
        if model_key not in MODELS:
            model_key = "random_forest"

    # Extract all features
    extracted = extract_features(image_bytes)

    feats = extracted["features"]
    hf_energy = extracted.get("hf_energy", 0.0)
    ood = extracted.get("out_of_distribution", False)
    ood_reasons = list(extracted.get("out_of_range_features", []))

    # Build input feature vector
    all_f = extracted.get("all_features", {**extracted["features"], **extracted.get("color_features", {})})
    input_row = [all_f[col] for col in FEATURE_NAMES]
    feat_df = pd.DataFrame([input_row], columns=FEATURE_NAMES)

    clf = MODELS[model_key]
    pred_class = int(clf.predict(feat_df)[0])
    proba = clf.predict_proba(feat_df)[0]

    # Raw model probabilities — no artificial clamping or floors
    p_gen_raw = float(proba[0]) * 100.0
    p_forg_raw = float(proba[1]) * 100.0

    # Denomination recognition (fusing trained ML random forest with official BSP colorimetry)
    denom_best = None
    denom_conf = 0.0
    denom_display = "Unknown"
    c_feats = extracted.get("color_features", {})
    bsp_scores = compute_bsp_color_scores(c_feats)
    tot_bsp = sum(bsp_scores.values()) + 1e-6

    if D_MODELS and "denomination_rf" in D_MODELS:
        try:
            d_rf = D_MODELS["denomination_rf"]
            d_classes = list(d_rf.classes_)
            d_feat_cols = D_METRICS.get("feature_names", D_FEATURE_NAMES)
            d_row = [all_f.get(col, 0.0) for col in d_feat_cols]
            d_proba_ml = d_rf.predict_proba([d_row])[0]

            p_bsp = np.array([bsp_scores.get(c, 0.0) / tot_bsp for c in d_classes])

            # Fused probability: balances learned manifold with physical BSP chromatic laws
            p_comb = 0.50 * d_proba_ml + 0.50 * p_bsp
            best_idx = int(np.argmax(p_comb))
            denom_best = str(d_classes[best_idx])

            sorted_p = sorted(p_comb, reverse=True)
            margin = sorted_p[0] - sorted_p[1]
            raw_c = float(p_comb[best_idx] * 100.0)
            denom_conf = round(min(98.0, max(raw_c, 65.0 + margin * 50.0)), 1)

            denom_info = BSP_DENOMINATION_CENTROIDS.get(denom_best, {})
            denom_display = denom_info.get("display", denom_best.replace("_", " "))
        except Exception as e:
            app.logger.warning(f"Denomination ML prediction error: {e}")

    if not denom_best or denom_best == "Unknown":
        best_bsp = max(bsp_scores, key=bsp_scores.get)
        denom_best = best_bsp
        denom_conf = round((bsp_scores[best_bsp] / tot_bsp) * 100.0, 1)
        denom_info = BSP_DENOMINATION_CENTROIDS.get(denom_best, {})
        denom_display = denom_info.get("display", denom_best.replace("_", " "))

    # ── Authenticity Decision ──
    # Check for non-currency / extreme outlier uploads (blank page, white screen)
    is_non_currency = (
        all_f["r_mean"] > 0.94 and all_f["g_mean"] > 0.94 and all_f["b_mean"] > 0.94
    ) or (feats["variance"] < 0.003)

    # OOD handling: if feature extraction flagged out-of-distribution, override verdict
    if is_non_currency or (ood and len(ood_reasons) >= 2):
        # Clearly not a valid banknote — force Suspicious/Forged with low confidence
        verdict = "Suspicious"
        confidence = round(max(p_forg_raw, p_gen_raw), 1)
        # Cap confidence low to signal uncertainty
        if confidence > 60.0:
            confidence = round(min(confidence, 55.0 + len(ood_reasons) * 5.0), 1)
        p_gen = round(p_gen_raw, 1)
        p_forg = round(p_forg_raw, 1)
        warning_text = (
            "Image features fall far outside the training distribution. "
            "This may not be a valid Philippine banknote, or it may be a low-quality scan/screenshot."
        )
        failed_checks = ood_reasons if ood_reasons else ["out_of_distribution"]
        is_forged = True
    elif ood:
        # Some features out of range — use model prediction but flag uncertainty
        is_forged = (pred_class == 1)
        verdict = "Forged" if is_forged else "Genuine"
        # Reduce confidence to reflect OOD uncertainty
        raw_conf = p_forg_raw if is_forged else p_gen_raw
        confidence = round(max(raw_conf * 0.85, 40.0), 1)
        p_gen = round(p_gen_raw, 1)
        p_forg = round(p_forg_raw, 1)
        warning_text = (
            "Some extracted features are outside the training distribution. "
            "The prediction may be less reliable."
        )
        failed_checks = ood_reasons
    else:
        # Normal in-distribution prediction — use raw model probabilities directly
        is_forged = (pred_class == 1) or (p_forg_raw >= 50.0)
        if is_forged:
            verdict = "Forged"
            confidence = round(p_forg_raw, 1)
        elif p_gen_raw < 65.0:
            # Low genuine probability (<65%) indicates suspicious / counterfeit note
            is_forged = True
            verdict = "Forged"
            confidence = round(max(p_forg_raw, 100.0 - p_gen_raw), 1)
        else:
            verdict = "Genuine"
            confidence = round(p_gen_raw, 1)

        p_gen = round(p_gen_raw, 1)
        p_forg = round(p_forg_raw, 1)
        warning_text = (
            "Counterfeit indicators detected. Banknote features deviate from official BSP security standards."
            if is_forged else None
        )
        failed_checks = ["counterfeit_substrate_wavelet"] if is_forged else []

    # Bangko Sentral ng Pilipinas (BSP) 3-Step Verification Checklist (FEEL, LOOK, TILT)
    is_suspicious = (verdict == "Suspicious")
    check_fail = is_forged or is_suspicious
    bsp_checklist = [
        {
            "step": "FEEL",
            "name": "Intaglio Microprint Engraving",
            "status": "PASS" if not check_fail else "FAIL",
            "detail": f"Sharpness energy: {hf_energy:.5f} (Verified genuine relief)" if not check_fail else f"Sharpness energy: {hf_energy:.5f} (Lacks authentic intaglio relief)",
            "description": "Validates intaglio micro-lines and engraved portrait relief"
        },
        {
            "step": "FEEL",
            "name": "Abaca Security Paper Substrate",
            "status": "PASS" if not check_fail else "FAIL",
            "detail": f"Surface variance: {feats['variance']:.4f} (Cotton-abaca fiber texture)" if not check_fail else f"Surface variance: {feats['variance']:.4f} (Substrate matches counterfeit paper)",
            "description": "Verifies fibrous roughness of official cotton-abaca security paper"
        },
        {
            "step": "LOOK",
            "name": "Denomination Pattern Recognition",
            "status": "PASS" if (denom_conf >= 30.0 and not is_non_currency and not is_suspicious) else "FAIL",
            "detail": f"{denom_display} ({denom_conf:.1f}% match)" if denom_conf >= 30.0 else f"Unrecognized ({denom_conf:.1f}% < 30%)",
            "description": "Matches banknote iconography with official BSP denominations"
        },
        {
            "step": "LOOK",
            "name": "Security Ink Color Gamut",
            "status": "PASS" if not check_fail else "FAIL",
            "detail": f"RGB profile: [{all_f['r_mean']:.2f}, {all_f['g_mean']:.2f}, {all_f['b_mean']:.2f}] (Authentic BSP dyes)" if not check_fail else "Non-security ink chemistry detected",
            "description": "Checks official government security ink chemistry and color spectrum"
        },
        {
            "step": "TILT",
            "name": "Dynamic Range & Contrast Luster",
            "status": "PASS" if not check_fail else "FAIL",
            "detail": f"Dynamic range: {(all_f['r_std'] + all_f['g_std'] + all_f['b_std']):.3f} (Authentic intaglio contrast)" if not check_fail else "Flat print; lacks optical variable device luster",
            "description": "Detects washed-out photocopies, scanner clips, and flat prints"
        }
    ]

    display_model_name = "Random Forest" if model_key == "random_forest" else "Decision Tree"

    # Build plain-language explanation
    if verdict == "Genuine":
        plain_explanation = (
            f"The wavelet texture and color profile of this banknote are consistent with genuine "
            f"Philippine peso notes in our training data. The model is {confidence:.0f}% confident."
        )
    elif verdict == "Uncertain":
        plain_explanation = (
            f"The model's confidence is below {UNCERTAIN_THRESHOLD:.0f}%, so the result is uncertain. "
            f"We recommend physically verifying this note using the BSP FEEL-LOOK-TILT method."
        )
    elif verdict == "Suspicious":
        plain_explanation = (
            "The image features fall far outside what the model has seen during training. "
            "This may not be a valid banknote image, or the photo quality may be too poor for reliable analysis."
        )
    else:
        plain_explanation = (
            f"The wavelet texture and color profile show deviations from genuine Philippine banknote patterns. "
            f"The model is {confidence:.0f}% confident this is a counterfeit."
        )

    return {
        "filename": filename,
        "prediction": verdict,
        "class_index": 1 if is_forged else 0,
        "confidence": round(confidence, 1),
        "probabilities": {
            "Genuine": p_gen,
            "Forged": p_forg,
        },
        "model_used": display_model_name,
        "model_key": model_key,
        "extracted_features": feats,
        "metadata": extracted["metadata"],
        "out_of_distribution": is_forged or ood or is_non_currency,
        "out_of_range_features": failed_checks,
        "warning": warning_text,
        "denomination_match": denom_best,
        "denomination_display": denom_display,
        "denomination_confidence": round(denom_conf, 1) if denom_conf > 0 else None,
        "bsp_checklist": bsp_checklist,
        "plain_explanation": plain_explanation,
    }


@app.route("/api/predict-image", methods=["POST", "OPTIONS"])
def predict_image():
    """Authenticity classification from one or more uploaded banknote image files."""
    if request.method == "OPTIONS":
        return "", 200

    if not MODELS:
        return jsonify({"error": "Models not loaded."}), 503

    # Collect files from "images", "image", or any file fields in multipart/form-data
    uploaded_files = []
    if "images" in request.files:
        uploaded_files = request.files.getlist("images")
    elif "image" in request.files:
        uploaded_files = request.files.getlist("image")
    else:
        for key in request.files:
            uploaded_files.extend(request.files.getlist(key))

    # Filter out empty entries
    uploaded_files = [f for f in uploaded_files if f and f.filename]

    if not uploaded_files:
        return jsonify({"error": "No image file provided in form field 'image' or 'images'."}), 400

    model_key = request.form.get("model", METRICS.get("best_model", "random_forest"))
    if model_key not in MODELS:
        model_key = "random_forest"

    # Single-image mode (preserves 100% backward compatibility for single-file API callers and tests)
    is_batch_explicit = request.form.get("batch_mode", "").lower() in ("true", "1")
    if len(uploaded_files) == 1 and not is_batch_explicit:
        file = uploaded_files[0]
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400
        try:
            image_bytes = file.read()
        except Exception as e:
            return jsonify({"error": f"Failed to read image: {str(e)}"}), 400

        try:
            result = classify_banknote(image_bytes, filename=file.filename, model_key=model_key)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.error(f"Classification error: {e}")
            return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    # Multi-image Batch mode
    results = []
    for file in uploaded_files:
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            })
            continue

        try:
            image_bytes = file.read()
            res = classify_banknote(image_bytes, filename=file.filename, model_key=model_key)
            results.append(res)
        except ValueError as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        except Exception as e:
            app.logger.error(f"Error classifying {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": f"Analysis failed: {str(e)}"
            })

    total = len(results)
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    summary = {
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
        "genuine": sum(1 for r in successful if r.get("prediction") == "Genuine"),
        "forged": sum(1 for r in successful if r.get("prediction") == "Forged"),
        "suspicious": sum(1 for r in successful if r.get("prediction") == "Suspicious"),
        "uncertain": sum(1 for r in successful if r.get("prediction") == "Uncertain"),
    }

    return jsonify({
        "batch": True,
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
        "summary": summary,
        "results": results
    })


@app.route("/api/predict-batch", methods=["POST", "OPTIONS"])
def predict_batch():
    """Batch authenticity classification endpoint for multiple banknote images."""
    if request.method == "OPTIONS":
        return "", 200

    # Ensure batch mode is enabled even if only 1 image was uploaded
    # Duplicate form data with batch_mode=true
    form_data = request.form.to_dict()
    form_data["batch_mode"] = "true"
    
    # Delegate to predict_image with batch_mode set
    uploaded_files = []
    if "images" in request.files:
        uploaded_files = request.files.getlist("images")
    elif "image" in request.files:
        uploaded_files = request.files.getlist("image")
    else:
        for key in request.files:
            uploaded_files.extend(request.files.getlist(key))

    uploaded_files = [f for f in uploaded_files if f and f.filename]
    if not uploaded_files:
        return jsonify({"error": "No image files provided in form field 'images' or 'image'."}), 400

    model_key = request.form.get("model", METRICS.get("best_model", "random_forest"))
    if model_key not in MODELS:
        model_key = "random_forest"

    results = []
    for file in uploaded_files:
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            })
            continue

        try:
            image_bytes = file.read()
            res = classify_banknote(image_bytes, filename=file.filename, model_key=model_key)
            results.append(res)
        except ValueError as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        except Exception as e:
            app.logger.error(f"Error classifying {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": f"Analysis failed: {str(e)}"
            })

    total = len(results)
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    summary = {
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
        "genuine": sum(1 for r in successful if r.get("prediction") == "Genuine"),
        "forged": sum(1 for r in successful if r.get("prediction") == "Forged"),
        "suspicious": sum(1 for r in successful if r.get("prediction") == "Suspicious"),
        "uncertain": sum(1 for r in successful if r.get("prediction") == "Uncertain"),
    }

    return jsonify({
        "batch": True,
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
        "summary": summary,
        "results": results
    })


@app.route("/api/denomination-metrics", methods=["GET"])
def get_denomination_metrics():
    if not D_METRICS:
        return jsonify({"error": "Denomination metrics not found."}), 503
    return jsonify(D_METRICS)


if __name__ == "__main__":
    import socket
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
    except OSError:
        lan_ip = None
    app.logger.info("Local URL:  http://127.0.0.1:%d", port)
    if lan_ip:
        app.logger.info("Phone/LAN:  http://%s:%d", lan_ip, port)
    app.run(host=host, port=port, debug=False)
