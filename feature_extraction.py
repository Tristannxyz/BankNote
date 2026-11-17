"""
feature_extraction.py — Wavelet-based feature extractor for banknote images.
Replicates and extends the feature extraction methodology of banknote authentication:
1. Load image in-memory, validate integrity.
2. Convert to grayscale and resize to 400x400 (LANCZOS).
3. 2D Discrete Wavelet Transform (DWT) with Haar wavelet ('db1').
4. Compute variance, skewness, Fisher curtosis, and continuous Shannon entropy.
5. Compute high-frequency wavelet detail energy (microprint / intaglio sharpness).
6. Extract RGB color distribution (mean and standard deviation).
7. Perform out-of-distribution sanity check against training ranges from metrics.json.
"""

import io
import os
import json
import colorsys
import numpy as np
import scipy.stats as stats
from PIL import Image

try:
    import pywt
except (ImportError, OSError):
    pywt = None

DEFAULT_WAVELET = "db1"  # Haar wavelet


def extract_banknote_foreground_color(img_rgb: Image.Image) -> dict:
    """
    Extracts color features specifically from the banknote foreground,
    rejecting dark desks, black mousepads, white margins, and neutral backgrounds.
    """
    resized = img_rgb.resize((160, 160), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float64) / 255.0
    h, w, _ = arr.shape

    # Sample perimeter/border pixels (outer 6% on each side) to model background
    bx = max(2, int(w * 0.06))
    by = max(2, int(h * 0.06))
    border_pixels = np.concatenate([
        arr[:by, :, :].reshape(-1, 3),
        arr[-by:, :, :].reshape(-1, 3),
        arr[:, :bx, :].reshape(-1, 3),
        arr[:, -bx:, :].reshape(-1, 3),
    ])
    border_mean = border_pixels.mean(axis=0)
    border_lum = 0.299 * border_mean[0] + 0.587 * border_mean[1] + 0.114 * border_mean[2]

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat_map = np.zeros_like(max_c)
    nonzero = max_c > 1e-4
    sat_map[nonzero] = (max_c[nonzero] - min_c[nonzero]) / max_c[nonzero]

    color_dist = np.sqrt(np.sum((arr - border_mean) ** 2, axis=-1))

    is_border_dark = border_lum < 0.35
    is_border_light = border_lum > 0.85

    fg_mask = np.ones((h, w), dtype=bool)
    if is_border_dark:
        fg_mask &= (lum > max(0.20, border_lum + 0.08))
    elif is_border_light:
        fg_mask &= (lum < min(0.92, border_lum - 0.05))

    border_sat = (border_mean.max() - border_mean.min()) / max(border_mean.max(), 1e-4)
    if border_sat < 0.15:
        fg_mask &= ((sat_map > 0.07) | (color_dist > 0.15))
    else:
        fg_mask &= (color_dist > 0.10)

    cy_min, cy_max = int(h * 0.15), int(h * 0.85)
    cx_min, cx_max = int(w * 0.15), int(w * 0.85)
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[cy_min:cy_max, cx_min:cx_max] = True

    valid_pixels = arr[fg_mask]
    total_pixels = h * w
    if len(valid_pixels) < (total_pixels * 0.10):
        valid_pixels = arr[center_mask]
    if len(valid_pixels) < (total_pixels * 0.05):
        valid_pixels = arr.reshape(-1, 3)

    fg_mean = valid_pixels.mean(axis=0)
    fg_std = valid_pixels.std(axis=0)

    r_val = float(fg_mean[0])
    g_val = float(fg_mean[1])
    b_val = float(fg_mean[2])

    hsv = colorsys.rgb_to_hsv(r_val, g_val, b_val)
    hue_deg = float(hsv[0] * 360.0)
    sat_val = float(hsv[1])
    val_val = float(hsv[2])

    tot = r_val + g_val + b_val + 1e-6
    coverage_pct = round((len(valid_pixels) / float(total_pixels)) * 100.0, 1)

    return {
        "r_mean": round(r_val, 4),
        "g_mean": round(g_val, 4),
        "b_mean": round(b_val, 4),
        "r_std": round(float(fg_std[0]), 4),
        "g_std": round(float(fg_std[1]), 4),
        "b_std": round(float(fg_std[2]), 4),
        "hue": round(hue_deg, 2),
        "sat": round(sat_val, 4),
        "val": round(val_val, 4),
        "r_minus_g": round(r_val - g_val, 4),
        "r_minus_b": round(r_val - b_val, 4),
        "g_minus_b": round(g_val - b_val, 4),
        "r_ratio": round(r_val / tot, 4),
        "g_ratio": round(g_val / tot, 4),
        "b_ratio": round(b_val / tot, 4),
        "foreground_coverage_pct": coverage_pct,
    }


def _haar_dwt2(arr: np.ndarray):
    """
    Compute single-level 2D Haar ('db1') DWT.
    Uses PyWavelets when available, or an exact pure-NumPy fallback when pywt
    cannot be loaded (e.g., if its native extension is blocked by Application Control).
    """
    if pywt is not None:
        return pywt.dwt2(arr, DEFAULT_WAVELET)

    data = np.asarray(arr, dtype=np.float64)
    if data.shape[0] % 2 != 0:
        data = np.vstack([data, data[-1:, :]])
    if data.shape[1] % 2 != 0:
        data = np.hstack([data, data[:, -1:]])

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    even_c = data[:, 0::2]
    odd_c = data[:, 1::2]
    lo_c = (even_c + odd_c) * inv_sqrt2
    hi_c = (odd_c - even_c) * inv_sqrt2

    cA = (lo_c[0::2, :] + lo_c[1::2, :]) * inv_sqrt2
    cH = (lo_c[1::2, :] - lo_c[0::2, :]) * inv_sqrt2
    cV = (hi_c[0::2, :] + hi_c[1::2, :]) * inv_sqrt2
    cD = (hi_c[1::2, :] - hi_c[0::2, :]) * inv_sqrt2
    return cA, (cH, cV, cD)
TARGET_IMAGE_SIZE = (400, 400)
ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "GIF", "BMP", "WEBP"}

METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "metrics.json"
)


def load_training_ranges(metrics_path=None):
    """Load min/max training feature ranges from metrics.json if available."""
    path = metrics_path or METRICS_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("training_feature_ranges", {})
        except Exception:
            return {}
    return {}


def extract_features(
    image_bytes: bytes,
    metrics_path: str = None,
    tolerance_pct: float = 0.05
) -> dict:
    """
    Extracts numerical features from raw image bytes.

    Args:
        image_bytes: Raw bytes of the image file.
        metrics_path: Optional custom path to metrics.json for range verification.
        tolerance_pct: Percentage of feature range span to allow beyond min/max.

    Returns:
        dict with extracted features, color statistics, metadata, and OOD flags.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image data is empty. Please upload a valid image.")

    # 1. Open with Pillow and verify integrity
    try:
        verify_img = Image.open(io.BytesIO(image_bytes))
        orig_format = (verify_img.format or "").upper()
        orig_size = verify_img.size
        orig_mode = verify_img.mode
        verify_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file: {str(e)}")

    # Re-open stream for decoding
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if hasattr(img, "is_animated") and img.is_animated:
            img.seek(0)
    except Exception as e:
        raise ValueError(f"Unable to decode image frames: {str(e)}")

    # Convert to RGB for color features
    img_rgb = img.convert("RGB")

    # Foreground banknote color features (background-rejecting)
    color_feats = extract_banknote_foreground_color(img_rgb)

    # Global unsegmented color stats (for reference / backwards compatibility)
    small_rgb = img_rgb.resize((128, 128), Image.Resampling.BILINEAR)
    c_arr = np.asarray(small_rgb, dtype=np.float64) / 255.0
    c_means = c_arr.mean(axis=(0, 1))
    c_stds = c_arr.std(axis=(0, 1))
    raw_color_feats = {
        "raw_r_mean": round(float(c_means[0]), 4),
        "raw_g_mean": round(float(c_means[1]), 4),
        "raw_b_mean": round(float(c_means[2]), 4),
        "raw_r_std": round(float(c_stds[0]), 4),
        "raw_g_std": round(float(c_stds[1]), 4),
        "raw_b_std": round(float(c_stds[2]), 4),
    }

    # 2. Convert to grayscale ("L") & resize to 400x400
    img_gray = img_rgb.convert("L")
    img_resized = img_gray.resize(TARGET_IMAGE_SIZE, resample=Image.Resampling.LANCZOS)
    norm_arr = np.asarray(img_resized, dtype=np.float64) / 255.0

    # 3. Apply 2D Discrete Wavelet Transform
    cA, (cH, cV, cD) = _haar_dwt2(norm_arr)

    # High-frequency detail energy (sharpness of intaglio printing / microprint)
    hf_energy = float(np.mean(cH**2 + cV**2 + cD**2))

    # Variance of cA
    var_val = float(np.var(cA))
    if not np.isfinite(var_val) or var_val < 1e-12:
        var_val = 0.0
        skew_val = 0.0
        curt_val = 0.0
        entropy_val = 0.0
    else:
        # Skewness
        raw_skew = float(stats.skew(cA.ravel()))
        skew_val = 0.0 if not np.isfinite(raw_skew) else raw_skew

        # Curtosis: Fisher excess kurtosis
        raw_curt = float(stats.kurtosis(cA.ravel(), fisher=True))
        curt_val = 0.0 if not np.isfinite(raw_curt) else raw_curt

        # Shannon differential entropy
        counts, _ = np.histogram(cA, bins=256)
        total_counts = np.sum(counts)
        if total_counts > 0:
            probs = counts / total_counts
            nonzero_probs = probs[probs > 0]
            h_discrete = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))
            val_range = float(cA.max() - cA.min())
            bin_width = max(val_range / 256.0, 1e-6)
            raw_entropy = float(h_discrete + np.log2(bin_width))
            entropy_val = 0.0 if not np.isfinite(raw_entropy) else raw_entropy
        else:
            entropy_val = 0.0

    extracted_features = {
        "variance": round(var_val, 4),
        "skewness": round(skew_val, 4),
        "curtosis": round(curt_val, 4),
        "entropy": round(entropy_val, 4),
    }

    metadata = {
        "original_size": list(orig_size),
        "original_mode": orig_mode,
        "original_format": orig_format,
        "processed_size": list(TARGET_IMAGE_SIZE),
        "wavelet_used": DEFAULT_WAVELET,
        "coefficient_type": "cA (Approximation)",
        "kurtosis_type": "Fisher (Excess)",
        "hf_sharpness_energy": round(hf_energy, 6),
    }

    # Sanity check against training distribution
    training_ranges = load_training_ranges(metrics_path)
    out_of_range_features = []
    out_of_distribution = False
    warning_msg = None

    all_features_dict = {**extracted_features, **color_feats, **raw_color_feats}
    if training_ranges:
        for feat, val in all_features_dict.items():
            if feat in training_ranges:
                f_min = training_ranges[feat]["min"]
                f_max = training_ranges[feat]["max"]
                span = max(f_max - f_min, 1e-4)
                allowed_min = f_min - (tolerance_pct * span)
                allowed_max = f_max + (tolerance_pct * span)
                if val < allowed_min or val > allowed_max:
                    out_of_range_features.append(feat)

        if len(out_of_range_features) > 0:
            out_of_distribution = True
            feat_list_str = ", ".join(out_of_range_features[:4])
            warning_msg = (
                f"Features [{feat_list_str}] fall outside the genuine training range. "
                "This indicates potential forgery, scan artifact, or non-currency image."
            )

    return {
        "features": extracted_features,
        "all_features": all_features_dict,
        "variance": extracted_features["variance"],
        "skewness": extracted_features["skewness"],
        "curtosis": extracted_features["curtosis"],
        "entropy": extracted_features["entropy"],
        "color_features": color_feats,
        "hf_energy": hf_energy,
        "metadata": metadata,
        "out_of_distribution": out_of_distribution,
        "out_of_range_features": out_of_range_features,
        "warning": warning_msg,
    }
