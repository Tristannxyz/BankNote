"""
build_authenticity_dataset.py — Builds a 50/50 balanced genuine vs. counterfeit dataset
directly from 'dataset/' (genuine Philippine banknotes) and 'fake_dataset/' (counterfeit banknotes).

Class 0 (Genuine):
  - 217 banknote crops from 'crops/' (stratified across 20, 50, 100, 200, 500, 1000 Pesos from 'dataset/')
  - 217 uncropped banknote photos from 'dataset/train/images/'
  Total Genuine = 434 samples

Class 1 (Forged / Counterfeit):
  - 217 banknote crops from 'fake_dataset/train/' (via YOLO bounding-box annotations)
  - 217 uncropped banknote photos from 'fake_dataset/train/images/'
  Total Forged = 434 samples

Includes 'source_image_id' to prevent train/test leakage when crops and full photos are split.

Output: data/banknote_authenticity.csv (868 records, exactly 50% Genuine / 50% Forged)
"""

import os
import csv
import random
import numpy as np
import scipy.stats as stats
import pywt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "crops")
DATASET_IMG_DIR = os.path.join(BASE_DIR, "dataset", "train", "images")
FAKE_DIR = os.path.join(BASE_DIR, "fake_dataset", "train")
OUT_CSV = os.path.join(BASE_DIR, "data", "banknote_authenticity.csv")

DENOMINATIONS = [
    "20_Pesos", "50_Pesos", "100_Pesos", "200_Pesos",
    "500_Pesos", "1000_Pesos"
]

FEATURE_COLS = ["variance", "skewness", "curtosis", "entropy"]
COLOR_COLS = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
ALL_COLS = FEATURE_COLS + COLOR_COLS + ["class", "source_type", "source_image_id"]

WAVELET = "db1"


def compute_features_from_pil(img_rgb: Image.Image) -> dict:
    """Extract wavelet and color features from a PIL RGB Image."""
    # 1. Color features (RGB downscaled to 128×128)
    small = img_rgb.resize((128, 128), Image.Resampling.BILINEAR)
    c_arr = np.asarray(small, dtype=np.float64) / 255.0
    c_means = c_arr.mean(axis=(0, 1))
    c_stds = c_arr.std(axis=(0, 1))

    # 2. Wavelet features (Grayscale 400×400, Haar DWT cA)
    gray = img_rgb.convert("L").resize((400, 400), Image.Resampling.LANCZOS)
    w_arr = np.asarray(gray, dtype=np.float64) / 255.0
    cA, _ = pywt.dwt2(w_arr, WAVELET)

    v = float(np.var(cA))
    if v < 1e-12:
        s, k, ent = 0.0, 0.0, 0.0
    else:
        raw_s = float(stats.skew(cA.ravel()))
        s = 0.0 if not np.isfinite(raw_s) else raw_s

        raw_k = float(stats.kurtosis(cA.ravel(), fisher=True))
        k = 0.0 if not np.isfinite(raw_k) else raw_k

        counts, _ = np.histogram(cA, bins=256)
        total = np.sum(counts)
        if total > 0:
            p = counts / total
            nonzero = p[p > 0]
            h_disc = -np.sum(nonzero * np.log2(nonzero))
            span = float(cA.max() - cA.min())
            bin_w = span / 256.0 if span > 1e-12 else (1.0 / 256.0)
            ent = float(h_disc + np.log2(bin_w))
        else:
            ent = 0.0

    return {
        "variance": round(v, 4),
        "skewness": round(s, 4),
        "curtosis": round(k, 4),
        "entropy": round(ent, 4),
        "r_mean": round(float(c_means[0]), 4),
        "g_mean": round(float(c_means[1]), 4),
        "b_mean": round(float(c_means[2]), 4),
        "r_std": round(float(c_stds[0]), 4),
        "g_std": round(float(c_stds[1]), 4),
        "b_std": round(float(c_stds[2]), 4),
    }


def yolo_to_pixel_bbox(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def process_fake_item(args):
    img_path, lbl_path, source_id = args
    results = {}
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
        img_w, img_h = img_rgb.size

        # 1. Full photo
        feats_full = compute_features_from_pil(img_rgb)
        feats_full["class"] = 1
        feats_full["source_type"] = "counterfeit_full_photo"
        feats_full["source_image_id"] = source_id
        results["full"] = feats_full

        # 2. Bounding box crop
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1, y1, x2, y2 = yolo_to_pixel_bbox(cx, cy, bw, bh, img_w, img_h)
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                        crop = img_rgb.crop((x1, y1, x2, y2))
                        feats_crop = compute_features_from_pil(crop)
                        feats_crop["class"] = 1
                        feats_crop["source_type"] = "counterfeit_crop"
                        feats_crop["source_image_id"] = source_id
                        results["crop"] = feats_crop
                    break
    except Exception:
        pass
    return results


def process_genuine_item(args):
    path, source_type, source_id = args
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
        feats = compute_features_from_pil(img_rgb)
        feats["class"] = 0
        feats["source_type"] = source_type
        feats["source_image_id"] = source_id
        return feats
    except Exception:
        return None


def main():
    print("=" * 65)
    print("BUILDING 1:1 EQUAL DATASET FROM dataset/ AND fake_dataset/")
    print("=" * 65)

    random.seed(42)
    np.random.seed(42)

    # 1. Collect all real fake images from fake_dataset/
    img_dir = os.path.join(FAKE_DIR, "images")
    lbl_dir = os.path.join(FAKE_DIR, "labels")
    fake_items = []
    if os.path.isdir(img_dir):
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(fname)[0]
                source_id = f"fake_{stem}"
                fake_items.append((
                    os.path.join(img_dir, fname),
                    os.path.join(lbl_dir, stem + ".txt"),
                    source_id
                ))

    print(f"Reading {len(fake_items)} counterfeit images from fake_dataset/train/...")
    fake_crops = []
    fake_fulls = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = ex.map(process_fake_item, fake_items)
        for res in results:
            if "crop" in res:
                fake_crops.append(res["crop"])
            if "full" in res:
                fake_fulls.append(res["full"])

    target_fake_count = len(fake_crops)  # 217
    print(f"Extracted {len(fake_crops)} counterfeit crops and {len(fake_fulls)} counterfeit full photos.")

    # 2. Collect genuine crops from crops/ (derived from dataset/)
    crops_by_denom = {den: [] for den in DENOMINATIONS}
    for root, _, files in os.walk(CROPS_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                for den in DENOMINATIONS:
                    if den in root:
                        crops_by_denom[den].append(os.path.join(root, f))
                        break

    genuine_crop_items = []
    per_denom = target_fake_count // len(DENOMINATIONS)
    remainder = target_fake_count % len(DENOMINATIONS)
    for i, (den, p_list) in enumerate(crops_by_denom.items()):
        count = per_denom + (1 if i < remainder else 0)
        sampled = random.sample(p_list, count)
        for path in sampled:
            stem = os.path.splitext(os.path.basename(path))[0]
            genuine_crop_items.append((path, "genuine_crop", f"gen_crop_{stem}"))
        print(f"  Denomination {den:<12}: {len(sampled)} genuine crops")

    # 3. Collect genuine uncropped photos from dataset/train/images (exactly 217 photos)
    dataset_img_paths = []
    if os.path.isdir(DATASET_IMG_DIR):
        for f in os.listdir(DATASET_IMG_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                dataset_img_paths.append(os.path.join(DATASET_IMG_DIR, f))
    sampled_dataset_photos = random.sample(dataset_img_paths, target_fake_count)
    genuine_photo_items = [
        (p, "genuine_full_photo", f"gen_photo_{os.path.splitext(os.path.basename(p))[0]}")
        for p in sampled_dataset_photos
    ]
    print(f"Sampled {len(sampled_dataset_photos)} genuine uncropped photos from dataset/train/images.")

    # Extract features for genuine samples
    genuine_crops = []
    genuine_fulls = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(process_genuine_item, genuine_crop_items):
            if r:
                genuine_crops.append(r)
        for r in ex.map(process_genuine_item, genuine_photo_items):
            if r:
                genuine_fulls.append(r)

    print(f"Extracted {len(genuine_crops)} genuine crops and {len(genuine_fulls)} genuine full photos.")

    # 4. Assemble the equal dataset:
    all_genuine = genuine_crops + genuine_fulls
    all_forged = fake_crops + fake_fulls

    # Ensure exact 1:1 count parity
    final_count = min(len(all_genuine), len(all_forged))
    all_genuine = all_genuine[:final_count]
    all_forged = all_forged[:final_count]

    all_rows = all_genuine + all_forged
    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Equal dataset saved -> {OUT_CSV}")
    print(f"     Total records : {len(all_rows)}")
    print(f"     Genuine (0)   : {len(all_genuine)} ({len(all_genuine)/len(all_rows)*100:.1f}%) from dataset/")
    print(f"     Forged (1)    : {len(all_forged)} ({len(all_forged)/len(all_rows)*100:.1f}%) from fake_dataset/")


if __name__ == "__main__":
    main()
