"""
build_dataset.py — Turn the Roboflow Philippine Banknotes (YOLO format) dataset
into a feature CSV that train_model.py can use.

Features extracted for each banknote:
  - Wavelet features: variance, skewness, curtosis, entropy (via DWT Haar 'db1')
  - Color features: r_mean, g_mean, b_mean, r_std, g_std, b_std (normalized 0-1)
  - Target class: Bill denomination (20_Pesos, 50_Pesos, 100_Pesos, etc.)

Supports reading from existing crops in 'crops/' or generating them from 'dataset/'.
"""

import os
import io
import csv
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.stats as stats
import pywt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CROPS_DIR = os.path.join(BASE_DIR, "crops")
OUT_CSV = os.path.join(BASE_DIR, "data", "peso_notes.csv")

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "1000_Pesos", "100_Pesos", "10_Pesos_New", "10_Pesos_Old", "1_Coin_New",
    "1_Coin_Old", "200_Pesos", "20_Coin", "20_Pesos", "25_Centavo_New",
    "25_Centavo_Old", "500_Pesos", "50_Pesos", "5_Coin_New", "5_Coin_Old",
    "Polymer_1000_Pesos",
]

BILL_CLASSES = {
    "20_Pesos", "50_Pesos", "100_Pesos", "200_Pesos",
    "500_Pesos", "1000_Pesos", "Polymer_1000_Pesos",
}

MIN_SIDE = 80
PAD_FRACTION = 0.02

FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]
COLOR_COLUMNS = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
CSV_COLUMNS = FEATURE_COLUMNS + COLOR_COLUMNS + ["class", "split", "source_image"]


def extract_crop_features(img_path, cls_name, split, fname):
    """Extract wavelet and color features directly from a cropped bill image."""
    try:
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        # 1. Color features (RGB downscaled)
        small = img.resize((128, 128), Image.Resampling.BILINEAR)
        c_arr = np.asarray(small, dtype=np.float64) / 255.0
        c_means = c_arr.mean(axis=(0, 1))
        c_stds = c_arr.std(axis=(0, 1))

        # 2. Wavelet features (Grayscale 400x400, Haar DWT cA)
        gray = img.convert("L").resize((400, 400), Image.Resampling.LANCZOS)
        w_arr = np.asarray(gray, dtype=np.float64) / 255.0
        cA, _ = pywt.dwt2(w_arr, "db1")

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
            "class": cls_name,
            "split": split,
            "source_image": fname,
        }
    except Exception as e:
        return None


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        nums = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(nums) == 4:
        xc, yc, w, h = nums
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        xs, ys = nums[0::2], nums[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None
    return cls, xc, yc, w, h

def build_from_crops():
    """Extract features directly from already generated crops/ folder."""
    tasks = []
    for split in SPLITS:
        split_dir = os.path.join(CROPS_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for cls_name in sorted(os.listdir(split_dir)):
            if cls_name not in BILL_CLASSES:
                continue
            cls_dir = os.path.join(split_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                    fpath = os.path.join(cls_dir, fname)
                    tasks.append((fpath, cls_name, split, fname))

    if not tasks:
        return None

    print(f"[INFO] Found {len(tasks)} pre-existing crops in '{CROPS_DIR}'!")
    print(f"[INFO] Extracting wavelet + color features with multithreading...")

    rows = []
    counts = {s: Counter() for s in SPLITS}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_crop_features, *t): t for t in tasks}
        completed = 0
        total = len(tasks)
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                rows.append(res)
                counts[res["split"]][res["class"]] += 1
            completed += 1
            if completed % 1000 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Processed {completed}/{total} crops ({completed*100/total:.1f}%) — {rate:.1f} crops/sec")

    return rows, counts


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # 1. Try reading existing crops first (fastest)
    result = build_from_crops()
    if result is not None:
        rows, counts = result
    else:
        print(f"[INFO] No existing crops found. Processing YOLO dataset from '{DATASET_DIR}'...")
        if not os.path.isdir(DATASET_DIR):
            raise SystemExit(f"Folder not found: {DATASET_DIR}")
        rows = []
        counts = {split: Counter() for split in SPLITS}
        # Iterate over YOLO labels and crop
        for split in SPLITS:
            img_dir = os.path.join(DATASET_DIR, split, "images")
            lbl_dir = os.path.join(DATASET_DIR, split, "labels")
            if not os.path.isdir(img_dir):
                continue

            fnames = sorted(os.listdir(img_dir))
            print(f"[INFO] Processing {len(fnames)} images for split '{split}'...")
            for idx, fname in enumerate(fnames):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in IMAGE_EXTS:
                    continue
                lbl_path = os.path.join(lbl_dir, stem + ".txt")
                if not os.path.exists(lbl_path):
                    continue

                try:
                    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                except Exception:
                    continue
                W, H = img.size

                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for b_idx, line in enumerate(lines):
                    parsed = parse_label_line(line)
                    if parsed is None:
                        continue
                    cls_id, xc, yc, w, h = parsed
                    if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                        continue
                    cls_name = CLASS_NAMES[cls_id]
                    if cls_name not in BILL_CLASSES:
                        continue

                    bw, bh = w * W, h * H
                    x0 = int(max(0, (xc * W) - bw / 2 - PAD_FRACTION * W))
                    y0 = int(max(0, (yc * H) - bh / 2 - PAD_FRACTION * H))
                    x1 = int(min(W, (xc * W) + bw / 2 + PAD_FRACTION * W))
                    y1 = int(min(H, (yc * H) + bh / 2 + PAD_FRACTION * H))
                    if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                        continue

                    crop = img.crop((x0, y0, x1, y1))
                    crop_dir = os.path.join(CROPS_DIR, split, cls_name)
                    os.makedirs(crop_dir, exist_ok=True)
                    crop_name = f"{stem}_{b_idx}.png"
                    crop_path = os.path.join(crop_dir, crop_name)
                    crop.save(crop_path)

                    feat_row = extract_crop_features(crop_path, cls_name, split, fname)
                    if feat_row:
                        rows.append(feat_row)
                        counts[split][cls_name] += 1

                if (idx + 1) % 500 == 0:
                    print(f"  [{split}] {idx + 1}/{len(fnames)} images processed...")

    if not rows:
        raise SystemExit("No bill crops could be processed.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Wrote {len(rows)} bill records to:\n  {OUT_CSV}")
    print("=" * 60)
    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"\n{split.upper()} SPLIT ({total} crops):")
        for name in sorted(BILL_CLASSES):
            print(f"  {name:<22} : {counts[split][name]}")


if __name__ == "__main__":
    main()
