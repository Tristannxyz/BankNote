"""
test_feature_extraction.py — Unit and self-tests for feature extraction.
Verifies:
1. Valid synthetic images extract 4 finite numerical features.
2. Formats: PNG, JPEG, GIF.
3. Rejecting corrupt / non-image bytes (e.g. .txt renamed to .png).
4. Out-of-distribution detection flags.
"""

import io
import math
import unittest
import numpy as np
from PIL import Image, ImageDraw

from feature_extraction import extract_features, DEFAULT_WAVELET


def create_synthetic_image_bytes(format="PNG", mode="L", size=(400, 400)):
    """Generate in-memory synthetic image bytes."""
    im = Image.new("RGB" if format == "JPEG" else "L", size, color=200)
    draw = ImageDraw.Draw(im)
    # Draw geometric patterns resembling banknote intaglio printing
    for y in range(20, size[1] - 20, 10):
        draw.line([(20, y), (size[0] - 20, y)], fill=50, width=1)
    for x in range(30, size[0] - 30, 25):
        draw.rectangle([x, 50, x + 15, size[1] - 50], fill=120)
    draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], fill=80)

    buf = io.BytesIO()
    if format == "GIF":
        im_converted = im.convert("P")
        im_converted.save(buf, format="GIF")
    else:
        im.save(buf, format=format)
    return buf.getvalue()


def test_synthetic_png_features():
    """Verify PNG extraction outputs 4 finite numbers with proper metadata."""
    img_bytes = create_synthetic_image_bytes(format="PNG")
    result = extract_features(img_bytes)

    assert "features" in result
    feats = result["features"]
    assert len(feats) == 4
    for key in ["variance", "skewness", "curtosis", "entropy"]:
        assert key in feats
        val = feats[key]
        assert isinstance(val, (int, float))
        assert not math.isnan(val) and not math.isinf(val)

    # Check metadata
    assert result["metadata"]["wavelet_used"] == DEFAULT_WAVELET
    assert result["metadata"]["processed_size"] == [400, 400]
    assert isinstance(result["out_of_distribution"], bool)
    print(f"[PASS] PNG Extracted: {feats}, Out-of-dist: {result['out_of_distribution']}")


def test_synthetic_jpeg_features():
    """Verify JPEG extraction works."""
    img_bytes = create_synthetic_image_bytes(format="JPEG", mode="RGB")
    result = extract_features(img_bytes)
    assert all(math.isfinite(result["features"][k]) for k in ["variance", "skewness", "curtosis", "entropy"])
    print(f"[PASS] JPEG Extracted: {result['features']}")


def test_synthetic_gif_features():
    """Verify GIF extraction works."""
    img_bytes = create_synthetic_image_bytes(format="GIF", mode="P")
    result = extract_features(img_bytes)
    assert all(math.isfinite(result["features"][k]) for k in ["variance", "skewness", "curtosis", "entropy"])
    print(f"[PASS] GIF Extracted: {result['features']}")


def test_corrupt_file_rejected():
    """Verify non-image bytes (e.g. text file renamed to .png) are rejected."""
    corrupt_bytes = b"This is just a plain text file pretending to be an image. 1234567890."
    rejected = False
    try:
        extract_features(corrupt_bytes)
    except ValueError as e:
        rejected = True
        assert "Corrupted or invalid image file" in str(e)
    assert rejected, "Expected ValueError was not raised for corrupt image bytes."
    print("[PASS] Corrupt file successfully rejected with ValueError.")


def test_empty_bytes_rejected():
    """Verify empty byte string is rejected."""
    rejected = False
    try:
        extract_features(b"")
    except ValueError:
        rejected = True
    assert rejected, "Expected ValueError was not raised for empty image bytes."
    print("[PASS] Empty payload rejected.")


def test_different_images_produce_different_features():
    """Verify two clearly different images produce distinct feature vectors."""
    # Pattern 1: dense horizontal lines
    im1 = Image.new("L", (400, 400), color=255)
    draw1 = ImageDraw.Draw(im1)
    for y in range(0, 400, 4):
        draw1.line([(0, y), (400, y)], fill=0, width=2)
    buf1 = io.BytesIO()
    im1.save(buf1, format="PNG")

    # Pattern 2: uniform solid white image
    im2 = Image.new("L", (400, 400), color=255)
    buf2 = io.BytesIO()
    im2.save(buf2, format="PNG")

    res1 = extract_features(buf1.getvalue())
    res2 = extract_features(buf2.getvalue())

    f1 = res1["features"]
    f2 = res2["features"]

    # At least variance or entropy must differ significantly
    assert f1["variance"] != f2["variance"], f"Variances should differ: {f1['variance']} vs {f2['variance']}"
    assert f2["variance"] == 0.0, f"Solid image variance should be 0.0: got {f2['variance']}"
    print(f"[PASS] Different images produce distinct features (Pattern var: {f1['variance']}, Solid var: {f2['variance']})")


def test_plain_white_detected_as_out_of_distribution():
    """Verify plain white or blank image triggers out-of-distribution."""
    im = Image.new("RGB", (400, 400), color=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    res = extract_features(buf.getvalue())
    assert res["out_of_distribution"] is True
    print(f"[PASS] Solid white image flagged as out_of_distribution: {res['out_of_range_features']}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FEATURE EXTRACTION SELF-TESTS")
    print("=" * 60)
    test_synthetic_png_features()
    test_synthetic_jpeg_features()
    test_synthetic_gif_features()
    test_corrupt_file_rejected()
    test_empty_bytes_rejected()
    test_different_images_produce_different_features()
    test_plain_white_detected_as_out_of_distribution()
    print("=" * 60)
    print("ALL FEATURE EXTRACTION TESTS PASSED!")
    print("=" * 60)
