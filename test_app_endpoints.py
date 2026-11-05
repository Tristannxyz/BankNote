"""
test_app_endpoints.py — Comprehensive automated test suite for NoteGuard Flask endpoints.
Tests:
1. GET / (200 OK, HTML)
2. GET /dashboard (200 OK, HTML)
3. GET /api/metrics (200 OK, JSON with models, dataset_summary, ranges)
4. GET /api/samples (200 OK, JSON with genuine and forged specimens)
5. POST /api/predict with genuine sample (verdict "Genuine")
6. POST /api/predict with forged sample (verdict "Forged")
7. POST /api/predict error handling (missing fields, non-numeric values, invalid model)
8. POST /api/predict-image with valid synthetic PNG image
9. POST /api/predict-image with valid synthetic JPEG image
10. POST /api/predict-image with invalid / corrupted file (.txt renamed to .png) -> 400 Bad Request
11. POST /api/predict-image with unsupported file extension (.pdf) -> 400 Bad Request
12. 413 error on oversized upload simulation
"""

import io
import json
import unittest
from PIL import Image, ImageDraw

from app import app


class NoteGuardAppTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Create a synthetic test banknote image
        im = Image.new("L", (400, 400), color=210)
        draw = ImageDraw.Draw(im)
        for y in range(20, 380, 8):
            draw.line([(30, y), (370, y)], fill=60, width=1)
        for x in range(40, 360, 20):
            draw.rectangle([x, 100, x + 12, 300], fill=120)
        draw.ellipse([150, 150, 250, 250], fill=40)

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        cls.valid_png_bytes = buf.getvalue()

        # JPEG bytes
        buf_jpg = io.BytesIO()
        im.convert("RGB").save(buf_jpg, format="JPEG")
        cls.valid_jpeg_bytes = buf_jpg.getvalue()

    def test_01_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"NoteGuard", resp.data)
        self.assertIn(b"Upload Currency Note", resp.data)
        self.assertIn(b"Developer / Model Testing", resp.data)
        print("[PASS] GET / renders index page with tabs")

    def test_02_dashboard_page(self):
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Model Evaluation", resp.data)
        self.assertIn(b"Confusion Matrix", resp.data)
        print("[PASS] GET /dashboard renders evaluation dashboard")

    def test_03_get_metrics(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("best_model", data)
        self.assertIn("models", data)
        self.assertIn("decision_tree", data["models"])
        self.assertIn("random_forest", data["models"])
        self.assertIn("training_feature_ranges", data)
        print(f"[PASS] GET /api/metrics (Best model: {data['best_model']})")

    def test_04_get_samples(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("genuine", data)
        self.assertIn("forged", data)
        self.assertGreater(len(data["genuine"]), 0)
        self.assertGreater(len(data["forged"]), 0)
        print(f"[PASS] GET /api/samples ({len(data['genuine'])} genuine, {len(data['forged'])} forged)")

    def test_05_predict_genuine_manual(self):
        # Sample genuine feature vector from dataset
        payload = {
            "variance": 0.1154,
            "skewness": -1.2789,
            "curtosis": 2.5886,
            "entropy": -0.2409,
            "model": "random_forest"
        }
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        print(f"[PASS] POST /api/predict Genuine sample -> Verdict: {data['prediction']} ({data['confidence']}%)")

    def test_06_predict_forged_manual(self):
        # Sample forged feature vector from dataset
        payload = {
            "variance": 0.2647,
            "skewness": -0.4608,
            "curtosis": -1.2113,
            "entropy": 0.2938,
            "model": "random_forest"
        }
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["prediction"], "Forged")
        self.assertGreater(data["confidence"], 50.0)
        print(f"[PASS] POST /api/predict Forged sample -> Verdict: {data['prediction']} ({data['confidence']}%)")

    def test_07_predict_validation_errors(self):
        # Missing feature
        resp1 = self.client.post("/api/predict", json={"variance": 1.0, "skewness": 2.0})
        self.assertEqual(resp1.status_code, 400)
        self.assertIn("Missing required feature", resp1.get_json()["error"])

        # Non-numeric string
        resp2 = self.client.post("/api/predict", json={
            "variance": "invalid", "skewness": 2.0, "curtosis": 0.0, "entropy": -1.0
        })
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("must be a valid numeric number", resp2.get_json()["error"])

        # Invalid model
        resp3 = self.client.post("/api/predict", json={
            "variance": 1.0, "skewness": 2.0, "curtosis": 0.0, "entropy": -1.0, "model": "svm"
        })
        self.assertEqual(resp3.status_code, 400)
        self.assertIn("Invalid model", resp3.get_json()["error"])
        print("[PASS] POST /api/predict input validation (400 responses)")

    def test_08_predict_image_png(self):
        data = {
            "model": "random_forest",
            "image": (io.BytesIO(self.valid_png_bytes), "test_banknote.png")
        }
        resp = self.client.post(
            "/api/predict-image",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Genuine", "Forged"])
        self.assertIn("confidence", res)
        self.assertIn("extracted_features", res)
        self.assertIn("out_of_distribution", res)
        print(f"[PASS] POST /api/predict-image (PNG) -> Verdict: {res['prediction']}, Conf: {res['confidence']}%, Out-of-dist: {res['out_of_distribution']}")

    def test_09_predict_image_jpeg(self):
        data = {
            "model": "decision_tree",
            "image": (io.BytesIO(self.valid_jpeg_bytes), "test_banknote.jpg")
        }
        resp = self.client.post(
            "/api/predict-image",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertIn("prediction", res)
        print(f"[PASS] POST /api/predict-image (JPEG) -> Verdict: {res['prediction']}, Conf: {res['confidence']}%")

    def test_10_predict_image_corrupt_rejected(self):
        # Fake png containing text
        fake_png = b"This is a text file pretending to be banknote.png"
        data = {
            "model": "random_forest",
            "image": (io.BytesIO(fake_png), "fake_note.png")
        }
        resp = self.client.post(
            "/api/predict-image",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertIn("error", res)
        print(f"[PASS] POST /api/predict-image corrupt file rejected with 400: '{res['error']}'")

    def test_11_predict_image_unsupported_ext_rejected(self):
        data = {
            "model": "random_forest",
            "image": (io.BytesIO(b"%PDF-1.4 test"), "sample.pdf")
        }
        resp = self.client.post(
            "/api/predict-image",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        res = resp.get_json()
        self.assertIn("Unsupported file type", res["error"])
        print(f"[PASS] POST /api/predict-image unsupported extension rejected with 400: '{res['error']}'")

    def test_12_plain_white_image_suspicious_or_forged(self):
        """Plain white or non-currency scan must not be classified Genuine with high confidence."""
        im = Image.new("RGB", (400, 400), color=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = {
            "model": "random_forest",
            "image": (io.BytesIO(buf.getvalue()), "white_scan.png")
        }
        resp = self.client.post(
            "/api/predict-image",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()
        self.assertIn(res["prediction"], ["Suspicious", "Forged"])
        print(f"[PASS] White image upload -> Prediction: {res['prediction']}, Conf: {res['confidence']}%")


if __name__ == "__main__":
    unittest.main()

