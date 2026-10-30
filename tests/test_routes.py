import io
import tempfile
import unittest
from pathlib import Path

from app import create_app


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test-secret",
                "UPLOAD_FOLDER": root / "datasets",
                "MODEL_PATH": root / "models" / "model.joblib",
                "RESULTS_PATH": root / "models" / "results.json",
                "ANALYSIS_PATH": root / "models" / "analysis.csv",
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_navigation_pages_load(self):
        for route in ("/", "/upload", "/analysis", "/predict", "/threats", "/performance", "/about"):
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)

    def test_upload_then_train_and_view_results(self):
        csv_data = (
            "duration,protocol,label\n"
            "1,tcp,BENIGN\n2,udp,Attack\n3,tcp,BENIGN\n4,udp,Attack\n"
            "5,tcp,BENIGN\n6,udp,Attack\n7,tcp,BENIGN\n8,udp,Attack\n"
            "9,tcp,BENIGN\n10,udp,Attack\n11,tcp,BENIGN\n12,udp,Attack\n"
        )
        response = self.client.post(
            "/upload",
            data={"dataset": (io.BytesIO(csv_data.encode()), "traffic.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Target / label column", response.data)
        response = self.client.post("/train", data={"target_column": "label"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/analysis").status_code, 200)
        self.assertEqual(self.client.get("/performance").status_code, 200)
        prediction_csv = (
            "duration,protocol\n13,tcp\n14,udp\n"
        )
        response = self.client.post(
            "/predict",
            data={"dataset": (io.BytesIO(prediction_csv.encode()), "new-flows.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)
        prediction_page = self.client.get("/predict")
        self.assertEqual(prediction_page.status_code, 200)
        self.assertIn(b"new-flows.csv", prediction_page.data)

    def test_rejects_non_csv_file(self):
        response = self.client.post(
            "/upload",
            data={"dataset": (io.BytesIO(b"not a dataset"), "traffic.exe")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Only .csv files are supported", response.data)

    def test_synthetic_demo_is_explicitly_labeled(self):
        response = self.client.post("/demo")
        self.assertEqual(response.status_code, 302)
        dashboard = self.client.get("/")
        self.assertIn(b"SYNTHETIC DEMONSTRATION", dashboard.data)
        self.assertIn(b"not real network traffic", dashboard.data)
        prediction_page = self.client.get("/predict")
        self.assertIn(b"DEMO MODEL ONLY", prediction_page.data)
        response = self.client.post(
            "/predict",
            data={"dataset": (io.BytesIO(b"duration,protocol\n1,tcp\n"), "new.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertIn(b"Train on a labeled dataset before classifying new records", response.data)


if __name__ == "__main__":
    unittest.main()
