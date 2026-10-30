import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predict import predict_records
from train_model import train_and_evaluate


class PredictionTests(unittest.TestCase):
    def test_training_saves_pipeline_and_predicts_consistently(self):
        frame = pd.DataFrame(
            {
                "duration": list(range(20)),
                "protocol": ["tcp", "udp"] * 10,
                "label": ["BENIGN", "Attack"] * 10,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            result = train_and_evaluate(frame, "label", model_path)
            test_records = result["analysis_rows"][["duration", "protocol"]]
            predictions = predict_records(model_path, test_records)
        self.assertEqual(len(predictions), result["records_analyzed"])
        self.assertTrue(set(predictions).issubset({"Normal", "Suspicious"}))
        self.assertEqual(result["metrics"]["classes"], ["Normal", "Suspicious"])


if __name__ == "__main__":
    unittest.main()
