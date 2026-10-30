import unittest

import pandas as pd

from preprocessing import DatasetValidationError, prepare_dataset, suggest_target_column


class PreprocessingTests(unittest.TestCase):
    def test_maps_common_binary_labels_and_excludes_identifiers(self):
        frame = pd.DataFrame(
            {
                "Flow ID": ["a", "b", "c", "d"],
                "Protocol": ["tcp", "udp", "tcp", "udp"],
                "Flow Duration": [10, 20, 30, 40],
                "Label": ["BENIGN", "DoS", "BENIGN", "DoS"],
            }
        )
        prepared = prepare_dataset(frame, "Label")
        self.assertEqual(prepared.binary_labels.tolist(), ["Normal", "Suspicious", "Normal", "Suspicious"])
        self.assertEqual(prepared.feature_columns, ["Protocol", "Flow Duration"])
        self.assertEqual(prepared.dropped_feature_columns, ["Flow ID"])

    def test_requires_both_binary_classes(self):
        frame = pd.DataFrame({"feature": [1, 2, 3], "label": ["normal"] * 3})
        with self.assertRaises(DatasetValidationError):
            prepare_dataset(frame, "label")

    def test_target_suggestion_is_conservative(self):
        self.assertEqual(suggest_target_column(["bytes", "Label"]), "Label")
        self.assertIsNone(suggest_target_column(["source_port", "duration"]))


if __name__ == "__main__":
    unittest.main()
