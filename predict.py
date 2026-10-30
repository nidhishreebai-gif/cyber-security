"""Prediction using a saved, project-generated preprocessing/model pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import pandas as pd


class ModelPredictionError(ValueError):
    """Raised when saved-model prediction inputs are incompatible."""


def load_model(model_path: str | Path) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise ModelPredictionError("No trained model is available. Train a model first.")
    try:
        artifact = joblib.load(path)
    except (
        OSError,
        EOFError,
        ValueError,
        KeyError,
        ImportError,
        AttributeError,
        TypeError,
        RuntimeError,
        pickle.UnpicklingError,
    ) as exc:
        raise ModelPredictionError("The saved model could not be loaded.") from exc
    if (
        not isinstance(artifact, dict)
        or "pipeline" not in artifact
        or not isinstance(artifact.get("feature_columns"), list)
    ):
        raise ModelPredictionError("The saved model artifact is invalid.")
    return artifact


def predict_records(
    model_path: str | Path,
    records: pd.DataFrame,
    *,
    artifact: dict | None = None,
) -> list[str]:
    artifact = artifact or load_model(model_path)
    expected_columns = artifact.get("feature_columns", [])
    missing_columns = [column for column in expected_columns if column not in records]
    if missing_columns:
        raise ModelPredictionError(
            "The prediction dataset is missing required feature columns: "
            + ", ".join(missing_columns[:10])
        )
    predictions = artifact["pipeline"].predict(records[expected_columns])
    return [str(prediction) for prediction in predictions]
