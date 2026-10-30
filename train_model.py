"""Reproducible Random Forest training and held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation import evaluate_predictions
from preprocessing import DatasetValidationError, prepare_dataset, read_dataset


RANDOM_STATE = 42
TEST_SIZE = 0.25


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate(
    data: pd.DataFrame,
    target_column: str,
    model_path: str | Path | None = None,
    synthetic_demo: bool = False,
) -> dict:
    prepared = prepare_dataset(data, target_column)
    class_counts = prepared.binary_labels.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 2:
        raise DatasetValidationError(
            "Each binary class needs at least two rows for a stratified train/test split."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            prepared.features,
            prepared.binary_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=prepared.binary_labels,
        )
    except ValueError as exc:
        raise DatasetValidationError(
            "The dataset is too small to create a stratified train/test split."
        ) from exc

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)
    analysis_rows = data.loc[X_test.index].copy()
    analysis_rows["Prediction"] = predictions
    analysis_rows["Actual class"] = y_test

    artifact = {
        "pipeline": pipeline,
        "feature_columns": prepared.feature_columns,
        "target_column": target_column,
        "class_labels": prepared.label_categories,
        "random_state": RANDOM_STATE,
        "synthetic_demo": synthetic_demo,
    }
    if model_path is not None:
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination)

    suspicious = analysis_rows[analysis_rows["Prediction"] == "Suspicious"]
    return {
        "artifact": artifact,
        "metrics": metrics,
        "analysis_rows": analysis_rows,
        "feature_columns": prepared.feature_columns,
        "dropped_feature_columns": prepared.dropped_feature_columns,
        "dropped_rows": prepared.dropped_rows,
        "target_column": target_column,
        "original_label_categories": prepared.label_categories,
        "records_analyzed": len(analysis_rows),
        "normal_count": int((analysis_rows["Prediction"] == "Normal").sum()),
        "suspicious_count": len(suspicious),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest using a labeled CSV dataset."
    )
    parser.add_argument("dataset", type=Path, help="Path to a labeled CSV dataset")
    parser.add_argument("--target", required=True, help="Target/label column name")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/traffic_random_forest.joblib"),
        help="Output path for the saved model pipeline",
    )
    arguments = parser.parse_args()

    try:
        dataset = read_dataset(arguments.dataset)
        result = train_and_evaluate(dataset, arguments.target, arguments.model)
    except (DatasetValidationError, OSError) as exc:
        parser.error(str(exc))

    print(json.dumps(result["metrics"], indent=2))
    print(f"\nSaved model pipeline to: {arguments.model}")


if __name__ == "__main__":
    main()
