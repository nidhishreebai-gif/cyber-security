"""Dataset validation and feature preparation for offline traffic analysis."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd


NORMAL_LABELS = {
    "0",
    "0.0",
    "benign",
    "normal",
    "normal traffic",
    "normal_traffic",
    "normaltraffic",
    "non-attack",
    "non attack",
    "no attack",
    "no_attack",
    "legitimate",
    "legit",
}
TARGET_CANDIDATES = ("label", "class", "attack_cat", "attack category", "target", "y")
MAX_DATASET_ROWS = 200_000


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be safely prepared for training."""


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    binary_labels: pd.Series
    original_labels: pd.Series
    target_column: str
    feature_columns: list[str]
    dropped_feature_columns: list[str]
    dropped_rows: int
    label_categories: list[str]


def read_dataset(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Read a bounded CSV file and return a useful validation error on failure."""
    try:
        data = pd.read_csv(path, low_memory=False, nrows=MAX_DATASET_ROWS + 1)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise DatasetValidationError(
            "The file could not be read as a valid UTF-8 CSV dataset."
        ) from exc

    if data.empty:
        raise DatasetValidationError("The CSV file has no data rows.")
    if len(data) > MAX_DATASET_ROWS:
        raise DatasetValidationError(
            f"Datasets are limited to {MAX_DATASET_ROWS:,} rows."
        )
    minimum_columns = 2 if require_target else 1
    if len(data.columns) < minimum_columns:
        raise DatasetValidationError(
            "The CSV must contain at least one feature column"
            + (" and one target column." if require_target else ".")
        )

    data.columns = [str(column).strip() for column in data.columns]
    if any(not column for column in data.columns):
        raise DatasetValidationError("Column names must not be empty.")
    if len(set(data.columns)) != len(data.columns):
        raise DatasetValidationError("Column names must be unique.")
    return data


def suggest_target_column(columns: list[str]) -> str | None:
    """Suggest a common label column without silently selecting an uncertain one."""
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _is_identifier_or_leakage(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    exact_exclusions = {
        "id",
        "flow_id",
        "srcip",
        "dstip",
        "sourceip",
        "destinationip",
        "ip",
        "ip_address",
        "ip_src",
        "ip_dst",
        "src_ip",
        "source_ip",
        "src_ip_address",
        "source_ip_address",
        "dst_ip",
        "dest_ip",
        "dst_ip_address",
        "destination_ip_address",
        "destination_ip",
        "src_addr",
        "dst_addr",
        "source_address",
        "destination_address",
        "timestamp",
        "stime",
        "ltime",
        "date_time",
        "datetime",
        "attack_cat",
        "attack_category",
        "attack_type",
        "attack_label",
        "attack_class",
        "attackclass",
        "attack_name",
        "attack",
        "is_attack",
        "is_benign",
        "binary_label",
        "binary_class",
        "class_label",
        "label_name",
        "class",
        "label",
        "target",
    }
    if normalized in exact_exclusions:
        return True
    tokens = set(normalized.split("_"))
    if tokens.intersection({"label", "class", "target"}):
        return True
    return any(
        token in normalized
        for token in ("flow_id", "timestamp", "datetime", "start_time", "end_time")
    )


def _binary_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in NORMAL_LABELS:
        return "Normal"
    return "Suspicious"


def prepare_dataset(data: pd.DataFrame, target_column: str) -> PreparedDataset:
    """Clean a dataset and separate labels from leakage-prone or unusable fields."""
    if target_column not in data.columns:
        raise DatasetValidationError("Select a target column present in the uploaded CSV.")

    cleaned = data.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    before_filtering = len(cleaned)
    cleaned = cleaned.dropna(subset=[target_column]).copy()
    if cleaned.empty:
        raise DatasetValidationError("The selected target column contains no labels.")

    original_labels = cleaned[target_column].astype(str).str.strip()
    valid_labels = original_labels.ne("") & original_labels.str.lower().ne("nan")
    cleaned = cleaned.loc[valid_labels].copy()
    original_labels = original_labels.loc[valid_labels]
    if original_labels.nunique() < 2:
        raise DatasetValidationError("At least two target classes are required.")

    features = cleaned.drop(columns=[target_column])
    dropped_columns: list[str] = []
    for column in list(features.columns):
        values = features[column]
        if _is_identifier_or_leakage(str(column)):
            dropped_columns.append(str(column))
            features = features.drop(columns=[column])
            continue

        numeric = pd.to_numeric(values, errors="coerce")
        non_missing = values.notna().sum()
        if non_missing and numeric.notna().sum() / non_missing >= 0.9:
            features[column] = numeric
        elif values.nunique(dropna=True) > max(100, int(len(features) * 0.05)):
            dropped_columns.append(str(column))
            features = features.drop(columns=[column])

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise DatasetValidationError(
            "No usable predictive features remain after removing identifiers and labels."
        )

    binary_labels = original_labels.map(_binary_label)
    if binary_labels.nunique() < 2:
        raise DatasetValidationError(
            "The target must include at least one normal and one suspicious label."
        )

    return PreparedDataset(
        features=features,
        binary_labels=binary_labels,
        original_labels=original_labels,
        target_column=target_column,
        feature_columns=list(features.columns),
        dropped_feature_columns=dropped_columns,
        dropped_rows=before_filtering - len(features),
        label_categories=sorted(original_labels.unique().tolist()),
    )
