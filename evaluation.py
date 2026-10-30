"""Model evaluation helpers."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASSES = ["Normal", "Suspicious"]


def evaluate_predictions(y_true, y_pred) -> dict:
    """Return metrics with a fixed binary class order for consistent display."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="Suspicious", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="Suspicious", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="Suspicious", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True
        ),
        "classes": CLASSES,
    }
