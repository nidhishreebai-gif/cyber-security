"""Flask dashboard for offline, dataset-based traffic classification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from preprocessing import (
    DatasetValidationError,
    read_dataset,
    suggest_target_column,
)
from demo_data import make_synthetic_demo_dataset
from predict import ModelPredictionError, load_model, predict_records
from train_model import train_and_evaluate


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "datasets"
MODEL_FOLDER = BASE_DIR / "models"
MODEL_PATH = MODEL_FOLDER / "traffic_random_forest.joblib"
RESULTS_PATH = MODEL_FOLDER / "latest_results.json"
ANALYSIS_PATH = MODEL_FOLDER / "latest_analysis.csv"
ALLOWED_EXTENSION = ".csv"


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or os.urandom(32),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        MODEL_PATH=MODEL_PATH,
        RESULTS_PATH=RESULTS_PATH,
        ANALYSIS_PATH=ANALYSIS_PATH,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULTS_PATH"]).parent.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(413)
    def file_too_large(_error):
        flash("Upload is too large. The maximum file size is 50 MB.", "error")
        return redirect(url_for("upload"))

    @app.route("/")
    def index():
        results = _load_results(app.config["RESULTS_PATH"])
        rows = _load_analysis(app.config["ANALYSIS_PATH"])
        suspicious = [row for row in rows if row.get("Prediction") == "Suspicious"]
        return render_template(
            "index.html",
            results=results,
            suspicious_rows=suspicious[:5],
            feature_columns=(results or {}).get("feature_columns", []),
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            uploaded = request.files.get("dataset")
            if uploaded is None or not uploaded.filename:
                flash("Choose a CSV file to upload.", "error")
                return redirect(url_for("upload"))
            filename = secure_filename(uploaded.filename)
            if Path(filename).suffix.lower() != ALLOWED_EXTENSION:
                flash("Only .csv files are supported.", "error")
                return redirect(url_for("upload"))

            destination = Path(app.config["UPLOAD_FOLDER"]) / "current_upload.csv"
            try:
                uploaded.save(destination)
                data = read_dataset(destination)
            except DatasetValidationError as exc:
                destination.unlink(missing_ok=True)
                flash(str(exc), "error")
                return redirect(url_for("upload"))
            except OSError:
                current_app.logger.exception("Uploaded dataset could not be saved.")
                flash("The uploaded dataset could not be saved. Check local disk space and folder permissions.", "error")
                return redirect(url_for("upload"))

            columns = data.columns.tolist()
            target_suggestion = suggest_target_column(columns)
            return render_template(
                "upload.html",
                columns=columns,
                dataset_name=filename,
                record_count=len(data),
                target_suggestion=target_suggestion,
            )

        return render_template("upload.html")

    @app.route("/train", methods=["POST"])
    def train():
        target_column = request.form.get("target_column", "").strip()
        dataset_path = Path(app.config["UPLOAD_FOLDER"]) / "current_upload.csv"
        if not dataset_path.is_file():
            flash("Upload a CSV dataset before training.", "error")
            return redirect(url_for("upload"))
        try:
            data = read_dataset(dataset_path)
            result = train_and_evaluate(
                data, target_column, model_path=app.config["MODEL_PATH"]
            )
            result["analysis_rows"].to_csv(app.config["ANALYSIS_PATH"], index=False)
            _clear_prediction_results(app.config["RESULTS_PATH"])
            summary = {
                "dataset_name": request.form.get("dataset_name", "Uploaded dataset"),
                "synthetic_demo": False,
                "target_column": result["target_column"],
                "feature_columns": result["feature_columns"],
                "dropped_feature_columns": result["dropped_feature_columns"],
                "dropped_rows": result["dropped_rows"],
                "original_label_categories": result["original_label_categories"],
                "records_analyzed": result["records_analyzed"],
                "normal_count": result["normal_count"],
                "suspicious_count": result["suspicious_count"],
                "metrics": result["metrics"],
            }
            Path(app.config["RESULTS_PATH"]).write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except (DatasetValidationError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("upload"))
        except OSError:
            current_app.logger.exception("Training artifacts could not be saved.")
            flash("Training results could not be saved. Check local disk space and folder permissions.", "error")
            return redirect(url_for("upload"))
        flash("Model trained and evaluated on a held-out test split.", "success")
        return redirect(url_for("analysis"))

    @app.route("/demo", methods=["POST"])
    def demo():
        data = make_synthetic_demo_dataset()
        try:
            result = train_and_evaluate(
                data, "label", model_path=app.config["MODEL_PATH"], synthetic_demo=True
            )
            result["analysis_rows"].to_csv(app.config["ANALYSIS_PATH"], index=False)
            _clear_prediction_results(app.config["RESULTS_PATH"])
            summary = {
                "dataset_name": "Synthetic demonstration dataset",
                "synthetic_demo": True,
                "target_column": result["target_column"],
                "feature_columns": result["feature_columns"],
                "dropped_feature_columns": result["dropped_feature_columns"],
                "dropped_rows": result["dropped_rows"],
                "original_label_categories": result["original_label_categories"],
                "records_analyzed": result["records_analyzed"],
                "normal_count": result["normal_count"],
                "suspicious_count": result["suspicious_count"],
                "metrics": result["metrics"],
            }
            Path(app.config["RESULTS_PATH"]).write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except (DatasetValidationError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("upload"))
        except OSError:
            current_app.logger.exception("Synthetic demo artifacts could not be saved.")
            flash("Demo results could not be saved. Check local disk space and folder permissions.", "error")
            return redirect(url_for("upload"))
        flash(
            "Synthetic demonstration complete. These metrics do not represent real traffic or detection performance.",
            "success",
        )
        return redirect(url_for("analysis"))

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        prediction_results = _load_results(
            Path(app.config["RESULTS_PATH"]).with_name("latest_predictions.json")
        )
        prediction_rows = _load_analysis(
            Path(app.config["ANALYSIS_PATH"]).with_name("latest_predictions.csv")
        )

        if request.method == "POST":
            if not Path(app.config["MODEL_PATH"]).is_file():
                flash("Train a model before predicting new records.", "error")
                return redirect(url_for("predict"))
            try:
                artifact = load_model(app.config["MODEL_PATH"])
            except ModelPredictionError as exc:
                flash(str(exc), "error")
                return redirect(url_for("predict"))
            if artifact.get("synthetic_demo"):
                flash(
                    "This model was trained on synthetic demo data. Train on a labeled dataset before classifying new records.",
                    "error",
                )
                return redirect(url_for("predict"))
            uploaded = request.files.get("dataset")
            if uploaded is None or not uploaded.filename:
                flash("Choose a CSV file to classify.", "error")
                return redirect(url_for("predict"))
            filename = secure_filename(uploaded.filename)
            if Path(filename).suffix.lower() != ALLOWED_EXTENSION:
                flash("Only .csv files are supported.", "error")
                return redirect(url_for("predict"))

            source_path = Path(app.config["UPLOAD_FOLDER"]) / "prediction_upload.csv"
            try:
                uploaded.save(source_path)
                records = read_dataset(source_path, require_target=False)
                predictions = predict_records(
                    app.config["MODEL_PATH"], records, artifact=artifact
                )
                records["Prediction"] = predictions
                prediction_path = Path(app.config["ANALYSIS_PATH"]).with_name(
                    "latest_predictions.csv"
                )
                records.to_csv(prediction_path, index=False)
                summary = {
                    "dataset_name": filename,
                    "records_analyzed": len(records),
                    "normal_count": predictions.count("Normal"),
                    "suspicious_count": predictions.count("Suspicious"),
                }
                Path(app.config["RESULTS_PATH"]).with_name(
                    "latest_predictions.json"
                ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except (DatasetValidationError, ModelPredictionError, ValueError) as exc:
                source_path.unlink(missing_ok=True)
                flash(str(exc), "error")
                return redirect(url_for("predict"))
            except OSError:
                current_app.logger.exception("Prediction records could not be saved.")
                source_path.unlink(missing_ok=True)
                flash("Prediction results could not be saved. Check local disk space and folder permissions.", "error")
                return redirect(url_for("predict"))

            flash(
                "New records classified with the saved model. Predictions are not confirmed incidents.",
                "success",
            )
            return redirect(url_for("predict"))

        model_artifact = None
        if Path(app.config["MODEL_PATH"]).is_file():
            try:
                model_artifact = load_model(app.config["MODEL_PATH"])
            except ModelPredictionError as exc:
                flash(str(exc), "error")

        return render_template(
            "predict.html",
            results=prediction_results,
            rows=prediction_rows,
            model_available=bool(model_artifact)
            and not model_artifact.get("synthetic_demo", False),
            model_synthetic=bool(model_artifact and model_artifact.get("synthetic_demo")),
        )

    @app.route("/analysis")
    def analysis():
        results = _load_results(app.config["RESULTS_PATH"])
        rows = _load_analysis(app.config["ANALYSIS_PATH"])
        return render_template("analysis.html", results=results, rows=rows)

    @app.route("/threats")
    def threats():
        results = _load_results(app.config["RESULTS_PATH"])
        rows = _load_analysis(app.config["ANALYSIS_PATH"])
        suspicious = [row for row in rows if row.get("Prediction") == "Suspicious"]
        return render_template(
            "threats.html",
            results=results,
            rows=suspicious,
            feature_columns=(results or {}).get("feature_columns", []),
        )

    @app.route("/performance")
    def performance():
        results = _load_results(app.config["RESULTS_PATH"])
        return render_template("performance.html", results=results)

    @app.route("/about")
    def about():
        return render_template("about.html")

    return app


def _load_results(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        current_app.logger.exception("Saved dashboard results could not be read.")
        flash("Saved results could not be loaded. Run the dataset workflow again.", "error")
        return None


def _clear_prediction_results(results_path: str | Path) -> None:
    results_file = Path(results_path).with_name("latest_predictions.json")
    analysis_file = Path(results_path).with_name("latest_predictions.csv")
    results_file.unlink(missing_ok=True)
    analysis_file.unlink(missing_ok=True)


def _load_analysis(path: str | Path) -> list[dict]:
    try:
        frame = pd.read_csv(path, nrows=500)
    except FileNotFoundError:
        return []
    except (pd.errors.ParserError, UnicodeDecodeError, OSError):
        current_app.logger.exception("Saved analysis records could not be read.")
        flash("Saved analysis records could not be loaded. Run the dataset workflow again.", "error")
        return []
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
