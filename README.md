# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

A Flask-based machine learning application for offline classification of unidirectional IP traffic records as Normal or Suspicious. The project combines dataset validation, preprocessing, Random Forest training, prediction, and a lightweight dashboard for evaluating model performance without capturing live network traffic.

## Problem Statement

Modern network environments generate large volumes of flow records, making it difficult to separate normal traffic from suspicious traffic using manual inspection alone. This project demonstrates a practical and reproducible way to preprocess network flow data, train a supervised model, and interpret classification results in a controlled academic setting.

## Objectives

- Validate uploaded CSV datasets before training.
- Prepare feature sets suitable for machine learning.
- Train a binary classifier for Normal vs Suspicious predictions.
- Evaluate results using standard metrics and confusion matrices.
- Provide a dashboard for analysis, prediction, and performance review.
- Keep the workflow offline and explain that predictions are decision-support signals, not definitive threats.

## Key Features

| Feature | Description |
| --- | --- |
| CSV upload and validation | Accepts labeled CSV files and checks for missing or malformed data. |
| Leakage-aware preprocessing | Removes duplicate rows, invalid labels, and identifier columns before model training. |
| Random Forest model | Trains a reproducible scikit-learn classifier using a stratified 75/25 split. |
| Model evaluation | Provides accuracy, precision, recall, F1-score, confusion matrix, and classification report. |
| Prediction workflow | Classifies new unlabeled feature-only CSV files using the trained pipeline. |
| Flask dashboard | Displays summary cards, analysis tables, threat views, and performance metrics in the browser. |
| Synthetic demo mode | Includes clearly labeled synthetic data to demonstrate the app flow without misrepresenting it as real traffic. |

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Application | Python, Flask |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, joblib |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js |
| Testing | unittest |

## System Architecture

```text
User uploads labeled CSV
        ?
Dataset validation and preprocessing
        ?
Feature selection and leakage filtering
        ?
Stratified train/test split (random_state=42)
        ?
Random Forest training and evaluation
        ?
Saved model artifact + analysis results
        ?
Flask dashboard for analysis, predictions, and performance metrics
```

## Project Folder Structure

```text
cyber-security/
+-- app.py                          # Flask application and routes
+-- train_model.py                  # Training and evaluation pipeline
+-- preprocessing.py                # CSV validation and feature preparation
+-- predict.py                      # Model loading and prediction helper
+-- evaluation.py                   # Performance metrics and confusion matrix logic
+-- demo_data.py                    # Synthetic demonstration dataset generator
+-- requirements.txt                # Python dependencies
+-- README.md                       # Project documentation
+-- .gitignore                      # Ignored files and directories
+-- datasets/                       # Uploaded datasets and runtime artifacts
¦   +-- .gitkeep
¦   +-- ...
+-- models/                         # Saved trained model and generated outputs
¦   +-- .gitkeep
¦   +-- ...
+-- templates/                      # Jinja HTML pages
¦   +-- base.html
¦   +-- index.html
¦   +-- upload.html
¦   +-- analysis.html
¦   +-- predict.html
¦   +-- threats.html
¦   +-- performance.html
¦   +-- about.html
+-- static/
¦   +-- css/
¦   ¦   +-- style.css
¦   +-- js/
¦   ¦   +-- dashboard.js
¦   +-- images/
+-- tests/
¦   +-- test_preprocessing.py
¦   +-- test_prediction.py
¦   +-- test_routes.py
+-- .venv/                          # Ignored local virtual environment
```

## Installation

### Prerequisites

- Python 3.10+
- pip
- Windows, Linux, or macOS

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or .\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

This project is designed for offline CSV-based analysis. It does not capture live traffic or connect to external monitoring systems.

### Official dataset sources

- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Notes

- Use a flow or network-record CSV with at least one feature column and a label column.
- Keep dataset sizes under the project limit for safe local processing.
- Review the original dataset documentation and label mappings before training.
- Synthetic demo data is included only to show the application workflow. It must never be mistaken for real network traffic or actual cyber-attack samples.

## Model Training

Place a labeled dataset in the `datasets/` directory and run:

```bash
python train_model.py datasets/your_dataset.csv --target Label --model models/traffic_random_forest.joblib
```

The script:

- reads the CSV with validation checks,
- prepares numerical and categorical features,
- splits data into train and test sets,
- trains a Random Forest classifier,
- saves the trained pipeline,
- prints evaluation metrics.

## Application Execution

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Web workflow

1. Upload a labeled CSV dataset.
2. Confirm the target/label column.
3. Train and evaluate the model.
4. Review analysis and performance pages.
5. Upload new feature-only CSV records for prediction.

## Model Evaluation

The application evaluates performance using the held-out test split and reports:

- Accuracy
- Precision for the Suspicious class
- Recall for the Suspicious class
- F1-score for the Suspicious class
- Confusion matrix
- Classification report

These metrics are helpful for local experimentation and academic demonstration, but they should not be treated as real-world detection guarantees.

## Screenshots

The project includes a web dashboard with the following views:

- Overview page with summary metrics
- Dataset upload and target selection page
- Traffic analysis page
- Prediction page
- Threat detection page
- Model performance page
- About page

Screenshots can be added in the repository once captured from the running app.

## Limitations

- This is a local prototype, not a production IDS deployment.
- It uses offline CSV data rather than live packet capture.
- Synthetic demo data is intentionally illustrative and not representative of real traffic.
- The model is binary only (Normal vs Suspicious).
- Classification results require human verification before action is taken.

## Future Enhancements

- Add support for multiclass attack-type detection.
- Integrate real-time anomaly detection workflows.
- Add model explainability and SHAP-style insights.
- Support broader dataset formats and ingestion pipelines.
- Add database-backed history, alerts, and user authentication.
- Improve deployment packaging for cloud or containerized environments.

## Contributors

| Name | Role |
| --- | --- |
| Nidhishree Bai | Project developer |
| OpenAI Codex | Project support and repository organization |

## Notes

This repository is intended as a mini-project demonstration for academic or engineering use. It should be used responsibly and only with properly licensed datasets. Do not upload sensitive personal or enterprise network data to public repositories.
