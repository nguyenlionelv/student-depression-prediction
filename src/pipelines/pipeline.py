import os, sys
import argparse
import logging
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import subprocess
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import mlflow
# import warnings
import shap
from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.pipelines.preprocessing import run_pipeline, DEFAULT_RAW_DATA_PATH, DEFAULT_PROCESSED_DATA_PATH
from src.pipelines.train import prepare_data, train_model
from src.pipelines.evaluate import evaluate_model, evaluate_model_kfold
from src.pipelines.save import save_artifacts, save_evaluation_reports, save_metadata, sha256_file

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Student Depression Model Training Pipeline.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw CSV data file")
    default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="Directory to save model artifacts")
    parser.add_argument("--reports-dir", type=str, default=None, help="Report root; defaults to <model-dir>/reports")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking backend")
    return parser.parse_args(argv)


def capture_provenance() -> dict:
    """Record the working code and environment, including uncommitted changes."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True,
        ).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in ("scikit-learn", "xgboost", "lightgbm", "pandas", "numpy", "shap", "mlflow", "joblib", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = sorted(Path(__file__).resolve().parent.glob("*.py")) + [root / "requirements.txt"]
    return {
        "git": {"commit": commit, "dirty": dirty},
        "environment": {"python": sys.version, "packages": packages},
        "source_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in sources},
    }


def model_parameters(model) -> dict:
    parameters = {}
    for key, value in model.get_params(deep=True).items():
        if value is None or isinstance(value, (str, bool, int, float)):
            # XGBoost exposes missing=nan; encode this as a string in strict JSON.
            parameters[key] = str(value) if isinstance(value, float) and not math.isfinite(value) else value
    return parameters

def main(argv=None):
    args = get_args(argv)
    logging.info("Starting training pipeline...")
    
    cached = Path(DEFAULT_PROCESSED_DATA_PATH).exists()
    source = Path(DEFAULT_PROCESSED_DATA_PATH if cached else (args.data_path or DEFAULT_RAW_DATA_PATH))
    input_source = {"file": source.name, "kind": "processed_cache" if cached else "raw_csv",
                    "sha256": sha256_file(source)}
    df = run_pipeline(data_path=args.data_path) if args.data_path else run_pipeline()
    X, y = prepare_data(df)
    feature_names = X.columns.tolist()
    logging.info(f"Using {len(feature_names)} features.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    provenance = capture_provenance()
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(experiment_name="Stacking Ensemble")
    with mlflow.start_run(experiment_id=exp.experiment_id,run_name="Stacking Ensemble") as run:

        model = train_model(X_train, y_train)
        evaluation = evaluate_model(model, X_test, y_test)
        cv = evaluate_model_kfold(model, X, y)
        parameters = model_parameters(model)
        metadata = {
            "schema_version": 1, "status": "evaluated", "created_at": datetime.now(timezone.utc).isoformat(),
            "mlflow": {"run_id": run.info.run_id, "experiment_id": exp.experiment_id},
            **provenance,
            "dataset": {"input_source": input_source, "rows": len(df), "features": feature_names,
                        "evaluated_table_sha256": hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest(),
                        "fingerprint_format": "pandas DataFrame.to_csv(index=False, lineterminator=LF), UTF-8",
                        "class_counts": {str(label): int(count) for label, count in y.value_counts().items()}},
            "evaluation_protocol": {
                "test_size": 0.2, "split_random_state": 42, "stratified": True,
                "train_rows": len(X_train), "test_rows": len(X_test),
                "cv_splits": 5, "cv_shuffle": False, "cv_scope": "full_dataset", "cv_std_ddof": 0,
                "positive_label": 1,
                "limitations": [
                    "Preprocessing is unchanged and runs before train/test splitting, or reuses a processed cache.",
                    "CV is unchanged and uses the full dataset, including the holdout rows.",
                ],
            },
            "model_type": type(model).__name__, "model_parameters": parameters, "model_artifacts": {},
        }
        report_dir = Path(args.reports_dir or Path(args.model_dir) / "reports") / run.info.run_id
        reports = save_evaluation_reports(evaluation, cv, metadata, report_dir)
        stage = "logging_evaluation"
        try:
            metrics = {f"test_{name}": value for name, value in evaluation["metrics"].items()}
            metrics.update({f"cv_{name}_{stat}": value for name, stats in cv["summary"].items() for stat, value in stats.items()})
            mlflow.log_metrics(metrics)
            mlflow.log_params(parameters)
            mlflow.log_params({"split_random_state": 42, "test_size": 0.2, "cv_splits": 5, "cv_scope": "full_dataset"})
            mlflow.log_artifacts(str(report_dir), artifact_path="reports")
            root = Path(__file__).resolve().parents[2]
            for relative_path in provenance["source_sha256"]:
                mlflow.log_artifact(str(root / relative_path), artifact_path="source")

            stage = "creating_shap_explainer"
            logging.info("Creating SHAP KernelExplainer with K-Means background")
            background = shap.kmeans(X_train, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            stage = "saving_model_artifacts"
            artifacts = save_artifacts(model, explainer, feature_names, args.model_dir)
            metadata["model_artifacts"] = {name: {"sha256": sha256_file(path), "mlflow_path": f"model/{name}"}
                                           for name, path in artifacts.items()}
            stage = "logging_model_artifacts"
            for path in artifacts.values():
                mlflow.log_artifact(str(path), artifact_path="model")
            metadata["status"] = "complete"
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_metadata(metadata, reports["metadata.json"])
            mlflow.log_artifact(str(reports["metadata.json"]), artifact_path="reports")
        except Exception as error:
            metadata["status"] = "failed"
            metadata["failure"] = {"stage": stage, "exception_type": type(error).__name__}
            save_metadata(metadata, reports["metadata.json"])
            try:
                mlflow.log_artifact(str(reports["metadata.json"]), artifact_path="reports")
            except Exception:
                logging.exception("Failed to log metadata; local reports remain at %s", report_dir)
            raise
        logging.info("Pipeline completed successfully. Reports: %s", report_dir)
        return report_dir

if __name__ == "__main__":
    main()
