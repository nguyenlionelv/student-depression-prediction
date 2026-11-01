import os
import joblib
import logging
import hashlib
import json
from pathlib import Path
import numpy as np

from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

def save_artifacts(model, explainer, feature_names: list, out_dir: str):
    """Save model, explainer, and feature names to the target directory."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    explainer_path = os.path.join(out_dir, "explainer.pkl")
    features_path = os.path.join(out_dir, "feature_names.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    joblib.dump(feature_names, features_path)
    
    logging.info(f"Saved model to {model_path}")
    logging.info(f"Saved explainer to {explainer_path}")
    logging.info(f"Saved features to {features_path}")
    return {"model.pkl": Path(model_path), "explainer.pkl": Path(explainer_path),
            "feature_names.pkl": Path(features_path)}


def sha256_file(path) -> str:
    """Fingerprint an artifact without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_metadata(metadata: dict, path):
    Path(path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_evaluation_reports(evaluation: dict, cross_validation: dict, metadata: dict, out_dir) -> dict:
    """Create one run's reports; refuse to overwrite a previous run directory."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    paths = {name: directory / name for name in (
        "metrics.json", "classification_report.json", "cv_scores.csv",
        "confusion_matrix.png", "metadata.json",
    )}
    save_metadata({
        "schema_version": 1,
        "positive_label": 1,
        "test": evaluation["metrics"],
        "cv": cross_validation["summary"],
        "confusion_matrix": {"labels": evaluation["labels"], "rows": "actual", "columns": "predicted",
                             "values": evaluation["confusion_matrix"]},
    }, paths["metrics.json"])
    save_metadata(evaluation["classification_report"], paths["classification_report.json"])
    cross_validation["fold_scores"].to_csv(paths["cv_scores.csv"], index=False)
    figure = Figure(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay(np.asarray(evaluation["confusion_matrix"]), display_labels=evaluation["labels"]).plot(
        ax=figure.subplots(), cmap="Blues", colorbar=False,
    )
    figure.axes[0].set_title("Held-out test set")
    figure.savefig(paths["confusion_matrix.png"], dpi=160)
    metadata["report_artifacts"] = {name: {"sha256": sha256_file(path)}
                                    for name, path in paths.items() if name != "metadata.json"}
    save_metadata(metadata, paths["metadata.json"])
    logging.info("Saved evaluation reports to %s", directory)
    return paths
