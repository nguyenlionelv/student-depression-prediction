import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

def evaluate_model(model: StackingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Keep console output and return the held-out results for persistence."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        },
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "labels": [0, 1],
    }

def evaluate_model_kfold(model: StackingClassifier, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> dict:
    """Expose fold scores without changing the existing cross_val_score calls.

    As before, integer cv uses sklearn's default unshuffled classification folds.
    The shuffle/random_state arguments remain unused for compatibility.
    """
    acc = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy", n_jobs=-1)
    f1 = cross_val_score(model, X, y, cv=n_splits, scoring="f1", n_jobs=-1)
    rec = cross_val_score(model, X, y, cv=n_splits, scoring="recall", n_jobs=-1)
    prec = cross_val_score(model, X, y, cv=n_splits, scoring="precision", n_jobs=-1)

    logging.info("Result K-fold:")
    logging.info(f"Accuracy: {acc.mean():.4f} +/- {acc.std():.4f}")
    logging.info(f"Precision: {prec.mean():.4f} +/- {prec.std():.4f}")
    logging.info(f"Recall: {rec.mean():.4f} +/- {rec.std():.4f}")
    logging.info(f"F1-score: {f1.mean():.4f} +/- {f1.std():.4f}")
    scores = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return {
        "summary": {name: {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
                    for name, values in scores.items()},
        "fold_scores": pd.DataFrame({"fold": range(1, n_splits + 1), **scores}),
    }
