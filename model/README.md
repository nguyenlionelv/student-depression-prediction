# Model artifacts

The selected model is the completed Stacking Ensemble run
`6784a0d69b0d4daa9a37ac37b5d961ea`. Its three Joblib artifacts were copied from
`trial/` into this directory for the API to load on its next startup.

Model binaries remain local and are ignored by Git. This README and the selected
evaluation reports are versioned.

```text
model/
├── README.md
├── model.pkl
├── explainer.pkl
├── feature_names.pkl
└── reports/
    └── 6784a0d69b0d4daa9a37ac37b5d961ea/
        ├── metrics.json
        ├── classification_report.json
        ├── cv_scores.csv
        ├── confusion_matrix.png
        └── metadata.json
```

| File | Contents |
| --- | --- |
| `model.pkl` | Trained Stacking classifier used to generate predictions. |
| `explainer.pkl` | SHAP explainer used to calculate feature contributions for predictions. |
| `feature_names.pkl` | Ordered list of feature names expected by the model. |

The checksums in [metadata.json](reports/6784a0d69b0d4daa9a37ac37b5d961ea/metadata.json)
identify the exact model, explainer, and feature-name files belonging to this run.

## Evaluation results

The run used 27,901 rows from the processed dataset. The stratified 80/20 split
used `random_state=42`: 22,320 training rows and 5,581 test rows. Precision, recall,
and F1 below use class `1` as the positive class.

| Metric | Holdout test (%) | 5-fold CV mean ± std (%) |
| --- | ---: | ---: |
| Accuracy | 83.84 | 84.21 ± 0.33 |
| Precision | 85.02 | 85.34 ± 0.49 |
| Recall | 87.88 | 88.19 ± 0.30 |
| F1-score | 86.43 | 86.74 ± 0.25 |

Values are rounded from [metrics.json](reports/6784a0d69b0d4daa9a37ac37b5d961ea/metrics.json).
The ± values are the standard deviations across folds, expressed in percentage
points. CV used five unshuffled folds over the full dataset.

## Report files

| File | Contents |
| --- | --- |
| [metrics.json](reports/6784a0d69b0d4daa9a37ac37b5d961ea/metrics.json) | Test metrics, CV means and standard deviations, and confusion-matrix counts. |
| [classification_report.json](reports/6784a0d69b0d4daa9a37ac37b5d961ea/classification_report.json) | Per-class precision, recall, F1, and support, plus macro and weighted averages. |
| [cv_scores.csv](reports/6784a0d69b0d4daa9a37ac37b5d961ea/cv_scores.csv) | Accuracy, precision, recall, and F1 for each of the five folds. |
| [confusion_matrix.png](reports/6784a0d69b0d4daa9a37ac37b5d961ea/confusion_matrix.png) | Confusion matrix for the held-out test set. |
| [metadata.json](reports/6784a0d69b0d4daa9a37ac37b5d961ea/metadata.json) | Run ID, dataset fingerprint, split configuration, model parameters, source and artifact checksums, and environment versions. |

## Interpretation and provenance

- The run reused a processed CSV. Preprocessing statistics were computed before
  splitting, so potential data leakage remains part of the current evaluation.
- CV includes the holdout rows; the CV and holdout results are not independent
  evaluations.
- The run recorded uncommitted code changes (`git.dirty=true`). Metadata contains
  the base commit and exact source checksums; source snapshots are also stored in
  the corresponding MLflow run.

The report files are preserved from the completed trial run without changing
their contents. Future training runs can overwrite the local model binaries;
use the artifact checksums to determine which run they belong to.
