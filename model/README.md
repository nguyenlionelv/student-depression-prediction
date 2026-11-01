# Model artifacts

The training pipeline saves three files in this directory:

```text
model/
├── README.md
├── model.pkl
├── explainer.pkl
└── feature_names.pkl
```

| File | Contents |
| --- | --- |
| `model.pkl` | Trained Stacking classifier used to generate predictions. |
| `explainer.pkl` | SHAP explainer used to calculate feature contributions for predictions. |
| `feature_names.pkl` | Ordered list of feature names expected by the model. |

All three artifacts are serialized with Joblib.