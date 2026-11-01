# Data

This directory contains the raw dataset and the processed CSV used by the
training pipeline. Dataset files are ignored by Git; this README remains versioned.

```text
data/
├── README.md
├── raw/
│   └── student_depression_dataset.csv
└── processed/
    └── student_depression_dataset_processed.csv
```

Run all commands below from the repository root with your Python environment
activated.

## 1. Download the raw dataset

Download the dataset from
[Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset).
Sign in to Kaggle if prompted, then extract the downloaded archive if needed.

Create the raw data directory on Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path data/raw
```

Or on macOS/Linux:

```bash
mkdir -p data/raw
```

Copy the dataset CSV into `data/raw/` and rename it to
`student_depression_dataset.csv`. Keep its original column names and contents.
The expected input path is:

```text
data/raw/student_depression_dataset.csv
```

## 2. Generate processed data

Install the project dependencies if you have not already done so:

```bash
python -m pip install -r requirements.txt
```

Run the preprocessing function:

```bash
python -c "from src.pipelines.preprocessing import run_pipeline; run_pipeline()"
```

It normalizes column names, removes unused columns, handles missing values,
encodes categorical features, and creates engineered features. The output
directory is created automatically, and the result is saved to:

```text
data/processed/student_depression_dataset_processed.csv
```

If the processed CSV already exists, the command loads that file instead of
processing the raw dataset again. After replacing the raw CSV or changing
preprocessing code, regenerate the processed file with:

```bash
python -c "from src.pipelines.preprocessing import run_pipeline; run_pipeline(force_reprocess=True)"
```

This overwrites the existing processed CSV. Both commands run preprocessing
without training a model.
