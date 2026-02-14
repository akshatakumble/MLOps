# Lab 1 data

Place your CSVs here or use the sample data generator.

## Required format

- **file.csv** (training): Must include columns used for clustering. Default column names: `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`. Extra columns are ignored.
- **test.csv**: Same feature columns (e.g. `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`); one or more rows for prediction.

If your CSV uses different column names, set the environment variable before running the DAG:

```bash
export LAB1_FEATURE_COLUMNS="your_col1,your_col2,your_col3"
```

## Generate sample data

If you don’t have the original dataset, from the `Labs/Airflow_Labs` directory run:

```bash
python dags/scripts/generate_sample_data.py
```

Options: `--out-dir dags/data` (default), `--train-rows 500`, `--seed 42`.
