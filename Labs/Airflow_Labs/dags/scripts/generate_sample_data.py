"""
Generate sample CSV files for Airflow Lab 1 when you don't have the original dataset.
Creates file.csv (training) and test.csv with columns: BALANCE, PURCHASES, CREDIT_LIMIT.

Run from repo root (Labs/Airflow_Labs) or from dags/:
  python dags/scripts/generate_sample_data.py
  python scripts/generate_sample_data.py --out-dir dags/data --train-rows 1000
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate Lab 1 sample data (file.csv, test.csv)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: dags/data)")
    parser.add_argument("--train-rows", type=int, default=500, help="Number of rows in file.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("Need pandas and numpy: pip install pandas numpy", file=sys.stderr)
        sys.exit(1)

    # Resolve output dir: if not set, use dags/data relative to this script
    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # scripts/ is inside dags/, so dags/data is sibling of scripts
        args.out_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    columns = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

    # Training data: realistic-ish ranges (balance 0–5k, purchases 0–10k, limit 1k–20k)
    n = args.train_rows
    train = pd.DataFrame({
        "BALANCE": rng.uniform(0, 5000, n).round(2),
        "PURCHASES": rng.uniform(0, 10000, n).round(2),
        "CREDIT_LIMIT": rng.integers(1000, 20000, n),
    })
    train_path = os.path.join(args.out_dir, "file.csv")
    train.to_csv(train_path, index=False)
    print(f"Wrote {train_path} ({n} rows)")

    # Test data: a few rows with same columns
    test = pd.DataFrame({
        "BALANCE": [3202.47, 100.0, 2000.0],
        "PURCHASES": [124.5, 500.0, 3000.0],
        "CREDIT_LIMIT": [2415, 5000, 10000],
    })
    test_path = os.path.join(args.out_dir, "test.csv")
    test.to_csv(test_path, index=False)
    print(f"Wrote {test_path} ({len(test)} rows)")

    print("Done. Run your DAG with default LAB1_* settings (data dir = dags/data).")


if __name__ == "__main__":
    main()
