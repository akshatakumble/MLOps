"""
W&B experiment tracking — multiclass digits: XGBoost vs LightGBM.

Dataset: sklearn.load_digits() (~1,797 samples, 64 features, 10 digit classes).
Larger than Wine (~178 rows) for more stable train/test metrics.

Requires: pip install wandb xgboost lightgbm scikit-learn numpy
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import wandb
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_digits_split(seed: int, test_size: float) -> tuple[np.ndarray, ...]:
    """Load Optdigits-style data; return train_x, test_x, train_y, test_y (labels 0..9)."""
    data = load_digits()
    x, y = data.data.astype(np.float32), data.target.astype(np.int32)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def run_xgboost(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    num_class: int,
    num_round: int,
    seed: int,
    use_wandb_callback: bool,
) -> tuple[np.ndarray, float]:
    params: dict[str, Any] = {
        "objective": "multi:softmax",
        "eta": 0.08,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "verbosity": 0,
        "nthread": 4,
        "num_class": num_class,
        "seed": seed,
    }
    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_test = xgb.DMatrix(test_x, label=test_y)
    evals = [(xg_train, "train"), (xg_test, "test")]
    callbacks = [wandb.xgboost.WandbCallback()] if use_wandb_callback else None
    bst = xgb.train(
        params,
        xg_train,
        num_round,
        evals=evals,
        callbacks=callbacks,
    )
    pred = bst.predict(xg_test)
    acc = float(accuracy_score(test_y, pred))
    return pred, acc


def run_lightgbm(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    num_class: int,
    num_round: int,
    seed: int,
    use_wandb_callback: bool,
) -> tuple[np.ndarray, float]:
    params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "learning_rate": 0.08,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    n_features = train_x.shape[1]
    feat_names = [f"f{i}" for i in range(n_features)]
    train_data = lgb.Dataset(train_x, label=train_y, feature_name=feat_names)
    valid_data = lgb.Dataset(test_x, label=test_y, reference=train_data)
    callbacks: list[Any] = []
    if use_wandb_callback:
        from wandb.integration.lightgbm import log_summary, wandb_callback

        callbacks.append(wandb_callback())
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=num_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "test"],
        callbacks=callbacks,
    )
    if use_wandb_callback:
        try:
            log_summary(bst, feature_importance=True, save_model_checkpoint=False)
        except Exception as e:
            wandb.termwarn(f"LightGBM log_summary skipped: {e}")
    n_iter = bst.best_iteration if bst.best_iteration is not None else num_round
    proba = bst.predict(test_x, num_iteration=n_iter)
    if proba.ndim == 1:
        pred = proba.astype(np.int32)
    else:
        pred = np.argmax(proba, axis=1)
    acc = float(accuracy_score(test_y, pred))
    return pred, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digits dataset: XGBoost vs LightGBM + Weights & Biases"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-round",
        type=int,
        default=25,
        help="Boosting rounds (slightly higher than Wine for 10-class task)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train both models; print metrics only (no wandb.init)",
    )
    parser.add_argument(
        "--project",
        default="digits-xgb-vs-lgb",
        help="W&B project name",
    )
    parser.add_argument(
        "--models",
        choices=["both", "xgboost", "lgb"],
        default="both",
        help="Which model(s) to train and log",
    )
    args = parser.parse_args()

    train_x, test_x, train_y, test_y = load_digits_split(args.seed, args.test_size)
    num_class = int(np.max(train_y)) + 1
    num_round = args.num_round
    class_labels = [float(c) for c in range(num_class)]

    xgb_config_log = {
        "model": "xgboost",
        "eta": 0.08,
        "max_depth": 6,
        "num_round": num_round,
        "num_class": num_class,
        "test_size": args.test_size,
        "seed": args.seed,
        "dataset": "sklearn_digits",
    }
    lgb_config_log = {
        "model": "lightgbm",
        "learning_rate": 0.08,
        "num_leaves": 48,
        "max_depth": 6,
        "num_round": num_round,
        "num_class": num_class,
        "test_size": args.test_size,
        "seed": args.seed,
        "dataset": "sklearn_digits",
    }

    if args.dry_run:
        print("Dry run: no W&B logging.\n")
        if args.models in ("both", "xgboost"):
            _, acc_xgb = run_xgboost(
                train_x,
                test_x,
                train_y,
                test_y,
                num_class,
                num_round,
                args.seed,
                use_wandb_callback=False,
            )
            print(f"XGBoost — test accuracy: {acc_xgb:.4f}")
        if args.models in ("both", "lgb"):
            _, acc_lgb = run_lightgbm(
                train_x,
                test_x,
                train_y,
                test_y,
                num_class,
                num_round,
                args.seed,
                use_wandb_callback=False,
            )
            print(f"LightGBM — test accuracy: {acc_lgb:.4f}")
        if args.models == "both":
            print("\nSame stratified train/test split for both models.")
        sys.exit(0)

    if args.models in ("both", "xgboost"):
        run_xgb = wandb.init(
            project=args.project,
            name="xgboost-digits",
            config=xgb_config_log,
        )
        pred_xgb, acc_xgb = run_xgboost(
            train_x,
            test_x,
            train_y,
            test_y,
            num_class,
            num_round,
            args.seed,
            use_wandb_callback=True,
        )
        run_xgb.summary["test_accuracy"] = acc_xgb
        wandb.sklearn.plot_confusion_matrix(test_y, pred_xgb, class_labels)
        wandb.finish()

    if args.models in ("both", "lgb"):
        run_lgb = wandb.init(
            project=args.project,
            name="lightgbm-digits",
            config=lgb_config_log,
        )
        pred_lgb, acc_lgb = run_lightgbm(
            train_x,
            test_x,
            train_y,
            test_y,
            num_class,
            num_round,
            args.seed,
            use_wandb_callback=True,
        )
        run_lgb.summary["test_accuracy"] = acc_lgb
        wandb.sklearn.plot_confusion_matrix(test_y, pred_lgb, class_labels)
        wandb.finish()

    print("Logged to W&B project:", args.project)
    print("Compare runs: xgboost-digits vs lightgbm-digits.")


if __name__ == "__main__":
    main()
