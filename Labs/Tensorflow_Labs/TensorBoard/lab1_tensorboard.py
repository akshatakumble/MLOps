"""
TensorBoard regression lab — standalone script.

Goal: log scalars and histograms to TensorBoard with an intentionally non-minimal setup:

  • Synthetic *bivariate* linear target (2 inputs), not 1D y = 0.5x + 2.
  • Custom `tf.Module` + `tf.GradientTape` training (no `Sequential` / `model.fit`).
  • Logging via `tf.summary` writers (not `keras.callbacks.TensorBoard`).
  • Log layout: `logs/script_runs/<timestamp>/tb/`
  • Optional TF Debugger V2: `--enable-debugger`

TensorBoard extras (vs. a minimal demo): per-batch loss, MAE, R², grad norms,
activation & bias histograms, generalization gap, run config text.

Run:
  python lab1_tensorboard.py

TensorBoard:
  python -m tensorboard.main --logdir logs
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from packaging import version


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TensorBoard demo with custom training loop.")
    p.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    p.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs"),
        help="Root directory for TensorBoard logs (default: ./logs).",
    )
    p.add_argument(
        "--enable-debugger",
        action="store_true",
        help="Enable TensorFlow Debugger V2 dumps under log-root (adds overhead).",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Remove log-root before training.",
    )
    p.add_argument(
        "--log-batch-scalars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-batch training loss (smoother curves in Time Series). Default: true.",
    )
    return p.parse_args()


def assert_tf2() -> None:
    print("TensorFlow version:", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, "This script requires TensorFlow 2.x."


def make_synthetic_data(
    n: int = 1200,
    train_fraction: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bivariate linear regression: y = 1.2*x1 - 0.7*x2 + 0.3 + noise.
    Features are uniform in [0, 1]^2 (not the 1D line from Lab1.ipynb).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    noise = rng.normal(0.0, 0.04, size=(n,)).astype(np.float32)
    y = (1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.3 + noise).astype(np.float32)

    n_train = int(n * train_fraction)
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


class BivariateRegressor(tf.Module):
    """Two hidden-layer ReLU net: 2 -> 32 -> 16 -> 1 (differs from notebook's 1 -> 16 -> 1)."""

    def __init__(self) -> None:
        super().__init__()
        s = 0.1
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 32], stddev=s))
        self.b1 = tf.Variable(tf.zeros([32]))
        self.w2 = tf.Variable(tf.random.truncated_normal([32, 16], stddev=s))
        self.b2 = tf.Variable(tf.zeros([16]))
        self.w3 = tf.Variable(tf.random.truncated_normal([16, 1], stddev=s))
        self.b3 = tf.Variable(tf.zeros([1]))

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        h = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        h = tf.nn.relu(tf.matmul(h, self.w2) + self.b2)
        return tf.squeeze(tf.matmul(h, self.w3) + self.b3, axis=-1)

    def forward_hidden(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return pre-activations after ReLU (layer outputs) for TensorBoard histograms."""
        h1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        y = tf.squeeze(tf.matmul(h2, self.w3) + self.b3, axis=-1)
        return h1, h2, y


def mse_loss(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(pred - target))


def mae_metric(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(pred - target))


@tf.function
def train_step(
    model: BivariateRegressor,
    xb: tf.Tensor,
    yb: tf.Tensor,
    opt: tf.keras.optimizers.Optimizer,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        pred = model(xb)
        loss = mse_loss(pred, yb)
    grads = tape.gradient(loss, model.trainable_variables)
    tensors = [g for g in grads if g is not None]
    grad_norm = tf.linalg.global_norm(tensors)
    opt.apply_gradients((g, v) for g, v in zip(grads, model.trainable_variables) if g is not None)
    return loss, grad_norm


def predict_all(model: BivariateRegressor, x: np.ndarray, batch_size: int) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = len(x)
    for start in range(0, n, batch_size):
        xb = tf.constant(x[start : start + batch_size])
        preds.append(model(xb).numpy())
    return np.concatenate(preds, axis=0)


def regression_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE, MAE, and R² on numpy arrays (interpretable validation quality)."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "r2": r2}


def evaluate_epoch_loss(model: BivariateRegressor, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    n = len(x)
    losses = []
    for start in range(0, n, batch_size):
        xb = tf.constant(x[start : start + batch_size])
        yb = tf.constant(y[start : start + batch_size])
        pred = model(xb)
        losses.append(float(mse_loss(pred, yb)))
    return float(np.mean(losses))


def log_param_histograms(writer: tf.summary.SummaryWriter, model: BivariateRegressor, step: int) -> None:
    with writer.as_default():
        for name, var in (
            ("w1", model.w1),
            ("w2", model.w2),
            ("w3", model.w3),
            ("b1", model.b1),
            ("b2", model.b2),
            ("b3", model.b3),
        ):
            kind = "weights" if name.startswith("w") else "biases"
            tf.summary.histogram(f"{kind}/{name}", var, step=step)


def log_activation_histograms(
    writer: tf.summary.SummaryWriter,
    model: BivariateRegressor,
    sample_x: tf.Tensor,
    step: int,
) -> None:
    """ReLU activations: useful to see dead neurons (all zeros) or saturation."""
    h1, h2, _ = model.forward_hidden(sample_x)
    with writer.as_default():
        tf.summary.histogram("activations/h1_after_relu", h1, step=step)
        tf.summary.histogram("activations/h2_after_relu", h2, step=step)
        tf.summary.scalar("activations/h1_frac_zero", tf.reduce_mean(tf.cast(h1 == 0, tf.float32)), step=step)
        tf.summary.scalar("activations/h2_frac_zero", tf.reduce_mean(tf.cast(h2 == 0, tf.float32)), step=step)


def write_run_config(writer: tf.summary.SummaryWriter, config: dict) -> None:
    text = "```\n" + json.dumps(config, indent=2) + "\n```"
    with writer.as_default():
        tf.summary.text("run_config", text, step=0)


def main() -> None:
    args = parse_args()
    assert_tf2()

    if args.fresh and args.log_root.exists():
        shutil.rmtree(args.log_root)

    if args.enable_debugger:
        tf.debugging.experimental.enable_dump_debug_info(
            str(args.log_root),
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1,
        )

    args.log_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logdir = args.log_root / "script_runs" / run_id / "tb"
    tb_logdir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = make_synthetic_data()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)
        .batch(args.batch_size)
        .repeat()
    )
    train_it = iter(train_ds)

    steps_per_epoch = max(math.ceil(len(x_train) / args.batch_size), 1)
    decay_steps = steps_per_epoch * max(args.epochs // 10, 1)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-3,
        decay_steps=decay_steps,
        decay_rate=0.9,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = BivariateRegressor()
    writer = tf.summary.create_file_writer(str(tb_logdir))

    # Fixed minibatch for activation histograms (same inputs each epoch for comparable shapes).
    n_vis = min(256, len(x_train))
    vis_x = tf.constant(x_train[:n_vis])

    write_run_config(
        writer,
        {
            "run_id": run_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps_per_epoch": steps_per_epoch,
            "lr_schedule": "ExponentialDecay",
            "lr_initial": 3e-3,
            "lr_decay_steps": decay_steps,
            "lr_decay_rate": 0.9,
            "data": "synthetic bivariate linear",
            "log_batch_scalars": args.log_batch_scalars,
        },
    )

    print("Training with GradientTape + tf.summary (custom loop; not Keras fit).")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        for _ in range(steps_per_epoch):
            xb, yb = next(train_it)
            loss_t, grad_norm_t = train_step(model, xb, yb, opt)
            loss_f = float(loss_t)
            gnorm_f = float(grad_norm_t)
            epoch_losses.append(loss_f)
            epoch_grad_norms.append(gnorm_f)
            global_step += 1

            if args.log_batch_scalars:
                with writer.as_default():
                    tf.summary.scalar("loss/train_batch", loss_f, step=global_step)
                    tf.summary.scalar("train/grad_norm", gnorm_f, step=global_step)

        train_loss = float(np.mean(epoch_losses))
        val_loss_epoch = evaluate_epoch_loss(model, x_test, y_test, args.batch_size)

        y_val_pred = predict_all(model, x_test, args.batch_size)
        vm = regression_metrics_np(y_test, y_val_pred)
        y_train_pred = predict_all(model, x_train, args.batch_size)
        tm = regression_metrics_np(y_train, y_train_pred)

        lr_step = (epoch - 1) * steps_per_epoch
        gap = val_loss_epoch - train_loss

        with writer.as_default():
            # Epoch-level scalars (x = epoch): easy to compare train vs val.
            tf.summary.scalar("loss/train_epoch_mse", train_loss, step=epoch)
            tf.summary.scalar("loss/val_epoch_mse", val_loss_epoch, step=epoch)
            tf.summary.scalar("metrics/train_mae", tm["mae"], step=epoch)
            tf.summary.scalar("metrics/val_mae", vm["mae"], step=epoch)
            tf.summary.scalar("metrics/val_r2", vm["r2"], step=epoch)
            tf.summary.scalar("metrics/generalization_gap_mse", gap, step=epoch)
            tf.summary.scalar("learning_rate", lr_schedule(lr_step), step=epoch)
            tf.summary.scalar("train/grad_norm_epoch_mean", float(np.mean(epoch_grad_norms)), step=epoch)

            log_param_histograms(writer, model, epoch)
            log_activation_histograms(writer, model, vis_x, epoch)

        writer.flush()

        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train_mse={train_loss:.6f}  val_mse={val_loss_epoch:.6f}  "
                f"val_mae={vm['mae']:.6f}  val_R2={vm['r2']:.4f}"
            )

    print(f"Final train mse (last epoch mean batch): {train_loss:.6f}  |  val mse: {val_loss_epoch:.6f}")
    print(f"TensorBoard logs: {tb_logdir}")
    print("Start TensorBoard from this folder:  python -m tensorboard.main --logdir logs")
    print("Tip: filter tags by 'metrics/', 'loss/', 'activations/', 'train/' in the Time Series dashboard.")


if __name__ == "__main__":
    main()
