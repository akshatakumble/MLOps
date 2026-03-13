# Intro to Keras Tuner


Key files:

- `Keras_Tuner.ipynb` – original notebook (read-only reference).
- `keras_tuner_lab.py` – modified end-to-end script that you can run from the command line.

---

## What this script does

The script:

- Loads and preprocesses the **Fashion-MNIST** dataset.
- Trains a **modified baseline model** (different from the notebook):
  - Two dense layers (`256` and `128` units) instead of one.
  - Higher dropout (`0.3` instead of `0.2`).
  - Batch normalization layer added.
  - Different learning rate (`0.0005` instead of `0.001`).
  - Different number of epochs (`12` instead of `10`).
- Uses **Keras Tuner** with:
  - **RandomSearch** tuner.
  - **BayesianOptimization** tuner.
  - (The original notebook used **Hyperband** only.)
- Tunes a **richer hyperparameter space**:
  - Units in the first dense layer (`64–512`, step `64`).
  - Optional second dense layer (`0–256`, step `64`; `0` means “no second dense layer”).
  - Activation functions (`relu` or `elu`).
  - Dropout rate (`0.1–0.5`).
  - Learning rate (`5e-4`, `1e-3`, `2e-3`).
- Optionally runs a **scikit-learn RandomizedSearchCV** on a logistic regression model for comparison.
- Prints a **final comparison** of test accuracies:
  - Modified baseline.
  - RandomSearch-tuned model.
  - Bayesian-optimized model.



---

## How to run

### 1. Create and activate a virtual environment (recommended)

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On PowerShell you may need:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Install TensorFlow, Keras Tuner, and scikit-learn (the last one is optional but recommended for the extra comparison):

```bash
pip install "tensorflow>=2.10" keras-tuner scikit-learn
```

> If you already have TensorFlow installed in your environment, you can skip reinstalling it.

### 3. Run the script

From the `Hyper_Parameters_Tuning` folder:

```bash
python keras_tuner.py
```

The script will:

1. Train and evaluate the **modified baseline model**.
2. Run **RandomSearch** tuning and retrain the best configuration.
3. Run **BayesianOptimization** tuning and retrain the best configuration.
4. Optionally run **scikit-learn RandomizedSearchCV** (skipped automatically if scikit-learn is not installed).
5. Print a **final accuracy summary**.

Training can take several minutes depending on your hardware.

---

## TensorBoard logs (optional)

The script writes TensorBoard logs for:

- Baseline: `./tb_logs_baseline`
- RandomSearch: `./tb_logs_random_search`
- BayesianOptimization: `./tb_logs_bayesian`

You can inspect them with:

```bash
tensorboard --logdir .
```

Then open the URL that TensorBoard prints in your browser.

---

