import os
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


def load_fashion_mnist() -> Tuple[tuple, tuple]:
    """Load and normalize the Fashion-MNIST dataset."""
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

    img_train = img_train.astype("float32") / 255.0
    img_test = img_test.astype("float32") / 255.0

    # Add channel dimension so it's easy to extend to CNNs if desired
    img_train = img_train[..., None]
    img_test = img_test[..., None]

    return (img_train, label_train), (img_test, label_test)


def print_results(model: keras.Model, model_name: str, layer_name: str, eval_dict: dict) -> None:
    """Print key hyperparameters and evaluation metrics for a model."""
    print(f"\n{model_name}:")
    dense_layer = model.get_layer(layer_name)
    print(f"number of units in '{layer_name}': {dense_layer.units}")
    print(f"activation for '{layer_name}': {dense_layer.activation.__name__}")
    print(f"learning rate for the optimizer: {float(model.optimizer.learning_rate.numpy()):.6f}")

    for key, value in eval_dict.items():
        print(f"{key}: {value}")


def build_baseline_model(input_shape=(28, 28, 1)) -> keras.Model:
    """
    Build a modified baseline model.

    Changes vs. original notebook:
    - Two dense layers instead of one
    - Higher dropout
    - Batch normalization
    - Different learning rate and number of epochs
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu", name="baseline_dense_1"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu", name="baseline_dense_2"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Changed learning rate from 0.001 (in notebook) to 0.0005
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_baseline(
    img_train, label_train, img_test, label_test, num_epochs: int = 12
):
    """Train and evaluate the modified baseline model."""
    baseline_model = build_baseline_model(input_shape=img_train.shape[1:])
    baseline_model.summary()

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir="./tb_logs_baseline", update_freq="epoch"
    )

    history = baseline_model.fit(
        img_train,
        label_train,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tensorboard_cb],
        verbose=2,
    )

    eval_dict = baseline_model.evaluate(img_test, label_test, return_dict=True)
    print_results(baseline_model, "BASELINE MODEL (MODIFIED)", "baseline_dense_1", eval_dict)

    return baseline_model, eval_dict, history


def dense_hypermodel(hp: kt.HyperParameters) -> keras.Model:
    """
    Hypermodel for fully connected network with several tunable hyperparameters.

    Tuned hyperparameters:
    - units_1: size of first dense layer
    - activation_1: activation of first dense layer
    - units_2: size of optional second dense layer (can be 0 = skip layer)
    - activation_2: activation of second dense layer (if used)
    - dropout_rate: dropout after dense layers
    - learning_rate: Adam learning rate
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    model.add(keras.layers.Flatten())

    units_1 = hp.Int("units_1", min_value=64, max_value=512, step=64)
    activation_1 = hp.Choice("activation_1", values=["relu", "elu"])
    model.add(keras.layers.Dense(units_1, activation=activation_1, name="tuned_dense_1"))

    units_2 = hp.Int("units_2", min_value=0, max_value=256, step=64)
    if units_2 > 0:
        activation_2 = hp.Choice("activation_2", values=["relu", "elu"])
        model.add(keras.layers.Dense(units_2, activation=activation_2, name="tuned_dense_2"))

    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(10, activation="softmax"))

    learning_rate = hp.Choice("learning_rate", values=[5e-4, 1e-3, 2e-3])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def run_random_search(
    img_train, label_train, img_test, label_test, max_trials: int = 10, num_epochs: int = 10
):
    """Run Keras Tuner RandomSearch on the dense hypermodel."""
    tuner_dir = "kt_random_search"
    os.makedirs(tuner_dir, exist_ok=True)

    tuner = kt.RandomSearch(
        hypermodel=dense_hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name="fashion_mnist_random",
        overwrite=True,
    )

    print("\nRandomSearch search space summary:")
    tuner.search_space_summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    tb_cb = keras.callbacks.TensorBoard(
        log_dir="./tb_logs_random_search",
        update_freq="batch",
    )

    tuner.search(
        img_train,
        label_train,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, tb_cb],
        verbose=2,
    )

    print("\nRandomSearch results summary:")
    tuner.results_summary()

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    best_model.summary()

    best_model.fit(
        img_train,
        label_train,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=2,
    )

    eval_dict = best_model.evaluate(img_test, label_test, return_dict=True)
    print_results(best_model, "RANDOMSEARCH TUNED MODEL", "tuned_dense_1", eval_dict)

    return best_model, best_hp, eval_dict


def run_bayesian_optimization(
    img_train, label_train, img_test, label_test, max_trials: int = 12, num_epochs: int = 10
):
    """Run Keras Tuner BayesianOptimization on the dense hypermodel."""
    tuner_dir = "kt_bayesian_optimization"
    os.makedirs(tuner_dir, exist_ok=True)

    tuner = kt.BayesianOptimization(
        hypermodel=dense_hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        directory=tuner_dir,
        project_name="fashion_mnist_bayesian",
        overwrite=True,
    )

    print("\nBayesianOptimization search space summary:")
    tuner.search_space_summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    tb_cb = keras.callbacks.TensorBoard(
        log_dir="./tb_logs_bayesian",
        update_freq="batch",
    )

    tuner.search(
        img_train,
        label_train,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, tb_cb],
        verbose=2,
    )

    print("\nBayesianOptimization results summary:")
    tuner.results_summary()

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    best_model.summary()

    best_model.fit(
        img_train,
        label_train,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=2,
    )

    eval_dict = best_model.evaluate(img_test, label_test, return_dict=True)
    print_results(best_model, "BAYESIAN TUNED MODEL", "tuned_dense_1", eval_dict)

    return best_model, best_hp, eval_dict


def run_sklearn_random_search(img_train, label_train):
    """
    Optional: show a simple scikit-learn RandomizedSearchCV example
    on flattened Fashion-MNIST images with logistic regression.
    """
    try:
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("\n[sklearn] scikit-learn is not installed; skipping sklearn example.")
        return

    print("\nRunning a small scikit-learn RandomizedSearchCV on flattened images...")

    # Use a subset for speed
    n_samples = 20000
    img_subset = img_train[:n_samples]
    y_subset = label_train[:n_samples]
    X_subset = img_subset.reshape(n_samples, -1)

    clf = LogisticRegression(max_iter=500)

    param_distributions = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["saga", "lbfgs"],
    }

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=4,
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_subset, y_subset)
    print(f"Best sklearn params: {search.best_params_}")
    print(f"Best sklearn CV score: {search.best_score_}")


def main():
    (img_train, label_train), (img_test, label_test) = load_fashion_mnist()

    # 1. Modified baseline model
    baseline_model, baseline_eval, _ = train_and_evaluate_baseline(
        img_train, label_train, img_test, label_test, num_epochs=12
    )

    # 2. Keras Tuner – RandomSearch
    rs_model, rs_hp, rs_eval = run_random_search(
        img_train, label_train, img_test, label_test, max_trials=10, num_epochs=10
    )

    # 3. Keras Tuner – BayesianOptimization
    bo_model, bo_hp, bo_eval = run_bayesian_optimization(
        img_train, label_train, img_test, label_test, max_trials=12, num_epochs=10
    )

    # 4. Optional scikit-learn example (non-Keras tuner)
    run_sklearn_random_search(img_train, label_train)

    print("\n=== FINAL COMPARISON (test accuracy) ===")
    print(f"Baseline accuracy:           {baseline_eval['accuracy']:.4f}")
    print(f"RandomSearch tuned accuracy: {rs_eval['accuracy']:.4f}")
    print(f"Bayesian tuned accuracy:     {bo_eval['accuracy']:.4f}")


if __name__ == "__main__":
    main()

