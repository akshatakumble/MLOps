import json
import os
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


if __name__ == "__main__":
    n_estimators = int(os.getenv("N_ESTIMATORS", "100"))
    test_size = float(os.getenv("TEST_SIZE", "0.2"))

    # Load the Wine dataset (13 features, 3 classes)
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model to a file
    model_path = Path("wine_model.pkl")
    joblib.dump(model, model_path)

    # Save metrics to a JSON file
    metrics = {
        "dataset": "wine",
        "model": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "test_size": test_size,
        "test_accuracy": float(accuracy),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    metrics_path = Path("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training and evaluation completed successfully.")
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Test accuracy: {accuracy:.4f}")
