import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import pickle
import os
import base64

# Configurable paths and columns (override via environment variables to replicate in your setup)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("LAB1_DATA_DIR", os.path.join(_BASE_DIR, "data"))
MODEL_DIR = os.environ.get("LAB1_MODEL_DIR", os.path.join(_BASE_DIR, "model"))
TRAIN_FILE = os.environ.get("LAB1_TRAIN_FILE", "file.csv")
TEST_FILE = os.environ.get("LAB1_TEST_FILE", "test.csv")
# Comma-separated column names for clustering features (e.g. "BALANCE,PURCHASES,CREDIT_LIMIT")
FEATURE_COLUMNS = [
    c.strip()
    for c in os.environ.get("LAB1_FEATURE_COLUMNS", "BALANCE,PURCHASES,CREDIT_LIMIT").split(",")
    if c.strip()
]


def _train_path():
    return os.path.join(DATA_DIR, TRAIN_FILE)


def _test_path():
    return os.path.join(DATA_DIR, TEST_FILE)


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    path = _train_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training data not found: {path}. "
            "Set LAB1_DATA_DIR / LAB1_TRAIN_FILE or run the sample data generator: "
            "python dags/scripts/generate_sample_data.py"
        )
    df = pd.read_csv(path)
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature columns not found in data: {missing}. "
            f"Available columns: {list(df.columns)}. "
            "Set LAB1_FEATURE_COLUMNS to match your CSV (e.g. BALANCE,PURCHASES,CREDIT_LIMIT)."
        )
    clustering_data = df[FEATURE_COLUMNS]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a GMM (Gaussian Mixture Model) on the preprocessed data and saves the best model.
    Uses BIC for model selection. Returns the BIC list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)
    if hasattr(X, "values"):
        X = np.asarray(X, dtype=np.float64)

    max_components = 25  # tune range if needed; GMM is slower than K-Means per k
    bic_list = []
    best_gmm = None
    best_bic = np.inf

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_list.append(float(bic))
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, filename)
    with open(output_path, "wb") as f:
        pickle.dump(best_gmm, f)

    return bic_list  # list is JSON-safe


def load_model_elbow(filename: str, bic_list: list):
    """
    Loads the saved GMM and reports optimal number of components (lowest BIC).
    Returns the first prediction (cluster id) for test.csv.
    """
    output_path = os.path.join(MODEL_DIR, filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # optimal k = number of components with lowest BIC
    optimal_k = int(np.argmin(bic_list)) + 1
    print(f"Optimal no. of components (by BIC): {optimal_k} (model has n_components={loaded_model.n_components})")

    # predict on test data
    test_path = _test_path()
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Test data not found: {test_path}. "
            "Set LAB1_DATA_DIR / LAB1_TEST_FILE or run the sample data generator."
        )
    df = pd.read_csv(test_path)
    # Use same feature columns as training (model expects same number of features)
    if any(c not in df.columns for c in FEATURE_COLUMNS):
        raise ValueError(
            f"Test CSV must contain feature columns: {FEATURE_COLUMNS}. "
            f"Available: {list(df.columns)}."
        )
    X_test = df[FEATURE_COLUMNS].values.astype(np.float64)
    pred = loaded_model.predict(X_test)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
