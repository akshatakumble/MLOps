# FastAPI Lab — Iris Classifier API

This is a small REST API that wraps a trained Iris classifier. Built with [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

**What’s inside:** A Decision Tree model trained on the Iris dataset, exposed through HTTP endpoints so you can send flower measurements and get back predicted species (setosa, versicolor, or virginica).

---

## Getting Started

**1. Virtual environment**
```bash
python -m venv fastapi_lab1_env
fastapi_lab1_env\Scripts\activate
```

**2. Dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
cd src
python train.py
```

**4. Run the API**
```bash
uvicorn main:app --reload
```

Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Layout

```
FastAPI_Labs/
├── model/          # Trained model (iris_model.pkl)
├── src/
│   ├── data.py     # Load & split Iris data
│   ├── train.py    # Fit model, save to disk
│   ├── predict.py  # Load model, run inference
│   └── main.py     # FastAPI routes
├── requirements.txt
└── README.md
```

---

## Endpoints

### `GET /`

**What it does:** Simple liveness check. Returns `{"status": "healthy"}` so load balancers or health checks can confirm the service is up.

---

### `POST /predict`

**What it does:** Sends one set of Iris measurements to the model and returns the predicted class (0, 1, or 2). Use this when you need a single prediction at a time.

**Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{"response": 0}
```
(0 = setosa, 1 = versicolor, 2 = virginica)

---

### `POST /predict/batch`

**What it does:** Accepts multiple Iris samples in one request and returns a list of predictions. Handy when you want to score many flowers without making repeated calls.

**Request:**
```json
{
  "samples": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}
  ]
}
```

**Response:**
```json
{"predictions": [0, 1]}
```
Predictions follow the same order as the input samples.

---

### `GET /classes`

**What it does:** Returns the mapping from numeric labels to species names. Use this when you have a prediction like `0` and need to show the user "setosa" instead.

**Response:**
```json
{
  "0": "setosa",
  "1": "versicolor",
  "2": "virginica"
}
```

---

### `GET /model/info`

**What it does:** Returns basic metadata about the deployed model—type, dataset, feature names, and number of classes. Useful for debugging or for clients that need to discover the expected schema.

**Response:**
```json
{
  "model_type": "DecisionTreeClassifier",
  "dataset": "Iris",
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "num_classes": 3
}
```

---

## Testing

- Use **Swagger UI** at `/docs` for interactive testing.
- Or tools like [Postman](https://www.postman.com/) or `curl`.

