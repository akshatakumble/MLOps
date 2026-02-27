# Docker Lab 1: Wine Classification Training

This lab packages a scikit-learn training script in Docker, runs training inside a container, and writes model + metrics artifacts.

## What this lab does

- Uses the `load_wine()` dataset from scikit-learn.
- Splits data into train/test sets.
- Trains `RandomForestClassifier`.
- Evaluates test accuracy.
- Saves:
  - `wine_model.pkl` (trained model)
  - `metrics.json` (run metadata + evaluation metrics)

## Project files

- `dockerfile`
  - Uses `python:3.10-slim`.
  - Uses Docker layer caching by copying `src/requirements.txt` before source code.
  - Installs dependencies with `--no-cache-dir`.
  - Runs as a non-root user (`appuser`).
- `.dockerignore`
  - Excludes git files, Python cache, notebooks, artifacts, and IDE/OS junk from build context.
- `docker-compose.yml`
  - Standardized run config (build + env vars + volume mount + command).
- `src/main.py`
  - Reads env vars:
    - `N_ESTIMATORS` (default: `100`)
    - `TEST_SIZE` (default: `0.2`)
  - Trains model and writes artifacts.

## Prerequisites

- Docker Desktop installed and running.
- Terminal opened in:
  - `C:\Users\aksha\MLOps\Labs\Docker_Labs\Lab1`

## Run Option A: Docker CLI

### 1) Build image

```powershell
docker build -t lab1:v1 -f dockerfile .
```

> Note: file name is `dockerfile` (lowercase), so `-f dockerfile` is required.

### 2) Run container (ephemeral)

```powershell
docker run --rm lab1:v1
```

This prints logs, but artifacts remain inside the container and are removed at exit.

### 3) Run and persist artifacts on host

```powershell
docker run --rm -v ${PWD}:/app lab1:v1
```

Now artifacts are written to this folder on your machine:

- `wine_model.pkl`
- `metrics.json`

### 4) Run with custom hyperparameters

```powershell
docker run --rm -v ${PWD}:/app -e N_ESTIMATORS=200 -e TEST_SIZE=0.3 lab1:v1
```

## Run Option B: Docker Compose (standardized)

### 1) Build and run

```powershell
docker compose up --build
```

Current compose config mounts `./src` to `/app`, so artifacts are written to:

- `src/wine_model.pkl`
- `src/metrics.json`

### 2) Change run parameters

Edit `docker-compose.yml` under `environment`:

- `N_ESTIMATORS: 100`
- `TEST_SIZE: 0.2`

Then rerun:

```powershell
docker compose up --build
```

## Expected output

Container logs should include:

- `Training and evaluation completed successfully.`
- `Model saved to wine_model.pkl`
- `Metrics saved to metrics.json`
- `Test accuracy: ...`

`metrics.json` contains:

- dataset name
- model name
- `n_estimators`
- `test_size`
- `test_accuracy`
- train/test sample counts

## Export image (optional)

```powershell
docker save lab1:v1 > my_image.tar
```

## Troubleshooting

- **`failed to read dockerfile: open Dockerfile: no such file or directory`**
  - Use: `docker build -t lab1:v1 -f dockerfile .`
- **Artifacts not visible after run**
  - You likely ran without volume mount.
  - Use: `docker run --rm -v ${PWD}:/app lab1:v1`
- **Artifacts appear in `src/` when using compose**
  - This is expected because compose mounts `./src:/app`.

