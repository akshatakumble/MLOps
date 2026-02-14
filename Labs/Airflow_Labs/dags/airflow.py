# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Optional: match the Lab README / tutorial (e.g. blog or video)
# - Set DAG_ID to "your_python_dag" and SCHEDULE to None for manual-only runs.
# - Data paths/columns are configurable via env: LAB1_DATA_DIR, LAB1_TRAIN_FILE,
#   LAB1_TEST_FILE, LAB1_FEATURE_COLUMNS (see src/lab.py).
# In Airflow 3.x, XCom pickling: export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

DAG_ID = "Airflow_Lab1"   # or "your_python_dag" to match README
SCHEDULE = None           # None = manual trigger; use e.g. "@daily" for scheduled

default_args = {
    "owner": "Akshata",
    "start_date": datetime(2025, 1, 15),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=15),
}

with DAG(
    DAG_ID,
    default_args=default_args,
    description="DAG for Lab 1: load data -> preprocess -> build/save GMM -> load model & BIC selection",
    schedule=SCHEDULE,
    catchup=False,
    tags=["lab1", "ml", "gmm"],
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
        doc_md="Loads CSV from data dir, serializes and returns base64 payload for XCom.",
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
        doc_md="Deserializes data, scales features (BALANCE, PURCHASES, CREDIT_LIMIT), returns serialized array.",
    )

    build_save_model_task = PythonOperator(
        task_id="build_save_model_task",
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
        doc_md="Fits GMM for k=1..25, selects best by BIC, saves to `model/model.sav`, returns BIC list.",
    )

    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
        doc_md="Loads GMM, prints optimal k (by BIC), predicts on test.csv and returns first cluster id.",
    )

    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
