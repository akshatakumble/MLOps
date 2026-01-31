from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/classes")
async def get_class_labels():
    return {
        0: "setosa",
        1: "versicolor", 
        2: "virginica"
    }


class IrisBatchRequest(BaseModel):
    samples: list[IrisData]

@app.post("/predict/batch")
async def predict_iris_batch(batch: IrisBatchRequest):
    features = [[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width] 
                for s in batch.samples]
    predictions = predict_data(features)
    return {"predictions": [int(p) for p in predictions]}


@app.get("/model/info")
async def model_info():
    return {
        "model_type": "DecisionTreeClassifier",
        "dataset": "Iris",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "num_classes": 3
    }