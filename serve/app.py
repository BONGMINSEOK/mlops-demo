import os

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "iris-classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")

app = FastAPI()


class Features(BaseModel):
    data: list[list[float]]


@app.post("/predict")
def predict(features: Features):
    preds = model.predict(features.data)
    return {"predictions": preds.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}
