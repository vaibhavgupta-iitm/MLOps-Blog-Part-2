"""
FastAPI application for IRIS classification prediction.
Part 2 of MLOps Blog - Model Serving with MLflow + Docker + Kubernetes
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import numpy as np
import os
import mlflow.sklearn

app = FastAPI(
    title="IRIS Classification API - Part 2",
    description="Model serving using FastAPI + MLflow (Part 2 of MLOps Blog)",
    version="2.0.0"
)

model = None
model_info = {}

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10)
    sepal_width: float = Field(..., ge=0, le=10)
    petal_length: float = Field(..., ge=0, le=10)
    petal_width: float = Field(..., ge=0, le=10)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str

@app.on_event("startup")
async def load_model():
    global model, model_info
    try:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        model_name = os.getenv("MODEL_NAME", "iris-classifier")
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        model_info = {"source": "mlflow", "model_name": model_name}
        print("Model loaded successfully from MLflow")
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        # Fallback: simple model
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        model_info = {"source": "fallback"}
        print("Loaded fallback model")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = np.array([[
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width
    ]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    class_names = ["setosa", "versicolor", "virginica"]
    predicted_class = class_names[prediction]
    confidence = float(probabilities[prediction])
    
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "model_version": model_info.get("model_name", "default")
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_info": model_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
