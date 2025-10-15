from fastapi import FastAPI
from pydantic import BaseModel
import torch
import json
from datetime import datetime
from model import TumorClassifier
from logger import setup_logger

# Initialize app and logger
app = FastAPI(title="Brain Tumor Classifier API")
logger = setup_logger()
logger.info("✅ API Service Started")

# Load latest model version from registry
try:
    with open("registry.json", "r") as f:
        registry = json.load(f)
        latest_model = registry[-1]
        model_version = latest_model["version"]
        model_path = latest_model["model_path"]
except Exception:
    model_version = "Unknown"
    model_path = "tumor_model.pth"

logger.info(f"Loaded model version: {model_version}")

# Load model
model = TumorClassifier(input_dim=5)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define request schema
class PatientData(BaseModel):
    age: float
    tumor_size: float
    density: float
    symptom_score: float
    risk_factor: float

@app.get("/")
def home():
    return {"message": "Welcome to Brain Tumor Classifier API", "model_version": model_version}

@app.post("/predict")
def predict(data: PatientData):
    input_tensor = torch.tensor([[
        data.age, data.tumor_size, data.density, data.symptom_score, data.risk_factor
    ]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_tensor).item()
    
    diagnosis = "Tumor Likely" if pred >= 0.5 else "No Tumor"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log the request and result
    logger.info(
        f"{timestamp} | Model v{model_version} | Input: {data.dict()} | Output: {diagnosis} ({pred:.4f})"
    )

    return {
        "timestamp": timestamp,
        "model_version": model_version,
        "prediction_probability": round(pred, 4),
        "diagnosis": diagnosis
    }
