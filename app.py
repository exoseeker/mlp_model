from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from model_script import predict_single

# Directory where your model artifacts are stored
ARTIFACTS_DIR = "./artifacts"  # change if needed

app = FastAPI(title="ExoSeeker ML Prediction Service")

# Allow CORS from local dev + GitHub Pages
origins = [
    "http://localhost:8080",
    "https://exoseeker.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],   # allow all headers
)

# Define the expected request schema
class Features(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_model_snr: float
    koi_impact: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float
    koi_fpflag_nt: float
    koi_fpflag_ss: float
    koi_fpflag_co: float
    koi_fpflag_ec: float

@app.get("/")
def root():
    return {"message": "FastAPI ML service running!"}

@app.post("/predict")
async def predict(features: Features):
    """
    Run prediction on a single planet's features.
    """
    feature_dict = features.dict()
    try:
        label, probs = predict_single(ARTIFACTS_DIR, **feature_dict)
        return {
            "prediction": label,
            "probabilities": probs
        }
    except Exception as e:
        return {"error": str(e)}
