from fastapi import FastAPI
from pydantic import BaseModel
import os
from model_script import predict_single

# load once at startup
ARTIFACTS_DIR = "./artifacts"  # change if needed

app = FastAPI()

# Define request schema
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

@app.post("/predict")
async def predict(features: Features):
    # Convert to dict for predict_single
    feature_dict = features.dict()
    label, probs = predict_single(ARTIFACTS_DIR, **feature_dict)
    return {
        "prediction": label,
        "probabilities": probs
    }
