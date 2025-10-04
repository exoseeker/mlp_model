# 🌍 Planet Classifier API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A FastAPI backend that serves a machine learning model to predict whether an object is likely to be a **planet**.  
It takes **15 input parameters** and returns a prediction.




## 📂 Project Structure
```
├── app.py # FastAPI app entrypoint
├── model_script.py # Model loading & prediction logic
├── artifacts/ # Trained model + scaler/encoder
├── requirements.txt # Dependencies
├── Procfile # Render start command
└── render.yaml # Optional Render config (IaC)
```




## Local Dev
pyenv local 3.11.9

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt



## Example
```
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "param1": 0.5, "param2": 1.2, "param3": 0.3,
  "param4": 0.9, "param5": 1.7, "param6": 2.1,
  "param7": 0.4, "param8": 3.2, "param9": 0.8,
  "param10": 1.9, "param11": 2.7, "param12": 1.0,
  "param13": 4.1, "param14": 0.6, "param15": 2.3
}'
```

should return `{"prediction": "Planet"}` with confience margins
