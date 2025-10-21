from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import json
import tensorflow
app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.h5")
SCALER_PATH = os.path.join("model", "scaler_meta.json")

# Charger le modèle et le scaler
model = load_model(MODEL_PATH)

with open(SCALER_PATH, "r", encoding="utf-8") as f:
    scaler = json.load(f)

MEAN = np.array(scaler["mean"], dtype=np.float32)
SCALE = np.array(scaler["scale"], dtype=np.float32)
FEATURES = scaler["feature_names"]

def normalize(X):
    return (X - MEAN) / SCALE

@app.get("/ping")
def ping():
    return {"status": "API ready"}, 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        X = np.array(data["instances"], dtype=np.float32)
        X = normalize(X)
        y_prob = model.predict(X, verbose=0)
        preds = (y_prob.ravel() >= 0.5).astype(int).tolist()
        return jsonify({"predictions": preds, "probabilities": y_prob.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
