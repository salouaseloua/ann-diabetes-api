from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Charger le modèle sauvegardé
model = tf.keras.models.load_model("iris_ann_model.h5")

# Créer l'application Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ API Iris est opérationnelle"

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Récupérer les features
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Les données doivent contenir une clé 'features'"}), 400

    # Mise en forme et prédiction
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return jsonify({"predicted_class": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
