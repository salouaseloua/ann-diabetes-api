# 1. Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# 2. Charger le dataset du diabète Pima Indians
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

print("Dataset chargé avec succès!")
print(f"Shape: {df.shape}")
print(df.head())

# 3. Séparer les features (X) et la target (y)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# 4. Normaliser les features avec StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDonnées normalisées!")

# 5. Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Créer le modèle ANN (2 couches denses : 12 neurones + 8 neurones)
model = Sequential([
    Dense(12, activation='relu', input_shape=(8,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 7. Compiler le modèle
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\nModèle créé!")
model.summary()

# 8. Entraîner le modèle
print("\nEntraînement en cours...")
history = model.fit(X_train, y_train, 
                    epochs=150, 
                    batch_size=10, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# 9. Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy sur le test: {accuracy*100:.2f}%")

# 10. Créer le dossier model s'il n'existe pas
os.makedirs("model", exist_ok=True)

# 11. Sauvegarder le modèle
model.save("model/model.h5")
print("\n✅ Modèle sauvegardé dans model/model.h5")

# 12. Sauvegarder les statistiques du scaler
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

meta = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": feature_names
}

with open("model/scaler_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Statistiques du scaler sauvegardées dans model/scaler_meta.json")
print("\n🎉 Partie 1 terminée avec succès!")
