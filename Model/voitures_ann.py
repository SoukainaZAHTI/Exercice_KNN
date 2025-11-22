# voitures_ann_optimized.py
"""
Estimation du prix des voitures d'occasion avec un ANN (Keras) - version optimisée.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------
# Paramètres
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "fichier.csv"
CURRENT_YEAR = 2023

# ------------------------
# 1) Chargement du dataset
# ------------------------
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1252')  # encodage pour accents
    print(f"Loaded dataset '{CSV_PATH}' with {len(df)} rows.")
else:
    raise FileNotFoundError(f"{CSV_PATH} not found. Please place your CSV in the folder.")

# ------------------------
# 2) Nettoyage et préparation
# ------------------------
# Supprimer colonne inutile
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Conversion des colonnes numériques
numeric_cols = ['year', 'price_in_euro', 'power_kw', 'mileage_in_km']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer lignes avec valeurs manquantes
df = df.dropna(subset=numeric_cols)

# Feature engineering
df['age'] = CURRENT_YEAR - df['year']
TARGET = 'price_in_euro'

# Encodage des variables catégorielles
categorical_cols = ['brand','model','fuel_type']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Sélection features et target
feature_cols = [c for c in df_encoded.columns if c not in [TARGET,'year']]
X = df_encoded[feature_cols].copy()
y = df_encoded[TARGET].values

print("\nFeatures used (sample):", feature_cols[:10])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------
# 3) Construction du modèle ANN
# ------------------------
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train_scaled.shape[1])
model.summary()

# ------------------------
# 4) Entraînement
# ------------------------
EPOCHS = 100
BATCH_SIZE = 32
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

# Courbe de perte
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# ------------------------
# 5) Évaluation
# ------------------------
y_pred = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
print(f"\nTest MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R2 : {r2:.4f}")

# Scatter réel vs prédit
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Prix réel vs Prix prédit")
plt.grid(True)
plt.savefig("real_vs_pred.png", dpi=150)
plt.show()

# Histogramme des erreurs
residuals = y_test - y_pred
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30)
plt.xlabel("Erreur (réel - prédit)")
plt.title("Distribution des erreurs")
plt.grid(True)
plt.savefig("residuals_hist.png", dpi=150)
plt.show()

# ------------------------
# 6) Sauvegarde des prédictions et du modèle
# ------------------------
out_df = pd.DataFrame({
    'price_real': y_test,
    'price_pred': y_pred,
    'residual': residuals
})
out_df.to_csv("predictions_vs_reels.csv", index=False)
model.save("voitures_ann_model.h5")

# Top 10 erreurs absolues
out_df['abs_error'] = out_df['residual'].abs()
worst = out_df.sort_values('abs_error', ascending=False).head(10)
print("\nTop 10 worst predictions (sample):")
print(worst)

print("\n--- Fin du script ---")
