
# ==========================================
# AGRISMART - TRAINING MACHINE LEARNING
# Algoritma: Random Forest
# ==========================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Menentukan lokasi dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, 'data', 'Crop_recommendation.csv'
)

# 2. Membaca dataset
df = pd.read_csv(DATASET_PATH)

print("=== DATASET AGRISMART ===")
print("Jumlah data:", df.shape[0])
print("Jumlah kolom:", df.shape[1])
print("\nData pertama:")
print(df.head())

# 3. Memisahkan fitur dan target
X = df.drop('label', axis=1)
y = df['label']

# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Membuat model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Melatih model
model.fit(X_train, y_train)

# 7. Melakukan prediksi
y_pred = model.predict(X_test)

# 8. Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)

print("\n=== HASIL PELATIHAN ===")
print("Data training:", len(X_train))
print("Data testing:", len(X_test))
print("Akurasi model:", round(accuracy * 100, 2), "%")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# 9. Menyimpan model
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')

joblib.dump(model, MODEL_PATH)

print("\nModel berhasil disimpan!")
print("Lokasi:", MODEL_PATH)
