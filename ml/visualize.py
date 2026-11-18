import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Membaca dataset
df = pd.read_csv(os.path.join(BASE_DIR, "data", "Crop_recommendation.csv"))

# Membaca model
model = joblib.load(os.path.join(BASE_DIR, "ml", "models", "crop_model.pkl"))

# Membagi data untuk pengujian
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Membuat folder hasil grafik
OUTPUT_DIR = os.path.join(BASE_DIR, "ml", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grafik 1: Distribusi tanaman
plt.figure(figsize=(12, 6))
df["label"].value_counts().plot(kind="bar")
plt.title("Distribusi Jenis Tanaman")
plt.xlabel("Jenis Tanaman")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribusi_tanaman.png"))
plt.close()

# Grafik 2: Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(14, 12))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot(xticks_rotation=90, cmap="Greens", values_format="d")
plt.title("Confusion Matrix Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Grafik 3: Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(9, 5))
importance.plot(kind="barh")
plt.title("Pengaruh Fitur terhadap Prediksi Tanaman")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

print("Semua grafik berhasil dibuat!")
print("Lokasi:", OUTPUT_DIR)
print("1. distribusi_tanaman.png")
print("2. confusion_matrix.png")
print("3. feature_importance.png")
