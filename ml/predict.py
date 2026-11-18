import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_model.pkl')

model = joblib.load(MODEL_PATH)


def prediksi_tanaman(n, p, k, suhu, kelembapan, ph, curah_hujan):

    data = pd.DataFrame([{
        'N': n,
        'P': p,
        'K': k,
        'temperature': suhu,
        'humidity': kelembapan,
        'ph': ph,
        'rainfall': curah_hujan
    }])

    hasil = model.predict(data)[0]

    return hasil


if __name__ == '__main__':
    hasil = prediksi_tanaman(
        n=90,
        p=42,
        k=43,
        suhu=20.8,
        kelembapan=82,
        ph=6.5,
        curah_hujan=202
    )

    print('=== HASIL PREDIKSI AGRISMART ===')
    print('Tanaman yang direkomendasikan:', hasil)
