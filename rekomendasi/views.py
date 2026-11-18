
from django.shortcuts import render
from ml.fuzzy_logic import hitung_kesesuaian
from ml.predict import prediksi_tanaman
from ml.ai_assistant import tanya_gemini


def home(request):
    return render(request, 'rekomendasi/home.html')


def fuzzy_test(request):
    hasil = None
    error = None

    if request.method == 'POST':
        try:
            suhu = float(request.POST.get('suhu'))
            kelembapan = float(request.POST.get('kelembapan'))
            ph = float(request.POST.get('ph'))
            curah_hujan = float(request.POST.get('curah_hujan'))

            hasil = hitung_kesesuaian(
                suhu, kelembapan, ph, curah_hujan
            )

        except (ValueError, TypeError) as e:
            error = str(e)

    return render(request, 'rekomendasi/fuzzy.html', {
        'hasil': hasil,
        'error': error
    })


def machine_learning(request):
    hasil = None
    error = None

    if request.method == 'POST':
        try:
            n = float(request.POST.get('n'))
            p = float(request.POST.get('p'))
            k = float(request.POST.get('k'))
            suhu = float(request.POST.get('suhu'))
            kelembapan = float(request.POST.get('kelembapan'))
            ph = float(request.POST.get('ph'))
            curah_hujan = float(request.POST.get('curah_hujan'))

            hasil = prediksi_tanaman(
                n, p, k, suhu, kelembapan, ph, curah_hujan
            )

        except (ValueError, TypeError) as e:
            error = str(e)

    return render(request, 'rekomendasi/machine_learning.html', {
        'hasil': hasil,
        'error': error
    })


def ai_assistant(request):
    jawaban = None
    error = None
    pertanyaan = ''

    if request.method == 'POST':
        pertanyaan = request.POST.get('pertanyaan', '').strip()

        if not pertanyaan:
            error = 'Silakan masukkan pertanyaan terlebih dahulu.'
        elif len(pertanyaan) > 1000:
            error = 'Pertanyaan maksimal 1000 karakter.'
        else:
            try:
                jawaban = tanya_gemini(pertanyaan)

                if jawaban.startswith('Maaf,'):
                    error = jawaban
                    jawaban = None

            except Exception:
                error = 'Maaf, AI Assistant sedang mengalami gangguan.'

    return render(request, 'rekomendasi/ai_assistant.html', {
        'jawaban': jawaban,
        'error': error,
        'pertanyaan': pertanyaan
    })

def visualisasi(request):
    return render(request, 'rekomendasi/visualisasi.html')