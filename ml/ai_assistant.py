
import os
from dotenv import load_dotenv
from google import genai

# Membaca API Key dari .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan!")

# Menghubungkan ke Gemini
client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 30000}
)


def tanya_gemini(pertanyaan):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"""
            Kamu adalah AgriSmart AI Assistant,
            asisten pertanian yang membantu pengguna.

            Jawab menggunakan bahasa Indonesia
            yang sederhana, singkat, dan jelas.

            Pertanyaan: {pertanyaan}
            """
        )

        return response.text

    except Exception as e:
        print("DETAIL ERROR:", repr(e))
        return "Maaf, AI Assistant sedang mengalami gangguan."


if __name__ == "__main__":
    print("=== PENGUJIAN GEMINI AGRISMART ===")
    print(tanya_gemini("Apa manfaat nitrogen bagi tanaman padi?"))
