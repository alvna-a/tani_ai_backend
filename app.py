from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import logging

# Setup Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Memperbolehkan semua origin (Netlify, Vercel, dll)

# Logging aktif
logging.basicConfig(level=logging.DEBUG)

# Health check endpoint
@app.route('/')
def home():
    return "TaniAI Backend Aktif!"

# Load model (debug jika gagal)
try:
    model = joblib.load("crop_model.pkl")
    app.logger.info("Model berhasil dimuat.")
except Exception as e:
    app.logger.error(f"Gagal memuat model: {e}")
    model = None

# Kamus bahasa Indonesia dan gambar
kamus = {
    'rice': 'padi',
    'maize': 'jagung',
    'chickpea': 'kacang arab',
    'banana': 'pisang',
    'mango': 'mangga',
    'apple': 'apel',
    'grapes': 'anggur',
    'watermelon': 'semangka',
    'muskmelon': 'blewah',
    'orange': 'jeruk',
    'papaya': 'pepaya',
    'coconut': 'kelapa',
    'cotton': 'kapas',
    'jute': 'jut',
    'coffee': 'kopi'
}

gambar_url = {
    key: f"https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/{key}.jpg"
    for key in kamus
}

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model belum dimuat"}), 500

    try:
        data = request.get_json()
        app.logger.debug(f"Data diterima: {data}")

        df = pd.DataFrame([data])
        prediction_raw = model.predict(df)[0]

        prediction = prediction_raw.strip().lower()
        app.logger.debug(f"Hasil prediksi mentah: {prediction_raw}")

        rekomendasi = kamus.get(prediction, prediction)
        gambar = gambar_url.get(prediction, "https://via.placeholder.com/150?text=Gambar+Tidak+Ditemukan")

        return jsonify({
            "rekomendasi": rekomendasi,
            "gambar": gambar
        })

    except Exception as e:
        app.logger.error(f"Error saat prediksi: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Jalankan server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
