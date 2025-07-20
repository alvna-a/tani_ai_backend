from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Health check endpoint
@app.route('/')
def home():
    return "TaniAI Backend Aktif!"

# Load model
model = joblib.load("crop_model.pkl")

# Kamus terjemahan dan gambar
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction_raw = model.predict(df)[0]

        # Sanitize: hapus spasi dan ubah ke lowercase
        prediction = prediction_raw.strip().lower()

        rekomendasi = kamus.get(prediction, prediction)
        gambar = gambar_url.get(prediction, "https://via.placeholder.com/150?text=Gambar+Tidak+Ditemukan")

        return jsonify({
            "rekomendasi": rekomendasi,
            "gambar": gambar
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
