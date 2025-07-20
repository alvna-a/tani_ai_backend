from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Mengizinkan akses dari domain lain (misalnya Netlify)

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
    'rice': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/rice.jpg',
    'maize': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/maize.jpg',
    'chickpea': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/chickpea.jpg',
    'banana': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/banana.jpg',
    'mango': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/mango.jpg',
    'apple': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/apple.jpg',
    'grapes': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/grapes.jpg',
    'watermelon': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/watermelon.jpg',
    'muskmelon': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/muskmelon.jpg',
    'orange': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/orange.jpg',
    'papaya': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/papaya.jpg',
    'coconut': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/coconut.jpg',
    'cotton': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/cotton.jpg',
    'jute': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/jute.jpg',
    'coffee': 'https://raw.githubusercontent.com/alvna-a/tani_ai2/main/gambar/coffee.jpg'
}

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        return jsonify({
            "rekomendasi": kamus.get(prediction, prediction),
            "gambar": gambar_url.get(prediction, None)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Menjalankan aplikasi
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
