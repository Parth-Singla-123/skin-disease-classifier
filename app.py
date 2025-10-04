# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
import os

# Load model
MODEL_PATH = 'disease_model.h5'
if not os.path.exists(MODEL_PATH):
    raise Exception("Model file not found. Train the model first!")

model = load_model(MODEL_PATH)

# Class labels (update as per your dataset folders)
class_labels = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis',
    'DrugEruption', 'Eczema', 'Infestation_Bites', 'Lichen', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer',
    'Sun_Sunlight_Damage', 'Tinea', 'Vascular_Tumors', 'Vasculitis',
    'Vitiligo', 'Warts'
]

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = image.load_img(file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return jsonify({'disease': predicted_class, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
