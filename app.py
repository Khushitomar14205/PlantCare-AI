"""
Plant Disease Detection Web Application
MobileNetV2 + Correct Class Mapping (PlantVillage Dataset)
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================

app = Flask(__name__)
app.secret_key = 'plant_disease_detection_secret_key'

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_INDEX_PATH = 'models/class_indices.json'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif','bmp','webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================

model = None
IDX_TO_CLASS = None

def load_model():
    global model, IDX_TO_CLASS
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
        
    if not os.path.exists(CLASS_INDEX_PATH):
        raise RuntimeError("class_indices.json missing! Save it from training notebook.")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # Load class mapping
    with open(CLASS_INDEX_PATH) as f:
        class_indices = json.load(f)

    IDX_TO_CLASS = {v:k for k,v in class_indices.items()}
    print(f"Loaded {len(IDX_TO_CLASS)} classes")

# ================= UTILITIES =================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224,224))

    img_array = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image_path):
    processed = preprocess_image(image_path)

    preds = model.predict(processed, verbose=0)[0]

    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    disease_name = IDX_TO_CLASS.get(idx, "Unknown")

    all_predictions = {
        IDX_TO_CLASS[i]: float(preds[i])*100
        for i in range(len(preds))
    }

    return disease_name, confidence, all_predictions

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded','error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file','error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return redirect(url_for('result', image=filename))

    return render_template('upload.html')

@app.route('/result')
def result():
    image_name = request.args.get('image')
    if not image_name:
        return redirect(url_for('upload'))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    disease, confidence, all_predictions = predict_disease(image_path)

    return render_template(
        'result.html',
        image=image_name,
        disease=disease,
        confidence=confidence,
        all_predictions=all_predictions
    )

# ================= MAIN =================

if __name__ == '__main__':
    print("\nStarting Plant Disease Detection Server...")
    load_model()
    app.run(debug=True, host='127.0.0.1', port=5000)