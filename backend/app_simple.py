"""
Brain Tumor Classification - Simple Flask Backend
Lightweight API for brain tumor prediction
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Redirect stdout to suppress verbose Keras loading messages
import logging
logging.disable(logging.WARNING)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

print("Loading TensorFlow...")
import tensorflow as tf
from tensorflow import keras
print(f"✓ TensorFlow {tf.__version__} loaded")

# Configuration
MODEL_DIR = Path("a:/BRAIN_ML/models")
UPLOAD_DIR = Path("a:/BRAIN_ML/backend/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Initialize Flask
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load model
print("Loading model...")

# Custom object scope to handle batch_shape compatibility
def custom_input_layer(*args, **kwargs):
    """Custom InputLayer that handles batch_shape -> input_shape conversion"""
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and len(batch_shape) > 1:
            kwargs['shape'] = batch_shape[1:]
    return keras.layers.InputLayer(*args, **kwargs)

# Try loading .h5 model with custom objects
model_path = MODEL_DIR / "brain_tumor_model.h5"
if model_path.exists():
    try:
        with keras.utils.custom_object_scope({'InputLayer': custom_input_layer}):
            model = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        model = None
else:
    print("⚠️ Model file not found!")
    model = None

def preprocess_image(image_bytes):
    """Preprocess uploaded image"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict brain tumor type"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read and preprocess
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)
        
        # Predict
        predictions = model.predict(processed, verbose=0)
        probs = predictions[0]
        
        # Get top prediction
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        # All probabilities
        all_probs = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }
        
        return jsonify({
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': all_probs,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models"""
    models = []
    for f in MODEL_DIR.glob("*.h5"):
        models.append({
            'name': f.name,
            'size_mb': round(f.stat().st_size / 1024 / 1024, 2)
        })
    for f in MODEL_DIR.glob("*.keras"):
        models.append({
            'name': f.name,
            'size_mb': round(f.stat().st_size / 1024 / 1024, 2)
        })
    return jsonify({'models': models})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 Brain Tumor Classification API")
    print("="*50)
    print(f"Backend URL: http://localhost:5000")
    print(f"Model: {model_path.name if model else 'None'}")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)