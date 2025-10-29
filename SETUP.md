# Brain Tumor Classification - ML Pipeline Setup Guide

## 📋 Project Overview

This is a complete machine learning pipeline for **brain tumor classification** from MRI images with 4 classes:
- **Glioma** - Most common malignant brain tumor
- **Meningioma** - Tumor of the brain's protective membrane
- **Pituitary** - Tumor in the pituitary gland
- **No Tumor** - Normal/healthy brain scan

## 🎯 Key Features

✅ **Comprehensive Dataset Analysis**
- Image count per class
- Image dimension analysis
- Format verification
- Distribution visualization

✅ **Multiple Model Architectures**
- Custom CNN (lightweight, fast)
- MobileNet (mobile-optimized)
- EfficientNet (high-performance, recommended)

✅ **Advanced Training Features**
- Data augmentation (rotation, zoom, flip)
- Early stopping (prevent overfitting)
- Learning rate scheduling
- Transfer learning support
- Class weight balancing

✅ **Complete Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- Training history plots
- Per-class performance metrics

✅ **Production Ready**
- Model persistence (save/load)
- Configuration management
- Inference wrapper
- Error handling

## 📁 Project Structure

```
a:\ml/
├── dataset/                          # Training/Testing data
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── src/                              # Core modules
│   ├── dataset_analysis.py           # Dataset statistics & visualization
│   └── ml_pipeline.py                # ML pipeline implementation
├── train_script/                     # Training & inference
│   ├── train_main.py                 # Main training script
│   └── predict.py                    # Inference/prediction
├── config.py                         # Configuration settings
├── requirements.txt                  # Python dependencies
├── README.md                         # Full documentation
├── SETUP.md                          # This file
├── trained_model.h5                  # Trained model (generated)
├── dataset_analysis.png              # Dataset visualization (generated)
├── training_history.png              # Training plots (generated)
└── confusion_matrix.png              # Evaluation matrix (generated)
```

## 🚀 Installation & Setup

### Step 1: Install Python Requirements

```bash
pip install -r requirements.txt
```

**Required packages:**
- tensorflow 2.13+
- keras 2.13+
- numpy 1.24+
- scikit-learn 1.3+
- matplotlib 3.7+
- seaborn 0.12+
- Pillow 10+

### Step 2: Verify Dataset Structure

Ensure your dataset is organized correctly:

```
dataset/
├── Training/
│   └── [class_name]/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── Testing/
    └── [class_name]/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

**Image formats supported:** JPG, JPEG, PNG, GIF, BMP

### Step 3: Configure Settings (Optional)

Edit `config.py` to customize:
- Model type and architecture
- Image size and augmentation
- Training parameters
- Hyperparameters
- Output paths

## 📊 Running the Pipeline

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline from dataset analysis to model evaluation:

```bash
python train_script/train_main.py
```

This will:
1. Analyze dataset statistics
2. Load and split data
3. Build the model
4. Train with validation
5. Evaluate on test set
6. Generate visualizations
7. Save the trained model

**Expected output:**
```
======================================================================
BRAIN TUMOR CLASSIFICATION - COMPLETE ML PIPELINE
======================================================================

[STEP 1] Analyzing Dataset...
  Loading glioma: 426 images...
  Loading meningioma: 374 images...
  Loading notumor: 395 images...
  Loading pituitary: 405 images...

[STEP 2] Loading Data...
📊 Data Split Summary:
  Training set:   1400 images
  Validation set: 350 images
  Test set:       1200 images

[STEP 3] Building Model...
Total parameters: 4,049,564

[STEP 4] Training Model...
Epoch 1/20
...
[STEP 5] Evaluating Model...
Test Accuracy: 0.9234
...

📈 Final Results:
  Test Accuracy: 0.9234

💾 Artifacts Generated:
  • Model: a:\ml\trained_model.h5
  • Dataset Analysis: a:\ml\dataset_analysis.png
  • Training History: a:\ml\training_history.png
  • Confusion Matrix: a:\ml\confusion_matrix.png
```

### Option 2: Dataset Analysis Only

```bash
python src/dataset_analysis.py
```

Generates distribution charts and statistics without training.

### Option 3: Use Trained Model for Predictions

```python
from train_script.predict import BrainTumorPredictor

predictor = BrainTumorPredictor("a:\\ml\\trained_model.h5")
result = predictor.predict_image("path/to/image.jpg")
print(result)
# Output:
# {
#     'predicted_class': 'glioma',
#     'confidence': 0.95,
#     'probabilities': {
#         'glioma': 0.95,
#         'meningioma': 0.03,
#         'notumor': 0.01,
#         'pituitary': 0.01
#     }
# }
```

## ⚙️ Configuration Guide

### Model Selection

**For fastest training (experiments):**
```python
MODEL_TYPE = 'cnn'
EPOCHS = 10
BATCH_SIZE = 64
```

**For best accuracy (production):**
```python
MODEL_TYPE = 'efficientnet'
EPOCHS = 50
BATCH_SIZE = 32
```

**For mobile deployment:**
```python
MODEL_TYPE = 'mobilenet'
USE_PRETRAINED = True
```

### Image Size Impact

| Size | Speed | Accuracy | Memory |
|------|-------|----------|--------|
| (128, 128) | ⚡⚡ | 🟡 | 💾 |
| (224, 224) | ⚡ | 🟢 | 💾💾 |
| (300, 300) | 🟡 | 🟢🟢 | 💾💾💾 |

### Data Augmentation

**Enable for small datasets (<5000 images):**
```python
ENABLE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'zoom_range': 0.3,
    'horizontal_flip': True,
}
```

**Disable for large datasets:**
```python
ENABLE_AUGMENTATION = False
```

## 📈 Understanding Results

### Confusion Matrix

Shows predicted vs actual labels:
```
             Predicted
         Glioma  Mening  NotTum  Pituit
Actual
Glioma     320      8       5       2
Meningioma  6     295      12       1
NotTumor    4       8     360       3
Pituitary   2       3       1     384
```

**Interpretation:**
- Diagonal values: Correct predictions ✅
- Off-diagonal values: Misclassifications ❌

### Training History

**Accuracy curve:**
- Should increase over epochs
- Training and validation should track closely
- Large gap = overfitting

**Loss curve:**
- Should decrease over epochs
- Should stabilize near end
- Increasing = learning rate too high

## 🔧 Troubleshooting

### Out of Memory (OOM) Error

```python
# Reduce batch size
BATCH_SIZE = 16

# Reduce image size
IMAGE_SIZE = (128, 128)

# Use smaller model
MODEL_TYPE = 'mobilenet'
```

### Poor Accuracy

```python
# Increase training data or use augmentation
ENABLE_AUGMENTATION = True

# Use better model
MODEL_TYPE = 'efficientnet'

# Longer training
EPOCHS = 50

# Lower learning rate
LEARNING_RATE = 0.0001
```

### Training Too Slow

```python
# Reduce image size
IMAGE_SIZE = (128, 128)

# Use faster model
MODEL_TYPE = 'mobilenet'

# Increase batch size
BATCH_SIZE = 64

# Disable augmentation (if speed critical)
ENABLE_AUGMENTATION = False
```

### Overfitting (High train, low val accuracy)

```python
# Increase dropout
DROPOUT_RATE = 0.7

# More aggressive augmentation
AUGMENTATION_CONFIG['rotation_range'] = 45
AUGMENTATION_CONFIG['zoom_range'] = 0.4

# Lower learning rate
LEARNING_RATE = 0.0001

# More regularization
# Add L1/L2 penalties in pipeline code
```

## 📚 Model Architecture Details

### CNN (Custom)
```
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool
    ↓
GlobalAvgPool → Dense(256) → Dropout(0.5) → Dense(4, softmax)
```

### MobileNet
- Lightweight depthwise separable convolutions
- Pre-trained on ImageNet
- ~4M parameters
- Great for mobile/edge devices

### EfficientNet
- State-of-the-art efficiency
- Compound scaling (depth, width, resolution)
- Pre-trained on ImageNet
- Multiple variants (B0-B7)

## 🎓 Learning Resources

- [TensorFlow Documentation](https://tensorflow.org)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Medical Image Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- [Keras Best Practices](https://keras.io)

## ⚡ Performance Tips

1. **Use GPU**: ~10x faster training
2. **Batch Norm**: Improves convergence and stability
3. **Data Augmentation**: Prevents overfitting with limited data
4. **Early Stopping**: Saves training time and prevents overfitting
5. **Learning Rate Schedule**: Improves final accuracy
6. **Transfer Learning**: Faster convergence with less data
7. **Mixed Precision**: Faster training on modern GPUs

## 📝 Next Steps

1. ✅ Setup environment and install dependencies
2. ✅ Verify dataset structure
3. ✅ Run dataset analysis
4. ✅ Train model
5. ⭕ Fine-tune hyperparameters
6. ⭕ Deploy to production
7. ⭕ Monitor predictions

## 🆘 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Review relevant section in this guide
3. Check TensorFlow/Keras documentation
4. Verify dataset format and paths
5. Try reducing model complexity

---

**Last Updated:** October 29, 2025  
**Framework:** TensorFlow 2.13+ / Keras 2.13+  
**Python Version:** 3.8+  
**License:** Educational Use
