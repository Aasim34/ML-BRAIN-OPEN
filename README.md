# Brain Tumor Classification - ML Pipeline

A comprehensive machine learning pipeline for binary and multi-class brain tumor classification using medical imaging data.

## 📊 Dataset Structure

```
dataset/
├── Training/
│   ├── glioma/          # Brain tumor type 1
│   ├── meningioma/      # Brain tumor type 2
│   ├── notumor/         # No tumor (negative class)
│   └── pituitary/       # Brain tumor type 3
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dataset Analysis
```bash
python src/dataset_analysis.py
```

This will:
- Count images per class
- Analyze image properties (size, format)
- Generate distribution visualizations
- Create: `dataset_analysis.png`

### 3. Train the Model
```bash
python train_script/train.py
```

The pipeline will:
1. **Analyze** the dataset structure
2. **Load** all training and testing data
3. **Preprocess** images (resize, normalize)
4. **Build** the neural network (EfficientNet by default)
5. **Train** with data augmentation
6. **Evaluate** on test set
7. **Generate** visualizations
8. **Save** the trained model

## 📈 Pipeline Components

### 1. Dataset Analysis (`src/dataset_analysis.py`)
- Comprehensive dataset statistics
- Class distribution analysis
- Image property analysis (dimensions, formats)
- Visualization generation

**Key Features:**
- Analyzes both training and testing splits
- Reports class imbalance
- Identifies image format inconsistencies
- Generates distribution charts

### 2. ML Pipeline (`src/ml_pipeline.py`)
Complete pipeline with multiple model architectures:

#### Supported Models:
- **CNN**: Custom convolutional neural network (lightweight)
- **MobileNet**: Mobile-optimized transfer learning
- **EfficientNet**: High-efficiency transfer learning (recommended)

#### Features:
- Automatic data loading and preprocessing
- Configurable model architecture
- Data augmentation (rotation, zoom, flip)
- Training with early stopping
- Comprehensive evaluation metrics
- Visualization generation

### 3. Training Script (`train_script/train.py`)
End-to-end execution orchestrating all pipeline components.

## ⚙️ Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    'image_size': (224, 224),      # Input image size
    'batch_size': 32,               # Batch size for training
    'epochs': 20,                   # Number of training epochs
    'learning_rate': 0.001,         # Optimizer learning rate
    'validation_split': 0.2,        # Train/val split ratio
    'data_augmentation': True,      # Enable data augmentation
    'model_type': 'efficientnet',   # Model architecture
    'pretrained': True,             # Use ImageNet weights
}
```

## 📊 Output Files

After running the pipeline, you'll get:

1. **trained_model.h5** - Trained model weights
2. **dataset_analysis.png** - Dataset distribution charts
3. **training_history.png** - Accuracy and loss curves
4. **confusion_matrix.png** - Prediction analysis matrix

## 📋 Model Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| CNN | ⚡ Fast | 🟡 Medium | 💾 Low | Experiments |
| MobileNet | ⚡ Fast | 🟢 Good | 💾 Low | Mobile/Edge |
| EfficientNet | 🟡 Medium | 🟢 Good | 💾 Medium | Production |

## 🔍 Key Metrics

The pipeline reports:
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: False negative rate
- **F1-Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: Per-class breakdown

## 💡 Tips for Better Results

1. **Data Quality**: Ensure images are properly labeled and consistent format
2. **Image Size**: Larger images → better accuracy but slower training
3. **Data Augmentation**: Enable for small datasets, disable for large ones
4. **Batch Size**: Higher on GPUs, lower on CPUs
5. **Epochs**: Monitor validation loss to prevent overfitting
6. **Learning Rate**: Lower for fine-tuning, higher for training from scratch

## 🛠️ Advanced Usage

### Custom Model
Modify `src/ml_pipeline.py`:

```python
def _build_custom_model(self, input_shape, num_classes):
    # Your custom architecture
    pass
```

### Fine-tuning
For transfer learning, unlock base model layers:

```python
base_model.trainable = True  # Unlock for fine-tuning
```

### Data Augmentation
Customize in `pipeline.train()`:

```python
train_generator = ImageDataGenerator(
    rotation_range=30,  # Increase rotation
    zoom_range=0.3,     # Increase zoom
    # ... more augmentations
)
```

## 📚 References

- TensorFlow/Keras: https://tensorflow.org
- EfficientNet: https://arxiv.org/abs/1905.11946
- Medical Image Classification: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

## 📝 License

This project is provided as-is for educational and research purposes.

## ✅ Checklist

- [x] Dataset analysis module
- [x] Multiple model architectures
- [x] Data augmentation
- [x] Training pipeline
- [x] Evaluation metrics
- [x] Visualization generation
- [x] Model persistence
- [x] Configuration management
- [ ] Model serving (REST API)
- [ ] Real-time prediction
- [ ] Hyperparameter tuning
- [ ] Cross-validation

---

**Created:** 2025-10-29  
**Framework:** TensorFlow/Keras 2.13+  
**Python:** 3.8+
