"""
PRODUCTION-READY BRAIN TUMOR CLASSIFICATION TRAINING SCRIPT
Complete ML pipeline with training, evaluation, and inference capabilities

This module provides:
- Dataset analysis and loading
- Model building (CNN, MobileNet, EfficientNet)
- Training with advanced features (augmentation, early stopping, scheduling)
- Evaluation and metrics calculation
- Model persistence and loading
- Inference and prediction
- Visualization generation
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Config:
    """Configuration management"""
    # Paths
    DATASET_PATH = "a:\\ml\\dataset"
    MODEL_DIR = "a:\\ml\\models"
    RESULTS_DIR = "a:\\ml\\results"
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    NORMALIZE = True
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model
    MODEL_TYPE = "efficientnet"  # "cnn", "mobilenet", "efficientnet"
    PRETRAINED = True
    
    # Augmentation
    AUGMENTATION = True
    ROTATION_RANGE = 20
    WIDTH_SHIFT = 0.2
    HEIGHT_SHIFT = 0.2
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 5
    LR_REDUCTION_FACTOR = 0.5
    LR_REDUCTION_PATIENCE = 3
    
    # Classes
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET ANALYSIS & LOADING
# ============================================================================

class DatasetManager:
    """Manages dataset loading, analysis, and preprocessing"""
    
    def __init__(self, dataset_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.class_names = None
        self.class_distribution = {}
        
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset structure and statistics"""
        print("\n" + "="*70)
        print("📊 DATASET ANALYSIS")
        print("="*70)
        
        stats = {}
        total_images = 0
        
        for split in ["Training", "Testing"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                print(f"⚠️  Warning: {split} folder not found")
                continue
            
            print(f"\n📁 {split} Set:")
            split_stats = {}
            
            for class_name in sorted(os.listdir(split_path)):
                class_path = split_path / class_name
                if os.path.isdir(class_path):
                    image_count = len([f for f in os.listdir(class_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
                    
                    if class_name not in self.class_distribution:
                        self.class_distribution[class_name] = {"total": 0, "train": 0, "test": 0}
                    
                    self.class_distribution[class_name]["total"] += image_count
                    self.class_distribution[class_name][split.lower()] = image_count
                    split_stats[class_name] = image_count
                    total_images += image_count
                    
                    print(f"  • {class_name:15s}: {image_count:5d} images")
            
            stats[split] = split_stats
        
        # Print summary
        print(f"\n{'─'*70}")
        print("SUMMARY:")
        for class_name, counts in self.class_distribution.items():
            pct = (counts["total"] / total_images * 100) if total_images > 0 else 0
            print(f"  {class_name:15s}: {counts['total']:5d} total ({pct:5.1f}%) "
                  f"[Train: {counts.get('train', 0):5d}, Test: {counts.get('test', 0):5d}]")
        print(f"  {'TOTAL':15s}: {total_images:5d} images")
        print(f"  Number of classes: {len(self.class_distribution)}")
        print("="*70)
        
        return stats
    
    def load_images(self, split: str = "Training") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load images from directory structure"""
        print(f"\n💾 Loading {split} images...")
        
        split_path = self.dataset_path / split
        images = []
        labels = []
        classes = sorted([d for d in os.listdir(split_path) 
                         if os.path.isdir(split_path / d)])
        
        if self.class_names is None:
            self.class_names = classes
        
        for class_idx, class_name in enumerate(classes):
            class_path = split_path / class_name
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            
            print(f"  Loading {class_name}: {len(image_files)} images...", end="")
            
            for img_file in image_files:
                try:
                    img_path = str(class_path / img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.image_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"⚠️  Error loading {img_file}: {str(e)}")
            
            print(" ✅")
        
        images = np.array(images)
        labels = keras.utils.to_categorical(labels, num_classes=len(classes))
        
        print(f"✅ Loaded {len(images)} images | Shape: {images.shape}")
        
        return images, labels, self.class_names

# ============================================================================
# MODEL BUILDING
# ============================================================================

class ModelBuilder:
    """Builds different neural network architectures"""
    
    @staticmethod
    def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build custom CNN model"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def build_mobilenet(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build MobileNet V2 model"""
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def build_efficientnet(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build EfficientNet B0 model"""
        base_model = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

# ============================================================================
# TRAINING ENGINE
# ============================================================================

class TrainingEngine:
    """Handles model training with advanced features"""
    
    def __init__(self, model: keras.Model, config: Config):
        self.model = model
        self.config = config
        self.history = None
        self.callbacks = self._setup_callbacks()
        
    def _setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup training callbacks"""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.LR_REDUCTION_FACTOR,
                patience=self.config.LR_REDUCTION_PATIENCE,
                min_lr=1e-6,
                verbose=1
            )
        ]
    
    def compile(self):
        """Compile model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled")
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray = None, y_val: np.ndarray = None) -> keras.callbacks.History:
        """Train model with data augmentation"""
        print(f"\n🚀 Training for {self.config.EPOCHS} epochs...")
        
        if self.config.AUGMENTATION:
            train_gen = ImageDataGenerator(
                rotation_range=self.config.ROTATION_RANGE,
                width_shift_range=self.config.WIDTH_SHIFT,
                height_shift_range=self.config.HEIGHT_SHIFT,
                horizontal_flip=self.config.HORIZONTAL_FLIP,
                zoom_range=self.config.ZOOM_RANGE,
                fill_mode='nearest'
            )
            train_gen.fit(x_train)
            
            self.history = self.model.fit(
                train_gen.flow(x_train, y_train, batch_size=self.config.BATCH_SIZE),
                validation_data=(x_val, y_val) if x_val is not None else None,
                epochs=self.config.EPOCHS,
                callbacks=self.callbacks,
                steps_per_epoch=len(x_train) // self.config.BATCH_SIZE,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val) if x_val is not None else None,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                callbacks=self.callbacks,
                verbose=1
            )
        
        print("✅ Training completed")
        return self.history

# ============================================================================
# EVALUATION & METRICS
# ============================================================================

class ModelEvaluator:
    """Evaluates model performance"""
    
    def __init__(self, model: keras.Model, class_names: List[str]):
        self.model = model
        self.class_names = class_names
        self.metrics = {}
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test set"""
        print(f"\n📊 EVALUATING MODEL")
        print("="*70)
        
        # Get predictions
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        
        self.metrics = {
            'accuracy': float(accuracy),
            'predictions': y_pred.tolist(),
            'true_labels': y_test_labels.tolist(),
            'probabilities': y_pred_probs.tolist()
        }
        
        print(f"\n✅ Test Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test_labels, y_pred, target_names=self.class_names))
        print("="*70)
        
        return self.metrics
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {filepath}")

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Generates visualizations"""
    
    @staticmethod
    def plot_training_history(history: keras.callbacks.History, save_path: str):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training', linewidth=2)
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training history saved to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], save_path: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {save_path}")
    
    @staticmethod
    def plot_class_distribution(class_distribution: Dict, save_path: str):
        """Plot class distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        classes = list(class_distribution.keys())
        totals = [class_distribution[c]['total'] for c in classes]
        training = [class_distribution[c].get('train', 0) for c in classes]
        testing = [class_distribution[c].get('test', 0) for c in classes]
        
        # Overall distribution
        axes[0].bar(classes, totals, color='steelblue', alpha=0.7)
        axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Images')
        axes[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(totals):
            axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Train/Test split
        x = np.arange(len(classes))
        width = 0.35
        axes[1].bar(x - width/2, training, width, label='Training', alpha=0.8)
        axes[1].bar(x + width/2, testing, width, label='Testing', alpha=0.8)
        axes[1].set_title('Train/Test Split by Class', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Images')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, rotation=45)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Class distribution saved to {save_path}")

# ============================================================================
# MODEL INFERENCE
# ============================================================================

class ModelInference:
    """Inference and prediction wrapper"""
    
    def __init__(self, model: keras.Model, class_names: List[str], 
                 image_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
    
    def predict_single(self, image_path: str, confidence_threshold: float = 0.0) -> Dict[str, Any]:
        """Predict class for single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            return {
                'image': str(image_path),
                'predicted_class': self.class_names[predicted_idx],
                'confidence': confidence,
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                },
                'above_threshold': confidence >= confidence_threshold
            }
        except Exception as e:
            return {'error': str(e), 'image': str(image_path)}
    
    def predict_batch(self, image_paths: List[str], 
                     confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict_single(img_path, confidence_threshold)
            results.append(result)
        return results
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Predict from numpy array"""
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(image_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        
        return {
            'predicted_class': self.class_names[predicted_idx],
            'confidence': float(predictions[0][predicted_idx]),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

class ModelManager:
    """Save and load models"""
    
    @staticmethod
    def save_model(model: keras.Model, filepath: str, metadata: Dict = None):
        """Save model and metadata"""
        model.save(filepath)
        print(f"✅ Model saved to {filepath}")
        
        if metadata:
            metadata_path = filepath.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_model(filepath: str) -> keras.Model:
        """Load saved model"""
        model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model
    
    @staticmethod
    def load_metadata(filepath: str) -> Dict:
        """Load model metadata"""
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - ML PRODUCTION PIPELINE 🧠")
    print("="*70)
    
    # Setup
    config = Config()
    config.create_directories()
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Dataset Analysis
    # ─────────────────────────────────────────────────────────────────────
    dataset_manager = DatasetManager(config.DATASET_PATH, config.IMAGE_SIZE)
    dataset_manager.analyze_dataset()
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Load Data
    # ─────────────────────────────────────────────────────────────────────
    x_train, y_train, class_names = dataset_manager.load_images("Training")
    x_test, y_test, _ = dataset_manager.load_images("Testing")
    
    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=SEED,
        stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training:   {x_train.shape[0]} images")
    print(f"  Validation: {x_val.shape[0]} images")
    print(f"  Testing:    {x_test.shape[0]} images")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Build Model
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n🏗️  Building {config.MODEL_TYPE} model...")
    builder = ModelBuilder()
    input_shape = config.IMAGE_SIZE + (3,)
    
    if config.MODEL_TYPE.lower() == 'cnn':
        model = builder.build_cnn(input_shape, len(class_names))
    elif config.MODEL_TYPE.lower() == 'mobilenet':
        model = builder.build_mobilenet(input_shape, len(class_names))
    elif config.MODEL_TYPE.lower() == 'efficientnet':
        model = builder.build_efficientnet(input_shape, len(class_names))
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    print(f"✅ Model built")
    print(f"   Total parameters: {model.count_params():,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Train Model
    # ─────────────────────────────────────────────────────────────────────
    trainer = TrainingEngine(model, config)
    trainer.compile()
    history = trainer.train(x_train, y_train, x_val, y_val)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Evaluate Model
    # ─────────────────────────────────────────────────────────────────────
    evaluator = ModelEvaluator(model, class_names)
    metrics = evaluator.evaluate(x_test, y_test)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Visualizations
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n📊 Generating visualizations...")
    Visualizer.plot_training_history(history, f"{config.RESULTS_DIR}/training_history.png")
    Visualizer.plot_confusion_matrix(
        np.argmax(y_test, axis=1),
        metrics['predictions'],
        class_names,
        f"{config.RESULTS_DIR}/confusion_matrix.png"
    )
    Visualizer.plot_class_distribution(
        dataset_manager.class_distribution,
        f"{config.RESULTS_DIR}/class_distribution.png"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Save Model & Metadata
    # ─────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{config.MODEL_DIR}/brain_tumor_model_{timestamp}.h5"
    
    metadata = {
        'model_type': config.MODEL_TYPE,
        'image_size': config.IMAGE_SIZE,
        'class_names': class_names,
        'accuracy': metrics['accuracy'],
        'created': datetime.now().isoformat(),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
        }
    }
    
    ModelManager.save_model(model, model_path, metadata)
    evaluator.save_metrics(f"{config.RESULTS_DIR}/metrics_{timestamp}.json")
    
    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n" + "="*70)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📈 Final Results:")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n💾 Generated Files:")
    print(f"   Model:                  {model_path}")
    print(f"   Training History:       {config.RESULTS_DIR}/training_history.png")
    print(f"   Confusion Matrix:       {config.RESULTS_DIR}/confusion_matrix.png")
    print(f"   Class Distribution:     {config.RESULTS_DIR}/class_distribution.png")
    print(f"   Metrics JSON:           {config.RESULTS_DIR}/metrics_{timestamp}.json")
    print("="*70)

if __name__ == "__main__":
    main()
