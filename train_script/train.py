"""
Brain Tumor Classification Training Script
Supports binary and multi-class classification tasks
"""

import os
import sys
from pathlib import Path
import json
import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
config = {
    # Data paths
    'data_dir': 'a:/BRAIN_ML/dataset/Training',  # Change this to your dataset path
    'test_dir': 'a:/BRAIN_ML/dataset/Testing',   # Change this to your test dataset path
    
    # Model parameters
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'num_classes': 4,  # 4 for multi-class: glioma, meningioma, notumor, pituitary
    'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
    
    # Training parameters
    'learning_rate': 0.001,
    'patience': 10,  # For early stopping
    
    # Model architecture
    'model_type': 'efficientnet',  # 'efficientnet', 'mobilenet', 'cnn'
    
    # Augmentation parameters
    'augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': False,
        'fill_mode': 'nearest'
    },
    
    # Output paths
    'model_save_path': 'a:/BRAIN_ML/models/brain_tumor_model_{timestamp}.h5',
    'metrics_save_path': 'a:/BRAIN_ML/models/training_metrics_{timestamp}.json',
    'plots_save_path': 'a:/BRAIN_ML/models/training_plots_{timestamp}.png'
}

def create_data_generators():
    """Create data generators for training, validation, and testing"""
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config['augmentation']['rotation_range'],
        width_shift_range=config['augmentation']['width_shift_range'],
        height_shift_range=config['augmentation']['height_shift_range'],
        shear_range=config['augmentation']['shear_range'],
        zoom_range=config['augmentation']['zoom_range'],
        horizontal_flip=config['augmentation']['horizontal_flip'],
        vertical_flip=config['augmentation']['vertical_flip'],
        fill_mode=config['augmentation']['fill_mode'],
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    print("Creating data generators...")
    print(f"Training data path: {config['data_dir']}")
    print(f"Test data path: {config['test_dir']}")
    
    # Check if directories exist
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(f"Training directory not found: {config['data_dir']}")
    if not os.path.exists(config['test_dir']):
        raise FileNotFoundError(f"Test directory not found: {config['test_dir']}")
    
    train_generator = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        classes=config['class_names'],
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        classes=config['class_names'],
        subset='validation',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        config['test_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        classes=config['class_names'],
        shuffle=False
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    
    return train_generator, validation_generator, test_generator

def build_model():
    """Build and compile the model"""
    print(f"Building {config['model_type']} model...")
    
    if config['model_type'] == 'efficientnet':
        # Use EfficientNetB0 as the base model
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*config['img_size'], 3)
        )
    elif config['model_type'] == 'mobilenet':
        # Use MobileNetV2 as the base model
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*config['img_size'], 3)
        )
    else:  # CNN
        # Simple CNN model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*config['img_size'], 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(config['num_classes'], activation='softmax')
        ])
        return model
    
    # Freeze base model layers for transfer learning
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(*config['img_size'], 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(config['num_classes'], activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks():
    """Create callbacks for training"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = config['model_save_path'].format(timestamp=timestamp)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path.replace('.h5', '_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks, model_save_path

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def evaluate_model(model, test_generator, class_names):
    """Evaluate the model and generate reports"""
    print("Evaluating model on test set...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return test_accuracy, test_loss, report, cm

def save_training_metrics(history, test_metrics, model_save_path, timestamp):
    """Save training metrics to JSON file"""
    metrics_save_path = config['metrics_save_path'].format(timestamp=timestamp)
    
    metrics = {
        "training_history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        },
        "test_metrics": {
            "test_accuracy": float(test_metrics[0]),
            "test_loss": float(test_metrics[1])
        },
        "training_config": config,
        "model_path": model_save_path,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training metrics saved to: {metrics_save_path}")

def main():
    """Main training function"""
    print("🧠 Brain Tumor Classification Training")
    print("=" * 50)
    
    try:
        # Create data generators
        train_gen, val_gen, test_gen = create_data_generators()
        
        # Build model
        model = build_model()
        print(f"Model built successfully with {model.count_params():,} parameters")
        
        # Create callbacks
        callbacks, model_save_path = create_callbacks()
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_gen,
            epochs=config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = model_save_path.format(timestamp=timestamp)
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Plot training history
        plots_save_path = config['plots_save_path'].format(timestamp=timestamp)
        plot_training_history(history, plots_save_path)
        print(f"Training plots saved to: {plots_save_path}")
        
        # Evaluate model
        test_metrics = evaluate_model(model, test_gen, config['class_names'])
        
        # Save training metrics
        save_training_metrics(history, test_metrics, final_model_path, timestamp)
        
        print("\n✅ Training completed successfully!")
        print(f"Final model: {final_model_path}")
        print(f"Test accuracy: {test_metrics[0]:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()