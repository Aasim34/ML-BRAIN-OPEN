"""
PRODUCTION-READY INFERENCE SCRIPT
Comprehensive inference and prediction capabilities for trained models

Features:
- Single image prediction
- Batch prediction
- Confidence thresholds
- Result exportation
- Visualization of predictions
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Inference configuration"""
    MODEL_DIR = "a:\\ml\\models"
    RESULTS_DIR = "a:\\ml\\results"
    IMAGE_SIZE = (224, 224)
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
    CONFIDENCE_THRESHOLD = 0.6

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Production-grade inference engine"""
    
    def __init__(self, model_path: str, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.class_names = self.config.CLASS_NAMES
        self._load_model_and_metadata()
    
    def _load_model_and_metadata(self):
        """Load model and metadata"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Model loaded: {self.model_path}")
        
        # Try to load metadata
        metadata_path = self.model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata.get('class_names', self.class_names)
            print(f"✅ Metadata loaded: {metadata_path}")
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Predict class for single image"""
        try:
            # Load and preprocess
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.config.IMAGE_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            return {
                'success': True,
                'image_path': str(image_path),
                'predicted_class': self.class_names[predicted_idx],
                'confidence': confidence,
                'confidence_pct': f"{confidence*100:.2f}%",
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                },
                'meets_threshold': confidence >= self.config.CONFIDENCE_THRESHOLD,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'image_path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, image_paths: List[str], 
                     save_results: bool = True) -> List[Dict[str, Any]]:
        """Predict for multiple images"""
        print(f"\n🔮 Running inference on {len(image_paths)} images...")
        results = []
        
        for idx, img_path in enumerate(image_paths):
            print(f"  [{idx+1}/{len(image_paths)}] Processing: {img_path}", end="")
            result = self.predict_image(img_path)
            results.append(result)
            
            status = "✅" if result['success'] else "❌"
            if result['success']:
                print(f" {status} {result['predicted_class']} ({result['confidence_pct']})")
            else:
                print(f" {status} Error: {result['error']}")
        
        if save_results:
            self._save_batch_results(results)
        
        return results
    
    def predict_directory(self, directory_path: str,
                         save_results: bool = True) -> List[Dict[str, Any]]:
        """Predict for all images in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        image_files = [
            str(f) for f in Path(directory_path).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"⚠️  No images found in {directory_path}")
            return []
        
        print(f"📁 Found {len(image_files)} images in {directory_path}")
        return self.predict_batch(image_files, save_results)
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Predict from numpy array"""
        try:
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Normalize if needed
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            predictions = self.model.predict(image_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            return {
                'success': True,
                'predicted_class': self.class_names[predicted_idx],
                'confidence': confidence,
                'confidence_pct': f"{confidence*100:.2f}%",
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                },
                'meets_threshold': confidence >= self.config.CONFIDENCE_THRESHOLD
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_batch_results(self, results: List[Dict[str, Any]]):
        """Save batch prediction results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.config.RESULTS_DIR}/predictions_{timestamp}.json"
        
        # Summary statistics
        successful = [r for r in results if r.get('success', False)]
        summary = {
            'total_images': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'timestamp': datetime.now().isoformat(),
            'predictions': results
        }
        
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"   Successful: {len(successful)}/{len(results)}")
        
        return output_path

# ============================================================================
# VISUALIZATION
# ============================================================================

class PredictionVisualizer:
    """Visualize predictions"""
    
    @staticmethod
    def visualize_single_prediction(image_path: str, result: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """Visualize single prediction"""
        if not result.get('success', False):
            print(f"Cannot visualize failed prediction: {result.get('error')}")
            return
        
        img = Image.open(image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Image
        axes[0].imshow(img)
        axes[0].set_title(f"Prediction: {result['predicted_class']}\n"
                         f"Confidence: {result['confidence_pct']}")
        axes[0].axis('off')
        
        # Probabilities
        probs = result['all_probabilities']
        classes = list(probs.keys())
        values = list(probs.values())
        colors = ['green' if c == result['predicted_class'] else 'lightblue' 
                 for c in classes]
        
        axes[1].barh(classes, values, color=colors)
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Class Probabilities')
        axes[1].set_xlim([0, 1])
        
        for i, v in enumerate(values):
            axes[1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def visualize_batch_results(results: List[Dict[str, Any]],
                               save_path: Optional[str] = None):
        """Visualize batch results"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("No successful predictions to visualize")
            return
        
        # Class distribution in predictions
        class_counts = {}
        for result in successful_results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Confidence distribution
        confidences = [r['confidence'] for r in successful_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class distribution
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        axes[0].bar(classes, counts, color='steelblue')
        axes[0].set_title('Predicted Class Distribution')
        axes[0].set_ylabel('Count')
        
        # Confidence distribution
        axes[1].hist(confidences, bins=20, color='green', alpha=0.7)
        axes[1].set_title('Confidence Score Distribution')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=np.mean(confidences), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Batch visualization saved to {save_path}")
        
        plt.show()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_latest_model(model_dir: str = "a:\\ml\\models") -> Optional[str]:
    """Get the latest trained model"""
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        print(f"No models found in {model_dir}")
        return None
    
    model_files.sort()
    latest = os.path.join(model_dir, model_files[-1])
    print(f"✅ Found latest model: {latest}")
    return latest

def benchmark_inference(model_path: str, test_image_path: str, 
                       num_runs: int = 5) -> Dict[str, float]:
    """Benchmark inference speed"""
    import time
    
    engine = InferenceEngine(model_path)
    times = []
    
    print(f"\n⏱️  Benchmarking inference ({num_runs} runs)...")
    
    for i in range(num_runs):
        start = time.time()
        engine.predict_image(test_image_path)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    times = np.array(times)
    results = {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'unit': 'seconds'
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Mean: {results['mean']*1000:.2f}ms")
    print(f"   Std:  {results['std']*1000:.2f}ms")
    print(f"   Min:  {results['min']*1000:.2f}ms")
    print(f"   Max:  {results['max']*1000:.2f}ms")
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - INFERENCE ENGINE 🧠")
    print("="*70)
    
    # Example 1: Single image prediction
    # model_path = get_latest_model()
    # if model_path:
    #     engine = InferenceEngine(model_path)
    #     result = engine.predict_image("path/to/image.jpg")
    #     print(f"Prediction: {result}")
    
    # Example 2: Batch prediction
    # results = engine.predict_batch([
    #     "path/to/image1.jpg",
    #     "path/to/image2.jpg",
    #     "path/to/image3.jpg"
    # ])
    
    # Example 3: Directory prediction
    # results = engine.predict_directory("path/to/test/folder")
    
    # Example 4: Visualization
    # if results:
    #     PredictionVisualizer.visualize_single_prediction(
    #         results[0]['image_path'],
    #         results[0],
    #         save_path="prediction_result.png"
    #     )
    
    print("✅ Inference engine ready to use!")
    print("\nExample usage:")
    print("  engine = InferenceEngine('path/to/model.h5')")
    print("  result = engine.predict_image('path/to/image.jpg')")
