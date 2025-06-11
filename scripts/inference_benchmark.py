import os
import time
import pickle
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

# Import models
from models.uni import UNI
from models.uni2 import UNI2
from models.phikon import Phikon
from models.phikonv2 import PhikonV2
from etc.efficientnetb0 import EfficientNetB0Classifier

# Import dataset
from data.glomerulus import GlomerulusDataset

class InferenceBenchmark:
    def __init__(self, 
                 data_root: str = "/datasets/terumo-data-jpeg/",
                 svm_root: str = "etc/models_SVM/",
                 device: str = "cuda"):
        """
        Initialize the benchmark class.
        
        Args:
            data_root: Path to the dataset
            svm_root: Path to SVM checkpoints and EfficientNet checkpoints
                     Expected structure:
                     svm_root/
                     ├── UNI/
                     ├── UNI2/
                     ├── PHIKON/
                     ├── PHIKONV2/
                     └── EFFICIENTNETB0/  (optional, for trained EfficientNet weights)
            device: Device to run inference on
        """
        self.data_root = data_root
        self.svm_root = svm_root
        self.device = device
        
        # Define model classes and their corresponding folder names
        self.models = {
            'UNI': UNI,
            'UNI2': UNI2, 
            'PHIKON': Phikon,
            'PHIKONV2': PhikonV2,
            'EFFICIENTNETB0': EfficientNetB0Classifier
        }
        
        # Define classes
        self.classes = ["Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis"]
        
        # Results storage
        self.results = []
        
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**2,    # MB
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        }
    
    def reset_gpu_memory_stats(self):
        """Reset GPU memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        
    def load_efficientnet_checkpoint(self, class_name: str, model: EfficientNetB0Classifier) -> EfficientNetB0Classifier:
        """Load trained EfficientNet checkpoint for a specific class."""
        # Look for EfficientNet checkpoints (assuming they follow a similar naming pattern)
        efficientnet_folder = os.path.join(self.svm_root, "EFFICIENTNETB0")
        
        if not os.path.exists(efficientnet_folder):
            print(f"Warning: No EfficientNet checkpoints found at {efficientnet_folder}")
            print("Using randomly initialized classifier weights")
            return model
        
        # Find the checkpoint file for this class
        for file in os.listdir(efficientnet_folder):
            if class_name in file and (file.endswith('.pth') or file.endswith('.pt')):
                checkpoint_path = os.path.join(efficientnet_folder, file)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"Loaded EfficientNet checkpoint: {file}")
                    return model
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {file}: {e}")
                    continue
        
        print(f"Warning: No valid checkpoint found for EfficientNet - {class_name}")
        print("Using randomly initialized classifier weights")
        return model
    
    def load_svm_checkpoint(self, model_name: str, class_name: str) -> object:
        """Load the appropriate SVM checkpoint for a model and class."""
        # EfficientNet doesn't use SVM, it has its own classifier
        if model_name == 'EFFICIENTNETB0':
            return None
            
        svm_folder = os.path.join(self.svm_root, model_name.upper())
        
        # Find the checkpoint file for this class
        for file in os.listdir(svm_folder):
            if class_name in file and file.endswith('.pkl'):
                checkpoint_path = os.path.join(svm_folder, file)
                with open(checkpoint_path, 'rb') as f:
                    svm_model = pickle.load(f)
                return svm_model
        
        raise FileNotFoundError(f"No SVM checkpoint found for {model_name} - {class_name}")
    
    def extract_features_batch(self, model, dataloader: DataLoader) -> np.ndarray:
        """Extract features from a batch of images using the foundation model."""
        features = []
        
        model.eval()
        with torch.no_grad():
            for batch_imgs, _, _ in dataloader:
                if isinstance(batch_imgs, torch.Tensor):
                    batch_imgs = batch_imgs.to(self.device)
                
                # Extract features
                if hasattr(model, 'extract_features'):
                    # EfficientNet case
                    batch_features = model.extract_features(batch_imgs)
                else:
                    # Foundation models case
                    batch_features = model(batch_imgs)
                
                # Convert to numpy
                if isinstance(batch_features, torch.Tensor):
                    batch_features = batch_features.cpu().numpy()
                
                features.append(batch_features)
        
        return np.vstack(features)
    
    def benchmark_model_class(self, 
                            model_name: str, 
                            class_name: str, 
                            sample_size: int = 100) -> Dict:
        """
        Benchmark a specific model-class combination.
        
        Args:
            model_name: Name of the foundation model
            class_name: Name of the class to test
            sample_size: Number of samples to test
            
        Returns:
            Dictionary with timing and memory results
        """
        print(f"\n--- Benchmarking {model_name} for {class_name} ---")

        # Reset GPU memory statistics at the start
        self.reset_gpu_memory_stats()
        baseline_memory = self.get_gpu_memory_usage()

        transforms = T.Compose([
            T.ToTensor(),
            T.Resize((224,224)),
            T.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),  # Handle RGBA by keeping only RGB channels
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
        ])
        
        # Load dataset for one-vs-all classification
        dataset = GlomerulusDataset(
            root_dir=self.data_root,
            classes=self.classes,
            one_vs_all=class_name,
            transforms=transforms,
            consider_augmented=True
        )
        
        # Limit sample size if specified
        if sample_size and len(dataset) > sample_size:
            # Take a random subset
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            dataset.data = [dataset.data[i] for i in indices]
        
        print(f"Testing on {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Initialize model
        print(f"Loading {model_name} model...")
        model_start = time.time()
        model = self.models[model_name](device=self.device)
        
        # Load trained weights for EfficientNet
        if model_name == 'EFFICIENTNETB0':
            model = self.load_efficientnet_checkpoint(class_name, model)
            
        model_load_time = time.time() - model_start
        model_memory = self.get_gpu_memory_usage()
        print(f"Model loaded in {model_load_time:.3f}s")
        print(f"GPU memory after model loading: {model_memory['allocated']:.1f}MB allocated, {model_memory['reserved']:.1f}MB reserved")
        
        # Load SVM checkpoint (skip for EfficientNet)
        svm_model = None
        svm_load_time = 0
        if model_name != 'EFFICIENTNETB0':
            print(f"Loading SVM checkpoint for {class_name}...")
            svm_start = time.time()
            svm_model = self.load_svm_checkpoint(model_name, class_name)
            svm_load_time = time.time() - svm_start
            print(f"SVM loaded in {svm_load_time:.3f}s")
        else:
            print("Using EfficientNet's built-in classifier")
        
        # Extract features
        print("Extracting features...")
        feature_start = time.time()
        features = self.extract_features_batch(model, dataloader)
        feature_time = time.time() - feature_start
        feature_memory = self.get_gpu_memory_usage()
        print(f"Feature extraction completed in {feature_time:.3f}s")
        print(f"GPU memory after feature extraction: {feature_memory['allocated']:.1f}MB allocated, {feature_memory['reserved']:.1f}MB reserved")
        
        # Classification
        print("Running classification...")
        svm_start = time.time()
        if model_name == 'EFFICIENTNETB0':
            # Use EfficientNet's own classifier
            predictions = []
            probabilities = []
            
            model.eval()
            with torch.no_grad():
                for batch_imgs, _, _ in dataloader:
                    batch_imgs = batch_imgs.to(self.device)
                    preds, probs = model.predict(batch_imgs)
                    predictions.extend(preds.flatten())
                    probabilities.extend(probs.flatten())
            
            predictions = np.array(predictions)
            probabilities = np.array(probabilities)
        else:
            # Use SVM classifier
            predictions = svm_model.predict(features)
            probabilities = svm_model.predict_proba(features) if hasattr(svm_model, 'predict_proba') else None
            
        svm_time = time.time() - svm_start
        classifier_name = "Neural Network" if model_name == 'EFFICIENTNETB0' else "SVM"
        final_memory = self.get_gpu_memory_usage()
        print(f"{classifier_name} inference completed in {svm_time:.3f}s")
        print(f"Final GPU memory usage: {final_memory['allocated']:.1f}MB allocated, {final_memory['reserved']:.1f}MB reserved")
        print(f"Peak GPU memory usage: {final_memory['max_allocated']:.1f}MB")
        
        # Calculate per-sample times
        total_inference_time = feature_time + svm_time
        per_sample_time = total_inference_time / len(dataset)
        
        # Calculate memory usage
        model_memory_usage = model_memory['allocated'] - baseline_memory['allocated']
        total_memory_usage = final_memory['max_allocated'] - baseline_memory['allocated']
        
        results = {
            'model_name': model_name,
            'class_name': class_name,
            'sample_count': len(dataset),
            'model_load_time': model_load_time,
            'svm_load_time': svm_load_time,
            'feature_extraction_time': feature_time,
            'svm_inference_time': svm_time,
            'total_inference_time': total_inference_time,
            'per_sample_time': per_sample_time,
            'feature_dim': model.get_feat_dim(),
            'classifier_type': 'Neural Network' if model_name == 'EFFICIENTNETB0' else 'SVM',
            'baseline_memory_mb': baseline_memory['allocated'],
            'model_memory_mb': model_memory['allocated'],
            'feature_memory_mb': feature_memory['allocated'],
            'final_memory_mb': final_memory['allocated'],
            'peak_memory_mb': final_memory['max_allocated'],
            'model_memory_usage_mb': model_memory_usage,
            'total_memory_usage_mb': total_memory_usage,
            'reserved_memory_mb': final_memory['reserved']
        }
        
        print(f"Results: {per_sample_time*1000:.2f}ms per sample, {total_memory_usage:.1f}MB peak GPU memory")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_full_benchmark(self, sample_size: int = 100):
        """Run benchmark for all model-class combinations."""
        print("Starting full benchmark...")
        print(f"Sample size per test: {sample_size}")
        print(f"Device: {self.device}")
        print(f"Models: {list(self.models.keys())}")
        print(f"Classes: {self.classes}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        for model_name in self.models.keys():
            for class_name in self.classes:
                try:
                    result = self.benchmark_model_class(model_name, class_name, sample_size)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {model_name} - {class_name}: {e}")
                    continue
        
        print(f"\nBenchmark completed! {len(self.results)} tests run.")
    
    def save_results(self, output_path: str = "inference_benchmark_results.csv"):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        if not self.results:
            print("No results to summarize!")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"Total tests run: {len(df)}")
        print(f"Average per-sample inference time: {df['per_sample_time'].mean()*1000:.2f}ms")
        print(f"Average peak GPU memory usage: {df['peak_memory_mb'].mean():.1f}MB")
        print(f"Average total memory usage: {df['total_memory_usage_mb'].mean():.1f}MB")
        
        # Per-model summary
        print("\nPER-MODEL PERFORMANCE:")
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            avg_time = model_data['per_sample_time'].mean() * 1000
            std_time = model_data['per_sample_time'].std() * 1000
            avg_memory = model_data['total_memory_usage_mb'].mean()
            std_memory = model_data['total_memory_usage_mb'].std()
            avg_peak = model_data['peak_memory_mb'].mean()
            feat_dim = model_data['feature_dim'].iloc[0]
            
            print(f"{model:>12}: {avg_time:6.2f}±{std_time:5.2f}ms | "
                  f"Memory: {avg_memory:6.1f}±{std_memory:5.1f}MB | "
                  f"Peak: {avg_peak:6.1f}MB | Feat: {feat_dim}")
        
        # Per-class summary
        print("\nPER-CLASS PERFORMANCE:")
        for class_name in df['class_name'].unique():
            class_data = df[df['class_name'] == class_name]
            avg_time = class_data['per_sample_time'].mean() * 1000
            avg_memory = class_data['total_memory_usage_mb'].mean()
            
            print(f"{class_name:>15}: {avg_time:6.2f}ms | Memory: {avg_memory:6.1f}MB")
        
        # Memory breakdown
        print("\nMEMORY BREAKDOWN BY MODEL:")
        memory_columns = ['model_memory_usage_mb', 'total_memory_usage_mb', 'peak_memory_mb']
        memory_summary = df.groupby('model_name')[memory_columns].mean().round(1)
        print(memory_summary)
        
        print("="*80)

def main():
    # Configuration
    DATA_ROOT = "/datasets/terumo-data-jpeg/"
    SVM_ROOT = "etc/models_SVM/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_SIZE = 100  # Adjust based on your needs
    
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    
    # Initialize benchmark
    benchmark = InferenceBenchmark(
        data_root=DATA_ROOT,
        svm_root=SVM_ROOT,
        device=DEVICE
    )
    
    # Run benchmark
    benchmark.run_full_benchmark(sample_size=SAMPLE_SIZE)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results("inference_benchmark_results.csv")

if __name__ == "__main__":
    main()