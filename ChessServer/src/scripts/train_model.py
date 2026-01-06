#!/usr/bin/env python3
"""
YOLO Training Script for Chess Piece Detection

This script handles training/fine-tuning of YOLO models for improved
chess piece detection, especially for:
- Angled views
- Overlapping pieces
- Various lighting conditions

Usage:
    python train_model.py --data datasets/final/data.yaml --epochs 100
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"

# Training hyperparameters optimized for chess pieces
TRAINING_CONFIG = {
    # Model selection
    "base_model": "yolov8m.pt",  # Medium model - good balance
    
    # Training params
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "patience": 20,  # Early stopping
    
    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,  # Final LR ratio
    "momentum": 0.937,
    "weight_decay": 0.0005,
    
    # Augmentation (Ultralytics built-in)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 20.0,       # Rotation range
    "translate": 0.1,
    "scale": 0.3,
    "shear": 5.0,
    "perspective": 0.001,
    "flipud": 0.0,         # No vertical flip for chess
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,     # Good for overlapping pieces
    
    # Other
    "workers": 8,
    "project": str(RUNS_DIR / "detect"),
    "exist_ok": False,
    "pretrained": True,
    "verbose": True,
    "seed": 42,
}


# =============================================================================
# Training Functions
# =============================================================================

def train_model(data_yaml: Path, custom_config: dict = None):
    """Train YOLO model with optimized settings for chess pieces."""
    
    config = TRAINING_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"chess_pieces_{timestamp}"
    config["name"] = run_name
    
    print("=" * 60)
    print("🚀 TRAINING CHESS PIECE DETECTION MODEL")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base model: {config['base_model']}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Run name: {run_name}")
    print()
    
    # Load base model
    model = YOLO(config.pop("base_model"))
    
    # Start training
    results = model.train(
        data=str(data_yaml),
        **config
    )
    
    # Get best model path
    best_model = Path(config["project"]) / run_name / "weights" / "best.pt"
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest model saved to: {best_model}")
    
    return best_model, results


def export_model(model_path: Path, formats: list = ["onnx"]):
    """Export trained model to deployment formats."""
    
    print("\n" + "=" * 60)
    print("📦 EXPORTING MODEL")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    exported_paths = []
    for fmt in formats:
        print(f"\nExporting to {fmt}...")
        path = model.export(format=fmt)
        exported_paths.append(path)
        print(f"  ✅ Saved: {path}")
    
    return exported_paths


def evaluate_model(model_path: Path, data_yaml: Path):
    """Evaluate model performance on validation set."""
    
    print("\n" + "=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)
    
    model = YOLO(model_path)
    results = model.val(data=str(data_yaml))
    
    print("\nResults:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def compare_models(old_model: Path, new_model: Path, test_images_dir: Path):
    """Compare predictions between old and new models."""
    
    print("\n" + "=" * 60)
    print("🔍 COMPARING MODELS")
    print("=" * 60)
    
    old = YOLO(old_model)
    new = YOLO(new_model)
    
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    print(f"\nTesting on {len(test_images)} images...")
    
    for img_path in test_images[:5]:  # Test first 5
        print(f"\n{img_path.name}:")
        
        old_results = old.predict(img_path, verbose=False)
        new_results = new.predict(img_path, verbose=False)
        
        old_count = len(old_results[0].boxes)
        new_count = len(new_results[0].boxes)
        
        old_conf = old_results[0].boxes.conf.mean().item() if old_count > 0 else 0
        new_conf = new_results[0].boxes.conf.mean().item() if new_count > 0 else 0
        
        print(f"  Old model: {old_count} pieces, avg confidence: {old_conf:.3f}")
        print(f"  New model: {new_count} pieces, avg confidence: {new_conf:.3f}")


def deploy_model(model_path: Path, destination: Path = None):
    """Copy model to deployment location."""
    
    if destination is None:
        destination = MODELS_DIR / "detect_pieces_v2.onnx"
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(model_path, destination)
    
    print(f"\n✅ Model deployed to: {destination}")
    return destination


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train YOLO for chess piece detection")
    
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="Base model to fine-tune")
    parser.add_argument("--export", action="store_true",
                       help="Export to ONNX after training")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model after training")
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy model after training")
    
    args = parser.parse_args()
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ Data file not found: {data_yaml}")
        sys.exit(1)
    
    # Custom config from args
    custom_config = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "base_model": args.model,
    }
    
    # Train
    best_model, results = train_model(data_yaml, custom_config)
    
    # Evaluate
    if args.evaluate:
        evaluate_model(best_model, data_yaml)
    
    # Export
    if args.export or args.deploy:
        exported = export_model(best_model, ["onnx"])
        
        if args.deploy and exported:
            deploy_model(Path(exported[0]))


if __name__ == "__main__":
    main()
