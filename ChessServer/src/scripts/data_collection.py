#!/usr/bin/env python3
"""
Chess Dataset Collection & Augmentation Script

Downloads chess piece datasets from various sources and applies augmentations
for improving YOLO model training.

Usage:
    python data_collection.py --download     # Download datasets
    python data_collection.py --augment      # Apply augmentations
    python data_collection.py --prepare      # Prepare for training
"""

import os
import sys
import argparse
import shutil
import zipfile
import random
from pathlib import Path
from typing import Optional

# Try importing required libraries
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm")
    import requests
    from tqdm import tqdm

try:
    import cv2
    import numpy as np
except ImportError:
    print("Installing opencv...")
    os.system(f"{sys.executable} -m pip install opencv-python numpy")
    import cv2
    import numpy as np

try:
    import albumentations as A
except ImportError:
    print("Installing albumentations...")
    os.system(f"{sys.executable} -m pip install albumentations")
    import albumentations as A


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "datasets"
RAW_DIR = DATA_DIR / "raw"
AUGMENTED_DIR = DATA_DIR / "augmented"
FINAL_DIR = DATA_DIR / "final"

# Dataset sources - Updated January 2026
DATASETS = {
    "roboflow_digital_university": {
        "workspace": "digital-university-u0zkl",
        "project": "chess-pieces-hxfdi", 
        "version": 1,
        "description": "4621 images from digital-university (May 2024)"
    },
    "roboflow_chess_v1": {
        "workspace": "test-kddvk",
        "project": "chess-lhfrd",
        "version": 1,
        "description": "4815 images (September 2024)"
    },
    "kaggle_chess": {
        "dataset": "roboflow-100/chess-pieces-detection-image-dataset",
        "description": "606 images from Kaggle/Roboflow"
    }
}

# Manual download instructions
MANUAL_DOWNLOAD_INSTRUCTIONS = """
╔════════════════════════════════════════════════════════════════════╗
║                    MANUAL DATASET DOWNLOAD                         ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Option 1: Roboflow (Recommended - ~4600 images)                   ║
║  ──────────────────────────────────────────────────────────────    ║
║  1. Go to: https://universe.roboflow.com/search?q=chess+pieces     ║
║  2. Select a dataset (e.g., "Chess-pieces by digital university")  ║
║  3. Click "Download Dataset" → Format: YOLOv8                      ║
║  4. Extract to: datasets/raw/roboflow/                             ║
║                                                                    ║
║  Option 2: Kaggle (606 images)                                     ║
║  ──────────────────────────────────────────────────────────────    ║
║  1. Go to: https://www.kaggle.com/roboflow-100                     ║
║  2. Search "chess pieces detection"                                ║
║  3. Download and extract to: datasets/raw/kaggle/                  ║
║                                                                    ║
║  Option 3: Use Roboflow API (automatic)                            ║
║  ──────────────────────────────────────────────────────────────    ║
║  1. Create free account at https://roboflow.com                    ║
║  2. Get API key from settings                                      ║
║  3. Run: python data_collection.py --download --api-key YOUR_KEY   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""

# Augmentation configuration for chess pieces
# Focused on handling angles, occlusions, and lighting variations
AUGMENTATION_CONFIG = {
    # Geometric transforms (handle different viewing angles)
    "rotation_range": (-25, 25),      # Degrees
    "perspective_range": 0.1,          # Perspective distortion
    "scale_range": (0.7, 1.3),        # Zoom in/out
    
    # Lighting variations
    "brightness_range": (0.6, 1.4),
    "contrast_range": (0.7, 1.3),
    "hue_shift_range": (-15, 15),
    
    # Blur/noise for robustness
    "blur_probability": 0.2,
    "noise_probability": 0.15,
    
    # Occlusion simulation
    "cutout_probability": 0.1,
    "cutout_size": (0.05, 0.15),
}


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_roboflow_dataset(api_key: Optional[str] = None, dataset_key: str = "roboflow_digital_university"):
    """Download dataset from Roboflow API."""
    print(f"\n📦 Downloading Roboflow dataset: {dataset_key}...")
    
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            print("⚠️  No Roboflow API key provided.")
            print(MANUAL_DOWNLOAD_INSTRUCTIONS)
            return False
    
    dataset_info = DATASETS.get(dataset_key)
    if not dataset_info or "workspace" not in dataset_info:
        print(f"❌ Invalid dataset key: {dataset_key}")
        return False
    
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        
        workspace = dataset_info["workspace"]
        project_name = dataset_info["project"]
        version = dataset_info.get("version", 1)
        
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov8", location=str(RAW_DIR / dataset_key))
        
        print(f"✅ Downloaded {dataset_info['description']} to {RAW_DIR / dataset_key}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from Roboflow: {e}")
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        return False


def download_kaggle_dataset():
    """Download chess dataset from Kaggle."""
    print("\n📦 Downloading Kaggle dataset...")
    
    try:
        import subprocess
        
        # Check if kaggle is configured
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print("⚠️  Kaggle not configured.")
            print("   1. Go to https://www.kaggle.com/settings")
            print("   2. Click 'Create New Token' under API")
            print("   3. Save kaggle.json to ~/.kaggle/")
            print(MANUAL_DOWNLOAD_INSTRUCTIONS)
            return False
        
        dest_dir = RAW_DIR / "kaggle"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = DATASETS["kaggle_chess"]["dataset"]
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset,
            "-p", str(dest_dir), "--unzip"
        ], check=True)
        
        print(f"✅ Downloaded Kaggle dataset to {dest_dir}")
        return True
        
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        return False


def download_all(api_key: Optional[str] = None):
    """Download all available datasets."""
    print("=" * 60)
    print("🔽 DOWNLOADING CHESS DATASETS")
    print("=" * 60)
    
    results = {}
    
    # Try Roboflow with API key if provided
    if api_key:
        for key in ["roboflow_digital_university", "roboflow_chess_v1"]:
            results[key] = download_roboflow_dataset(api_key, key)
    else:
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        results["roboflow"] = False
    
    # Try Kaggle
    results["kaggle"] = download_kaggle_dataset()
    
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    # Check if any data exists
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"\n✅ Found data in {RAW_DIR}")
        for d in RAW_DIR.iterdir():
            if d.is_dir():
                img_count = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
                print(f"   {d.name}: {img_count} images")
    
    return results


# =============================================================================
# Augmentation Functions
# =============================================================================

def create_augmentation_pipeline():
    """Create augmentation pipeline optimized for chess pieces."""
    
    return A.Compose([
        # Geometric transforms (crucial for angle variations)
        A.OneOf([
            A.Rotate(limit=AUGMENTATION_CONFIG["rotation_range"], p=0.7),
            A.Perspective(scale=AUGMENTATION_CONFIG["perspective_range"], p=0.5),
            A.Affine(
                scale=AUGMENTATION_CONFIG["scale_range"],
                rotate=AUGMENTATION_CONFIG["rotation_range"],
                shear=(-10, 10),
                p=0.6
            ),
        ], p=0.8),
        
        # Flip (horizontal only - vertical doesn't make sense for chess)
        A.HorizontalFlip(p=0.5),
        
        # Color/Lighting transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(AUGMENTATION_CONFIG["brightness_range"][0] - 1, 
                                  AUGMENTATION_CONFIG["brightness_range"][1] - 1),
                contrast_limit=(AUGMENTATION_CONFIG["contrast_range"][0] - 1,
                               AUGMENTATION_CONFIG["contrast_range"][1] - 1),
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=AUGMENTATION_CONFIG["hue_shift_range"],
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
        ], p=0.7),
        
        # Blur/Noise for robustness
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=AUGMENTATION_CONFIG["blur_probability"]),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=AUGMENTATION_CONFIG["noise_probability"]),
        ], p=0.3),
        
        # Simulate occlusion (pieces behind others)
        A.CoarseDropout(
            max_holes=3,
            max_height=0.1,
            max_width=0.1,
            min_holes=1,
            min_height=0.03,
            min_width=0.03,
            fill_value=0,
            p=AUGMENTATION_CONFIG["cutout_probability"]
        ),
        
        # Shadow simulation (common in real photos)
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_image(image: np.ndarray, bboxes: list, labels: list, transform) -> tuple:
    """Apply augmentation to a single image with bounding boxes."""
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    except Exception as e:
        print(f"   Warning: Augmentation failed: {e}")
        return image, bboxes, labels


def augment_dataset(input_dir: Path, output_dir: Path, augmentations_per_image: int = 3):
    """Augment all images in a dataset."""
    print("\n" + "=" * 60)
    print("🔄 AUGMENTING DATASET")
    print("=" * 60)
    
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    
    if not images_dir.exists():
        # Try to find images folder
        for subdir in ["train", "valid", "test"]:
            if (input_dir / subdir / "images").exists():
                images_dir = input_dir / subdir / "images"
                labels_dir = input_dir / subdir / "labels"
                break
    
    if not images_dir.exists():
        print(f"❌ Cannot find images directory in {input_dir}")
        return
    
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    transform = create_augmentation_pipeline()
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"   Found {len(image_files)} images")
    print(f"   Creating {augmentations_per_image} augmentations per image")
    
    for img_path in tqdm(image_files, desc="Augmenting"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels (YOLO format)
        label_path = labels_dir / (img_path.stem + ".txt")
        bboxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])
        
        # Copy original
        cv2.imwrite(str(output_images / img_path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_path.exists():
            shutil.copy(label_path, output_labels / label_path.name)
        
        # Generate augmentations
        for i in range(augmentations_per_image):
            aug_image, aug_bboxes, aug_labels = augment_image(image, bboxes, labels, transform)
            
            # Save augmented image
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(output_images / aug_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            # Save augmented labels
            if aug_bboxes:
                with open(output_labels / f"{img_path.stem}_aug{i}.txt", 'w') as f:
                    for label, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{label} {' '.join(map(str, bbox))}\n")
    
    total_images = len(image_files) * (augmentations_per_image + 1)
    print(f"\n✅ Augmentation complete!")
    print(f"   Original: {len(image_files)} images")
    print(f"   Total: {total_images} images")
    print(f"   Output: {output_dir}")


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_training_dataset():
    """Prepare final dataset for YOLO training."""
    print("\n" + "=" * 60)
    print("📋 PREPARING TRAINING DATASET")
    print("=" * 60)
    
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    train_dir = FINAL_DIR / "train"
    val_dir = FINAL_DIR / "valid"
    
    for d in [train_dir, val_dir]:
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
    
    # Collect all images from augmented directory
    all_images = []
    if AUGMENTED_DIR.exists():
        all_images.extend(list((AUGMENTED_DIR / "images").glob("*.jpg")))
        all_images.extend(list((AUGMENTED_DIR / "images").glob("*.png")))
    
    print(f"   Found {len(all_images)} total images")
    
    # 80/20 split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"   Train: {len(train_images)} images")
    print(f"   Valid: {len(val_images)} images")
    
    # Copy files
    for img_path in tqdm(train_images, desc="Copying train"):
        shutil.copy(img_path, train_dir / "images" / img_path.name)
        label_path = AUGMENTED_DIR / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, train_dir / "labels" / label_path.name)
    
    for img_path in tqdm(val_images, desc="Copying valid"):
        shutil.copy(img_path, val_dir / "images" / img_path.name)
        label_path = AUGMENTED_DIR / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, val_dir / "labels" / label_path.name)
    
    # Create data.yaml for YOLO training
    data_yaml = f"""# Chess Pieces Dataset
path: {FINAL_DIR}
train: train/images
val: valid/images

# Classes (12 chess piece types)
names:
  0: white-king
  1: white-queen
  2: white-rook
  3: white-bishop
  4: white-knight
  5: white-pawn
  6: black-king
  7: black-queen
  8: black-rook
  9: black-bishop
  10: black-knight
  11: black-pawn

nc: 12
"""
    
    with open(FINAL_DIR / "data.yaml", 'w') as f:
        f.write(data_yaml)
    
    print(f"\n✅ Dataset prepared at {FINAL_DIR}")
    print(f"   Config: {FINAL_DIR / 'data.yaml'}")
    
    return FINAL_DIR / "data.yaml"


def print_training_instructions(data_yaml_path: Path):
    """Print instructions for training YOLO."""
    print("\n" + "=" * 60)
    print("🚀 TRAINING INSTRUCTIONS")
    print("=" * 60)
    print("""
To train the improved model, run:

    # Using Ultralytics YOLO
    yolo detect train \\
        data={data_yaml} \\
        model=yolov8m.pt \\
        epochs=100 \\
        imgsz=640 \\
        batch=16 \\
        name=chess_pieces_v2

    # Or using Python:
    from ultralytics import YOLO
    model = YOLO('yolov8m.pt')
    model.train(
        data='{data_yaml}',
        epochs=100,
        imgsz=640,
        batch=16,
        name='chess_pieces_v2'
    )

After training, the model will be saved to:
    runs/detect/chess_pieces_v2/weights/best.pt

Convert to ONNX for deployment:
    yolo export model=runs/detect/chess_pieces_v2/weights/best.pt format=onnx
""".format(data_yaml=data_yaml_path))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Chess Dataset Collection & Augmentation")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--augment", action="store_true", help="Apply augmentations")
    parser.add_argument("--prepare", action="store_true", help="Prepare training dataset")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--augment-count", type=int, default=3, 
                       help="Number of augmentations per image")
    
    args = parser.parse_args()
    
    if not any([args.download, args.augment, args.prepare, args.all]):
        parser.print_help()
        return
    
    if args.download or args.all:
        download_all(args.api_key)
    
    if args.augment or args.all:
        # Find and augment each raw dataset
        if RAW_DIR.exists():
            for dataset_dir in RAW_DIR.iterdir():
                if dataset_dir.is_dir():
                    augment_dataset(
                        dataset_dir, 
                        AUGMENTED_DIR / dataset_dir.name,
                        args.augment_count
                    )
    
    if args.prepare or args.all:
        data_yaml = prepare_training_dataset()
        print_training_instructions(data_yaml)


if __name__ == "__main__":
    main()
