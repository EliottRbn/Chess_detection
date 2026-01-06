#!/usr/bin/env python3
"""
Quick script to reorganize and augment the chess dataset.
Fixes the bad train/valid/test split and applies augmentations.
"""

import os
import shutil
import random
from pathlib import Path

try:
    import cv2
    import numpy as np
    import albumentations as A
    from tqdm import tqdm
except ImportError:
    print("Installing dependencies...")
    os.system("pip install opencv-python albumentations tqdm --quiet")
    import cv2
    import numpy as np
    import albumentations as A
    from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "Chess Pieces.v18i.yolov8"
OUTPUT_DIR = BASE_DIR / "datasets" / "chess_augmented"

# Augmentation pipeline for chess
transform = A.Compose([
    A.OneOf([
        A.Rotate(limit=20, p=0.7),
        A.Perspective(scale=(0.02, 0.08), p=0.5),
        A.Affine(scale=(0.85, 1.15), rotate=(-15, 15), shear=(-8, 8), p=0.6),
    ], p=0.8),
    
    A.HorizontalFlip(p=0.5),
    
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ], p=0.7),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10, 40), p=0.15),
    ], p=0.25),
    
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.15),
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def collect_all_images():
    """Collect all images from train/valid/test into one list."""
    all_images = []
    
    for split in ['train', 'valid', 'test']:
        img_dir = DATASET_DIR / split / 'images'
        lbl_dir = DATASET_DIR / split / 'labels'
        
        if not img_dir.exists():
            continue
            
        for img_path in img_dir.glob('*.jpg'):
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                all_images.append((img_path, lbl_path))
    
    return all_images


def load_yolo_labels(label_path):
    """Load YOLO format labels."""
    bboxes = []
    labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
    
    return bboxes, labels


def save_yolo_labels(label_path, bboxes, labels):
    """Save YOLO format labels."""
    with open(label_path, 'w') as f:
        for label, bbox in zip(labels, bboxes):
            f.write(f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n")


def augment_image(image, bboxes, labels, num_augmentations=3):
    """Generate augmented versions of an image."""
    results = []
    
    for i in range(num_augmentations):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
            if transformed['bboxes']:  # Only keep if bboxes survived
                results.append((
                    transformed['image'],
                    transformed['bboxes'],
                    transformed['class_labels']
                ))
        except Exception as e:
            pass  # Skip failed augmentations
    
    return results


def main():
    print("=" * 60)
    print("🔄 CHESS DATASET REORGANIZATION & AUGMENTATION")
    print("=" * 60)
    
    # Collect all images
    print("\n📁 Collecting images from all splits...")
    all_images = collect_all_images()
    print(f"   Found {len(all_images)} images total")
    
    if not all_images:
        print("❌ No images found!")
        return
    
    # Create output directories
    train_img_dir = OUTPUT_DIR / 'train' / 'images'
    train_lbl_dir = OUTPUT_DIR / 'train' / 'labels'
    valid_img_dir = OUTPUT_DIR / 'valid' / 'images'
    valid_lbl_dir = OUTPUT_DIR / 'valid' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split 85/15
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.85)
    train_images = all_images[:split_idx]
    valid_images = all_images[split_idx:]
    
    print(f"\n📊 New split: {len(train_images)} train / {len(valid_images)} valid")
    
    # Process training images with augmentation
    print("\n🔄 Augmenting training images (4x)...")
    train_count = 0
    aug_per_image = 3  # Original + 3 augmented = 4x
    
    for img_path, lbl_path in tqdm(train_images, desc="Training"):
        # Load image and labels
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = load_yolo_labels(lbl_path)
        
        if not bboxes:
            continue
        
        # Save original
        out_img = train_img_dir / f"{img_path.stem}_orig.jpg"
        out_lbl = train_lbl_dir / f"{img_path.stem}_orig.txt"
        cv2.imwrite(str(out_img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        save_yolo_labels(out_lbl, bboxes, labels)
        train_count += 1
        
        # Generate augmentations
        augmented = augment_image(image, bboxes, labels, aug_per_image)
        for i, (aug_img, aug_bboxes, aug_labels) in enumerate(augmented):
            out_img = train_img_dir / f"{img_path.stem}_aug{i}.jpg"
            out_lbl = train_lbl_dir / f"{img_path.stem}_aug{i}.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            save_yolo_labels(out_lbl, aug_bboxes, aug_labels)
            train_count += 1
    
    # Copy validation images (no augmentation)
    print("\n📋 Copying validation images...")
    valid_count = 0
    for img_path, lbl_path in tqdm(valid_images, desc="Validation"):
        shutil.copy(img_path, valid_img_dir / img_path.name)
        shutil.copy(lbl_path, valid_lbl_dir / lbl_path.name)
        valid_count += 1
    
    # Create data.yaml
    data_yaml = f"""# Chess Pieces Augmented Dataset
path: {OUTPUT_DIR}
train: train/images
val: valid/images

nc: 12
names:
  0: black-bishop
  1: black-king
  2: black-knight
  3: black-pawn
  4: black-queen
  5: black-rook
  6: white-bishop
  7: white-king
  8: white-knight
  9: white-pawn
  10: white-queen
  11: white-rook
"""
    
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        f.write(data_yaml)
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY!")
    print("=" * 60)
    print(f"\n📊 Final counts:")
    print(f"   Training: {train_count} images (with augmentations)")
    print(f"   Validation: {valid_count} images")
    print(f"   Total: {train_count + valid_count} images")
    print(f"\n📁 Output: {OUTPUT_DIR}")
    print(f"📄 Config: {OUTPUT_DIR / 'data.yaml'}")
    
    print("\n🚀 To train the model, run:")
    print(f"   python train_model.py --data {OUTPUT_DIR / 'data.yaml'} --epochs 50")


if __name__ == "__main__":
    main()
