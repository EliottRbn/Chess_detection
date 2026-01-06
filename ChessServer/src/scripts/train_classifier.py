"""
Train Secondary Classifier

Trains the MobileNetV3 classifier on the chess dataset.
Used to verify/correct low-confidence YOLO detections.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from piece_classifier import PieceClassifier, PIECE_CLASSES

# Configuration
DATASET_DIR = Path("../datasets/chess_augmented")
MODEL_SAVE_PATH = "../models/piece_classifier.pt"
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class ChessClassificationDataset(Dataset):
    """
    Creates a classification dataset from YOLO formatted data.
    Crops pieces from images based on YOLO labels.
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        self.transform = transform
        self.samples = []
        
        print(f"Loading {split} dataset...")
        
        # Scan all labels
        label_files = list(self.labels_dir.glob('*.txt'))
        
        for lbl_path in tqdm(label_files):
            img_path = self.images_dir / (lbl_path.stem + '.jpg')
            if not img_path.exists():
                continue
                
            # Read labels
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # Store image path and bbox instead of pre-cropping (save memory)
                    self.samples.append({
                        'img_path': str(img_path),
                        'cls_id': cls_id,
                        'bbox': [float(x) for x in parts[1:5]] # cx, cy, w, h
                    })
                    
        print(f"Found {len(self.samples)} pieces in {len(label_files)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load full image
        img = cv2.imread(sample['img_path'])
        if img is None:
            # Fallback for bad image
            return torch.zeros((3, 96, 96)), sample['cls_id']
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Crop piece
        cx, cy, bw, bh = sample['bbox']
        
        # Convert relative YOLO format to absolute pixel coords
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Add a little padding (10%)
        pad_x = int(bw * w * 0.1)
        pad_y = int(bh * h * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        crop = img[y1:y2, x1:x2]
        
        # If crop is empty or invalid, return zeros
        if crop.size == 0:
            return torch.zeros((3, 96, 96)), sample['cls_id']
            
        # Convert to PIL for transforms
        pil_img = Image.fromarray(crop)
        
        if self.transform:
            return self.transform(pil_img), sample['cls_id']
            
        return pil_img, sample['cls_id']


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = ChessClassificationDataset(DATASET_DIR, 'train', train_transform)
    val_dataset = ChessClassificationDataset(DATASET_DIR, 'valid', val_transform)
    
    # Create smaller subset if too big to speed up demo
    # if len(train_dataset) > 10000:
    #     indices = torch.randperm(len(train_dataset))[:10000]
    #     train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    classifier = PieceClassifier(device=device)
    model = classifier.model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"   Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   💾 Model saved!")
            
    print(f"\n✅ Training complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"   Saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # Create models dir if not exists
    os.makedirs("../models", exist_ok=True)
    try:
        train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
