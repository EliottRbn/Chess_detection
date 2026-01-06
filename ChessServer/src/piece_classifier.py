"""
Secondary Piece Classifier

A lightweight CNN classifier (MobileNetV3-Small) to confirm/correct
pieces detected with low confidence by YOLO.

Use case:
- YOLO detects a piece with confidence < 0.7
- We crop the piece from the image
- This classifier confirms or corrects the piece type

Training: Uses the same dataset as YOLO but reformatted for classification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Chess piece classes (same order as YOLO)
PIECE_CLASSES = [
    'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
    'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

# Piece similarity groups (pieces that look alike)
SIMILAR_PIECES = {
    'queen': ['king', 'rook', 'bishop'],
    'king': ['queen'],
    'rook': ['queen'],
    'bishop': ['queen', 'pawn'],
    'knight': [],  # Knights are distinctive
    'pawn': ['bishop'],
}


class PieceClassifier:
    """
    Lightweight classifier to confirm/correct uncertain YOLO detections.
    Uses MobileNetV3-Small for fast inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available() 
            else 'mps' if device == 'auto' and torch.backends.mps.is_available()
            else 'cpu'
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Small input for speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._create_model()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Classifier] Loaded model from {model_path}")
        else:
            print(f"[Classifier] Initialized with pretrained MobileNetV3 (no fine-tuning)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _create_model(self) -> nn.Module:
        """Create MobileNetV3-Small with custom head for chess pieces."""
        # Use pretrained MobileNetV3-Small
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Replace classifier head
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(PIECE_CLASSES))
        )
        
        return model
    
    def classify(self, image: np.ndarray, top_k: int = 3) -> list:
        """
        Classify a cropped piece image.
        
        Args:
            image: Cropped piece image (numpy array, RGB)
            top_k: Return top K predictions
            
        Returns:
            List of (class_name, confidence) tuples
        """
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform and predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results.append((PIECE_CLASSES[idx], float(prob)))
        
        return results
    
    def verify_detection(self, image: np.ndarray, yolo_class: str, yolo_confidence: float) -> dict:
        """
        Verify or correct a YOLO detection.
        
        Args:
            image: Cropped piece image
            yolo_class: Class predicted by YOLO
            yolo_confidence: YOLO's confidence score
            
        Returns:
            dict with verification result
        """
        # Get classifier predictions
        predictions = self.classify(image, top_k=3)
        classifier_class, classifier_conf = predictions[0]
        
        # Determine if YOLO and classifier agree
        yolo_piece_type = yolo_class.split('-')[1] if '-' in yolo_class else yolo_class
        classifier_piece_type = classifier_class.split('-')[1] if '-' in classifier_class else classifier_class
        
        # Check color match
        yolo_color = yolo_class.split('-')[0] if '-' in yolo_class else 'unknown'
        classifier_color = classifier_class.split('-')[0] if '-' in classifier_class else 'unknown'
        
        color_match = yolo_color == classifier_color
        type_match = yolo_piece_type == classifier_piece_type
        full_match = yolo_class == classifier_class
        
        # Calculate combined confidence
        if full_match:
            # Both agree - boost confidence
            combined_confidence = min(0.99, (yolo_confidence + classifier_conf) / 2 * 1.2)
            decision = 'confirmed'
            final_class = yolo_class
        elif type_match and not color_match:
            # Same piece type, different color - trust classifier for color
            combined_confidence = classifier_conf * 0.9
            decision = 'color_corrected'
            final_class = classifier_class
        elif classifier_conf > yolo_confidence + 0.2:
            # Classifier is much more confident - use classifier
            combined_confidence = classifier_conf
            decision = 'corrected'
            final_class = classifier_class
        else:
            # Disagreement but similar confidence - keep YOLO, flag uncertainty
            combined_confidence = yolo_confidence * 0.85
            decision = 'uncertain'
            final_class = yolo_class
        
        return {
            'final_class': final_class,
            'confidence': combined_confidence,
            'decision': decision,
            'yolo': {'class': yolo_class, 'confidence': yolo_confidence},
            'classifier': {'class': classifier_class, 'confidence': classifier_conf},
            'alternatives': predictions[1:],
            'details': {
                'color_match': color_match,
                'type_match': type_match,
                'full_match': full_match
            }
        }


def crop_piece_from_image(image: np.ndarray, bbox: list, padding: float = 0.1) -> np.ndarray:
    """
    Crop a piece from the image with some padding.
    
    Args:
        image: Full image
        bbox: [x1, y1, x2, y2] bounding box
        padding: Fraction of box size to add as padding
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)
    
    # Apply padding with bounds checking
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return image[y1:y2, x1:x2].copy()


class PieceVerifier:
    """
    High-level interface to verify YOLO detections using the classifier.
    """
    
    def __init__(self, classifier_model_path: Optional[str] = None, 
                 confidence_threshold: float = 0.7):
        """
        Args:
            classifier_model_path: Path to trained classifier weights
            confidence_threshold: Verify pieces below this YOLO confidence
        """
        self.classifier = PieceClassifier(classifier_model_path)
        self.confidence_threshold = confidence_threshold
        
        print(f"[Verifier] Will verify detections with confidence < {confidence_threshold}")
    
    def verify_detections(self, image: np.ndarray, detections: list) -> list:
        """
        Verify a list of YOLO detections.
        
        Args:
            image: Full image (RGB numpy array)
            detections: List of dicts with 'class', 'confidence', 'bbox'
            
        Returns:
            Updated detections with verification results
        """
        verified = []
        
        for det in detections:
            if det['confidence'] < self.confidence_threshold:
                # Low confidence - verify with classifier
                piece_crop = crop_piece_from_image(image, det['bbox'])
                
                if piece_crop.size > 0:
                    result = self.classifier.verify_detection(
                        piece_crop, 
                        det['class'], 
                        det['confidence']
                    )
                    
                    # Update detection
                    verified_det = det.copy()
                    verified_det['class'] = result['final_class']
                    verified_det['confidence'] = result['confidence']
                    verified_det['verification'] = result
                    verified.append(verified_det)
                else:
                    verified.append(det)
            else:
                # High confidence - keep as is
                verified.append(det)
        
        return verified
