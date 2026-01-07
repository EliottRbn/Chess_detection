"""
Ensemble Model Detection for Chess Pieces

Uses multiple YOLO models and combines their predictions for higher accuracy.
Strategies:
1. Weighted Box Fusion (WBF) - merge overlapping predictions
2. Confidence boost when both models agree
3. Use disagreements to flag uncertain pieces
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class EnsembleDetector:
    """
    Ensemble detector that combines predictions from multiple YOLO models.
    """
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble with multiple models.
        
        Args:
            model_paths: List of paths to YOLO models (.pt or .onnx)
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = [YOLO(path) for path in model_paths]
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
        self.model_names = [path.split('/')[-1] for path in model_paths]
        
        print(f"[Ensemble] Loaded {len(self.models)} models:")
        for name, weight in zip(self.model_names, self.weights):
            print(f"  - {name} (weight: {weight:.2f})")
    
    def predict(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        """
        Run ensemble prediction on an image.
        
        Returns combined predictions with boosted confidence for agreements.
        """
        all_predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            results = model.predict(image, verbose=False, conf=conf_threshold)
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    all_predictions.append({
                        'model_idx': i,
                        'class': cls_id,
                        'class_name': model.names[cls_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'weight': self.weights[i]
                    })
        
        # Merge predictions using Weighted Box Fusion
        merged = self._weighted_box_fusion(all_predictions, iou_threshold)
        
        return merged
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _weighted_box_fusion(self, predictions: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Merge overlapping predictions using Weighted Box Fusion.
        
        - Boxes with IoU > threshold are merged
        - Confidence is boosted when multiple models agree
        - Returns merged boxes with combined confidence
        """
        if not predictions:
            return []
        
        # Group by class
        class_groups = {}
        for pred in predictions:
            cls = pred['class_name']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(pred)
        
        merged_results = []
        
        for cls_name, preds in class_groups.items():
            # Sort by confidence
            preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
            used = [False] * len(preds)
            
            for i, pred in enumerate(preds):
                if used[i]:
                    continue
                
                # Find overlapping predictions
                matched = [pred]
                matched_models = {pred['model_idx']}
                used[i] = True
                
                for j in range(i + 1, len(preds)):
                    if used[j]:
                        continue
                    
                    if self._iou(pred['bbox'], preds[j]['bbox']) > iou_threshold:
                        matched.append(preds[j])
                        matched_models.add(preds[j]['model_idx'])
                        used[j] = True
                
                # Merge matched predictions
                merged_box = self._merge_boxes(matched)
                
                # Boost confidence if multiple models agree
                n_models_agree = len(matched_models)
                agreement_boost = 1.0 + (n_models_agree - 1) * 0.15  # +15% per extra model
                merged_box['confidence'] = min(0.99, merged_box['confidence'] * agreement_boost)
                merged_box['models_agree'] = n_models_agree
                merged_box['agreement_note'] = 'high' if n_models_agree > 1 else 'single'
                
                merged_results.append(merged_box)
        
        return merged_results
    
    def _merge_boxes(self, predictions: List[Dict]) -> Dict:
        """Merge multiple predictions into one using weighted average."""
        total_weight = sum(p['confidence'] * p['weight'] for p in predictions)
        
        # Weighted average of boxes
        merged_bbox = [0, 0, 0, 0]
        for p in predictions:
            w = p['confidence'] * p['weight'] / total_weight
            for i in range(4):
                merged_bbox[i] += p['bbox'][i] * w
        
        # Weighted average confidence
        merged_conf = sum(p['confidence'] * p['weight'] for p in predictions) / sum(p['weight'] for p in predictions)
        
        return {
            'class': predictions[0]['class'],
            'class_name': predictions[0]['class_name'],
            'confidence': merged_conf,
            'bbox': merged_bbox,
            'n_sources': len(predictions)
        }


def create_ensemble_detector(old_model_path: str, new_model_path: str, 
                             old_weight: float = 0.4, new_weight: float = 0.6):
    """
    Create ensemble detector with old and new models.
    
    Default: new model weighted slightly higher (trained on more diverse data).
    """
    return EnsembleDetector(
        model_paths=[old_model_path, new_model_path],
        weights=[old_weight, new_weight]
    )


class ConfirmationEnsembleDetector:
    """
    Confirmation-based ensemble where primary model (best.pt) is the source of truth
    and secondary model (ONNX) only confirms/corrects color disagreements.
    """
    
    def __init__(self, primary_model_path: str, confirmation_model_path: str,
                 primary_weight: float = 0.8, confirmation_weight: float = 0.2):
        """
        Args:
            primary_model_path: Path to primary model (best.pt - YOLOv11)
            confirmation_model_path: Path to confirmation model (ONNX - legacy)
            primary_weight: Weight for primary model (default 0.8)
            confirmation_weight: Weight for confirmation (default 0.2)
        """
        self.primary_model = YOLO(primary_model_path)
        self.confirmation_model = YOLO(confirmation_model_path)
        self.primary_weight = primary_weight
        self.confirmation_weight = confirmation_weight
        
        print(f"[ConfirmationEnsemble] Primary: {primary_model_path.split('/')[-1]} (weight: {primary_weight})")
        print(f"[ConfirmationEnsemble] Confirmation: {confirmation_model_path.split('/')[-1]} (weight: {confirmation_weight})")
    
    def _get_piece_type(self, class_name: str) -> str:
        """Extract piece type without color (e.g., 'white-pawn' -> 'pawn')"""
        return class_name.split('-')[-1] if '-' in class_name else class_name
    
    def _get_piece_color(self, class_name: str) -> str:
        """Extract color from class name (e.g., 'white-pawn' -> 'white')"""
        return class_name.split('-')[0] if '-' in class_name else 'unknown'
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict(self, image, conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        """
        Run confirmation-based ensemble prediction.
        
        1. Get predictions from primary model (best.pt)
        2. Get predictions from confirmation model (ONNX)
        3. For each primary prediction:
           - If confirmation agrees on piece type AND color: boost confidence
           - If confirmation agrees on type but different color: use higher weighted confidence
           - If no confirmation match: keep primary prediction as-is
        """
        # Primary predictions
        primary_preds = []
        primary_results = self.primary_model.predict(image, verbose=False, conf=conf_threshold)
        for result in primary_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                primary_preds.append({
                    'class': cls_id,
                    'class_name': self.primary_model.names[cls_id],
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'source': 'primary'
                })
        
        # Confirmation predictions
        confirm_preds = []
        confirm_results = self.confirmation_model.predict(image, verbose=False, conf=conf_threshold)
        for result in confirm_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                confirm_preds.append({
                    'class': cls_id,
                    'class_name': self.confirmation_model.names[cls_id],
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'source': 'confirmation'
                })
        
        # Match and confirm
        final_preds = []
        for pred in primary_preds:
            best_match = None
            best_iou = 0
            
            # Find best matching confirmation prediction
            for cpred in confirm_preds:
                iou = self._iou(pred['bbox'], cpred['bbox'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = cpred
            
            if best_match:
                primary_type = self._get_piece_type(pred['class_name'])
                confirm_type = self._get_piece_type(best_match['class_name'])
                primary_color = self._get_piece_color(pred['class_name'])
                confirm_color = self._get_piece_color(best_match['class_name'])
                
                if primary_type == confirm_type:
                    if primary_color == confirm_color:
                        # Full agreement - boost confidence
                        pred['confidence'] = min(0.99, pred['confidence'] * 1.15)
                        pred['confirmation'] = 'agreed'
                        print(f"[Confirm] ✓ {pred['class_name']} confirmed (conf: {pred['confidence']:.2f})")
                    else:
                        # Same piece type, different color - weighted vote
                        primary_score = pred['confidence'] * self.primary_weight
                        confirm_score = best_match['confidence'] * self.confirmation_weight
                        
                        if primary_score >= confirm_score:
                            # Keep primary color
                            pred['confirmation'] = 'primary_color_kept'
                            print(f"[Confirm] → {pred['class_name']} kept (primary: {primary_score:.2f} >= confirm: {confirm_score:.2f})")
                        else:
                            # Use confirmation color
                            old_class = pred['class_name']
                            pred['class_name'] = best_match['class_name']
                            pred['class'] = best_match['class']
                            pred['confirmation'] = 'color_corrected'
                            print(f"[Confirm] ⚡ {old_class} → {pred['class_name']} (confirm won: {confirm_score:.2f} > {primary_score:.2f})")
                else:
                    # Different piece types - trust primary
                    pred['confirmation'] = 'type_mismatch'
            else:
                pred['confirmation'] = 'no_match'
            
            final_preds.append(pred)
        
        return final_preds

