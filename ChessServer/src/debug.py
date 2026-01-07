import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import torch

# --- configuration ---
INPUT_SIZE = 640

# Helper functions copied/adapted from server_fen.py to avoid side effects on import

def preprocess(image_path):
    """Load and preprocess image from path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    return img_resized, img # Return both resized (RGB) and original (BGR)

def order_corners(points):
    points = np.array(points, dtype=np.float32)
    sorted_by_y = points[np.argsort(points[:, 1])]
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def get_4_corners_from_polygon(polygon_points):
    points = np.array(polygon_points, dtype=np.float32)
    if len(points) == 4: return order_corners(points)
    if len(points) < 4: return None
    hull = cv2.convexHull(points)
    hull_points = hull.reshape(-1, 2)
    if len(hull_points) < 4: return None
    if len(hull_points) == 4: return order_corners(hull_points)
    # Simple bounding box approximation for now if complex hull
    rect = cv2.minAreaRect(hull_points)
    box = cv2.boxPoints(rect)
    return order_corners(box) 

def extract_board_polygon(board_results):
    masks = board_results[0].masks
    if masks is None or masks.data is None: return None
    merged_mask = torch.any(masks.data, dim=0).int()
    merged = merged_mask.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon_points = approx.reshape(-1, 2).tolist()
    return get_4_corners_from_polygon(polygon_points)

# --- Main Debug Logic ---

def debug_inference(image_path, piece_model_path, board_model_path=None):
    print(f"Loading piece model: {piece_model_path}")
    model = YOLO(piece_model_path)
    
    board_model = None
    if board_model_path and os.path.exists(board_model_path):
        print(f"Loading board model: {board_model_path}")
        board_model = YOLO(board_model_path)
    else:
        print("No board model provided or found, skipping board detection.")

    print(f"Processing image: {image_path}")
    img_rgb, img_original_bgr = preprocess(image_path)
    
    # We work on the resized RGB image for inference
    debug_img = img_rgb.copy()

    # Inference
    results = model.predict(img_rgb, verbose=False, conf=0.25)
    
    # Annotate Pieces
    print("\n--- Detections ---")
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names[cls_id]
            print(f"Piece: {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
            
            # Draw on debug image
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(debug_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Annotate Board if available
    if board_model:
        board_results = board_model.predict(img_rgb, verbose=False)
        board_corners = extract_board_polygon(board_results)
        if board_corners is not None:
            print(f"Board detected: {board_corners.tolist()}")
            pts = board_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, (0, 0, 255), 3)
            for i, corner in enumerate(board_corners):
                pt = (int(corner[0]), int(corner[1]))
                cv2.circle(debug_img, pt, 5, (255, 0, 0), -1)
        else:
            print("Board model loaded but no board detected.")

    # Save output
    output_path = "debug_output.jpg"
    # Convert back to BGR for saving
    debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, debug_img_bgr)
    print(f"\nSaved annotated image to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug YOLOv11 Chess Model")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="runs/best.pt", help="Path to piece detection model (default: runs/best.pt)")
    parser.add_argument("--board_model", type=str, default="chessboard_detection.pt", help="Path to board detection model")

    args = parser.parse_args()
    
    # Handle relative paths, check src/runs if default not found
    model_path = args.model
    if not os.path.exists(model_path):
        # Try src/runs/best.pt as fallback if running from src
        alt_path = os.path.join("runs", "best.pt")
        if os.path.exists(alt_path):
             model_path = alt_path
        else:
             # Try absolute path based on what we found in file listing earlier
             # The user has files in /Users/rapha/Documents/A5/Chess/ChessServer/src/runs/best.pt
             # If we are in src/, then it is runs/best.pt
             pass

    debug_inference(args.image_path, model_path, args.board_model)
