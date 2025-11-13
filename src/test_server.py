import numpy as np
import cv2
from ultralytics import YOLO
import torch
import chess

import chess.svg

MODEL_PATH = "src/runs/yolo8-20epoch.pt"
model = YOLO(MODEL_PATH)
board_model = YOLO("src/runs/chessboard_detection.pt")
INPUT_SIZE = 640

CLASSES = model.names

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    return img

def affiche_fen(fen):
    chess.Board(fen)
    board = chess.Board(fen)
    svg_board = chess.svg.board(board=board)    
    with open("board.svg", "w") as f:
        f.write(svg_board)

def FEN_extract(matrix):
    CLASSES = {
        'white-pawn': 'P', 'white-rook': 'R', 
        'white-knight': 'N', 'white-bishop': 'B',
        'white-queen': 'Q', 'white-king': 'K',
        
        'black-pawn': 'p', 'black-rook': 'r', 
        'black-knight': 'n', 'black-bishop': 'b',
        'black-queen': 'q', 'black-king': 'k'
    }
    centers = []
    board_polygon = []
    for item in matrix:
        if item['class'] == "chessboard":
            board_polygon = item['polygon']
        else:
            x1, y1, x2, y2 = item['bbox']
            center_x = (x1 + x2) / 2
            bottom_y = y2 -10  # Adjust to 10px below the bottom of the piece
            centers.append((center_x, bottom_y, CLASSES.get(item['class'])))

    if len(board_polygon) == 4:
        board_polygon = np.array(board_polygon, dtype=np.float32)
        dst_pts = np.array([[0, 0], [INPUT_SIZE, 0], [INPUT_SIZE, INPUT_SIZE], [0, INPUT_SIZE]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(board_polygon, dst_pts)

        fen_positions = [['1' for _ in range(8)] for _ in range(8)]

        for center in centers:
            pt = np.array([[center[0], center[1]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(np.array([pt]), M)[0][0]
            x, y = int(warped_pt[0] // (INPUT_SIZE / 8)), int(warped_pt[1] // (INPUT_SIZE / 8))
            if 0 <= x < 8 and 0 <= y < 8:
                # Check for overlapping pieces and prioritize based on confidence
                if fen_positions[y][x] == '1' or center[2].isupper():  # Prioritize white pieces over black
                    fen_positions[y][x] = center[2]

        fen_rows = []
        for row in reversed(fen_positions):  # Reverse the rows to correct the FEN orientation
            fen_row = ''
            empty_count = 0
            for cell in row:
                if cell == '1':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen_string = '/'.join(fen_rows)
        return fen_string
    return "8/8/8/8/8/8/8/8"

def detect(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img = preprocess(img_bytes)

        # Inférence YOLO pour objets
        results = model.predict(img, imgsz=INPUT_SIZE)
        # Inférence YOLO pour plateau
        board_results = board_model.predict(img, imgsz=INPUT_SIZE)

        detections = []

        # Détection des objets avec polygones (si disponible)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    polygon = []
                    if box.xywh is not None:
                        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    detections.append({
                        "class": CLASSES[cls_id],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "polygon": polygon
                    })
                    # Draw bounding boxes on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    cv2.putText(img, CLASSES[cls_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Détection du plateau via masque
        masks = board_results[0].masks.data if board_results[0].masks is not None else None
        if masks is not None:
            merged_mask = torch.any(masks, dim=0).int()
            merged = merged_mask.cpu().numpy()
            contours, _ = cv2.findContours(merged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                polygon_points = approx.reshape(-1, 2).tolist()
                detections.append({
                    "class": "chessboard",
                    "confidence": 1.0,
                    "bbox": [],
                    "polygon": polygon_points
                })
                # Draw the chessboard polygon on the image
                cv2.polylines(img, [np.array(polygon_points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

        matrix = detections
        fen = FEN_extract(matrix)
        print("FEN Matrix:", fen)
        affiche_fen(fen)

        # Save the image with detections
        output_path = "output_board.jpg"
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(output_path, img_bgr)
        print(f"Results saved to {output_path}")

        return {"detections": detections}

    except Exception as e:
        print(f"Error processing image: {e}")


# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "2.jpg"  # Remplacez par le chemin de votre image
    detect(image_path)