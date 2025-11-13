
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import chess
import chess.svg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            board_polygon =  item['polygon']
        else:
            x1, y1, x2, y2 = item['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y, CLASSES.get(item['class'])))
    
    # Calculer le placement des pièces sur l'échiquier
    if len(board_polygon) == 4:
        board_polygon = np.array(board_polygon, dtype=np.float32)
        src_pts = np.array(centers)[:, :2].astype(np.float32)
        dst_pts = np.array([[0, 0], [INPUT_SIZE, 0], [INPUT_SIZE, INPUT_SIZE], [0, INPUT_SIZE]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(board_polygon, dst_pts)
        fen_positions = [['1' for _ in range(8)] for _ in range(8)]

        for center in centers:
            pt = np.array([[center[0], center[1]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(np.array([pt]), M)[0][0]
            x, y = int(warped_pt[0] // (INPUT_SIZE / 8)), int(warped_pt[1] // (INPUT_SIZE / 8))
            if 0 <= x < 8 and 0 <= y < 8:
                fen_positions[y][x] = center[2]

        fen_rows = []
        for row in fen_positions:
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

@app.post("/detect")
async def detect(file: UploadFile):
    try:
        img_bytes = await file.read()
        img = preprocess(img_bytes)

        # Debug: sauvegarde image prétraitée
        # cv2.imwrite("debug_preprocessed_image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Inférence YOLO pour objets
        results = model.predict(img)
        # Inférence YOLO pour plateau
        board_results = board_model.predict(img)

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
                        # YOLO ne donne pas directement le contour, mais on peut approximer par rectangle
                        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    detections.append({
                        "class": CLASSES[cls_id],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "polygon": polygon
                    })

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
                    "bbox": [],  # On peut ignorer bbox si on a polygon
                    "polygon": polygon_points
                })

        matrix = detections
        # Placeholder for FEN extraction logic 
        fen = FEN_extract(matrix)
        print("FEN Matrix:", fen)
        affiche_fen(fen)
        return {"detections": detections}

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)