import os
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths - configurable via environment for Docker deployment
# Priority: env var > models/ (Docker) > src/runs/ (local) > runs/ (local fallback)
MODEL_PATH = os.getenv("PIECE_MODEL_PATH")
MODEL_PATH_2 = os.getenv("PIECE_MODEL_PATH_2")  # Optional second model for ensemble
BOARD_MODEL_PATH = os.getenv("BOARD_MODEL_PATH")

# Find primary piece detection model
if not MODEL_PATH:
    for path in ["models/detect_pieces_1.onnx", "src/runs/detect_pieces_1.onnx", "runs/detect_pieces_1.onnx"]:
        if os.path.exists(path):
            MODEL_PATH = path
            break

# Find secondary piece detection model (for ensemble)
if not MODEL_PATH_2:
    for path in ["models/detect_pieces_v2.onnx", "models/detect_pieces_v2.pt", 
                 "runs/detect/chess_v2/weights/best.pt", "runs/detect/chess_v2/weights/best.onnx"]:
        if os.path.exists(path):
            MODEL_PATH_2 = path
            break

if not BOARD_MODEL_PATH:
    for path in ["models/chessboard_detection.pt", "src/runs/chessboard_detection.pt", "runs/chessboard_detection.pt"]:
        if os.path.exists(path):
            BOARD_MODEL_PATH = path
            break

# Load models
print(f"Loading piece model from: {MODEL_PATH}")
if MODEL_PATH_2:
    print(f"Loading secondary model from: {MODEL_PATH_2}")
print(f"Loading board model from: {BOARD_MODEL_PATH}")

# Initialize ensemble if second model available, otherwise use single model
USE_ENSEMBLE = MODEL_PATH_2 is not None
if USE_ENSEMBLE:
    from ensemble_detector import EnsembleDetector
    # New model gets higher weight (trained on more diverse data)
    ensemble_model = EnsembleDetector([MODEL_PATH, MODEL_PATH_2], weights=[0.4, 0.6])
    model = YOLO(MODEL_PATH)  # Keep for class names
    print(f"✅ Ensemble mode: Using {MODEL_PATH} + {MODEL_PATH_2}")
else:
    model = YOLO(MODEL_PATH)
    ensemble_model = None
    print(f"📦 Single model mode: Using {MODEL_PATH}")

board_model = YOLO(BOARD_MODEL_PATH)

# Optional: Secondary classifier for low-confidence detections
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH")
if not CLASSIFIER_MODEL_PATH:
    for path in ["models/piece_classifier.pt", "src/models/piece_classifier.pt"]:
        if os.path.exists(path):
            CLASSIFIER_MODEL_PATH = path
            break

USE_CLASSIFIER = os.getenv("USE_CLASSIFIER", "false").lower() == "true"
if USE_CLASSIFIER:
    try:
        from piece_classifier import PieceVerifier
        piece_verifier = PieceVerifier(CLASSIFIER_MODEL_PATH, confidence_threshold=0.65)
        print(f"✅ Secondary classifier enabled (threshold: 0.65)")
    except Exception as e:
        print(f"⚠️ Secondary classifier not available: {e}")
        piece_verifier = None
else:
    piece_verifier = None

INPUT_SIZE = 640


CLASSES = model.names

PIECE_MAP = {
    'white-pawn': 'P', 'white-rook': 'R', 
    'white-knight': 'N', 'white-bishop': 'B',
    'white-queen': 'Q', 'white-king': 'K',
    
    'black-pawn': 'p', 'black-rook': 'r', 
    'black-knight': 'n', 'black-bishop': 'b',
    'black-queen': 'q', 'black-king': 'k'
}

# Maximum pieces per type (accounting for pawn promotions)
# King: always 1, Queen: 1 + 8 pawn promotions max = 9, etc.
PIECE_LIMITS = {
    'K': 1, 'Q': 9, 'R': 10, 'B': 10, 'N': 10, 'P': 8,
    'k': 1, 'q': 9, 'r': 10, 'b': 10, 'n': 10, 'p': 8
}

# Confusion matrix: what a piece might be misdetected as
LIKELY_CONFUSIONS = {
    'Q': ['R', 'B', 'K'],  # Queen often confused with rook, bishop, or king
    'q': ['r', 'b', 'k'],
    'K': ['Q'],
    'k': ['q'],
    'R': ['Q'],
    'r': ['q'],
    'B': ['Q', 'P'],
    'b': ['q', 'p'],
}


def get_best_move(fen, turn='w'):
    """
    Get best move for a position using local Stockfish engine.
    
    Args:
        fen: FEN string (board position only, no turn/castling info)
        turn: 'w' for white to move, 'b' for black to move
    
    Returns:
        Best move in UCI format (e.g., 'e2e4') or None if error
    """
    import subprocess
    import shutil
    
    try:
        # Find Stockfish binary
        stockfish_path = shutil.which("stockfish")
        if not stockfish_path:
            # Try common locations (Linux + macOS)
            for path in ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"]:
                if os.path.exists(path):
                    stockfish_path = path
                    break
        
        if not stockfish_path:
            print("[Stockfish] Not found, falling back to Lichess API")
            return get_best_move_lichess(fen, turn)
        
        # Complete FEN with turn info (no castling, no en passant, counters)
        full_fen = f"{fen} {turn} - - 0 1"
        
        # Run Stockfish as subprocess
        process = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send commands to Stockfish
        commands = f"""uci
isready
position fen {full_fen}
go depth 15
"""
        stdout, stderr = process.communicate(input=commands, timeout=10)
        
        # Parse output for best move
        for line in stdout.split('\n'):
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                    if best_move != '(none)':
                        print(f"[Stockfish] Best move: {best_move}")
                        return best_move
        
        return None
        
    except Exception as e:
        print(f"[Stockfish] Error: {e}")
        return None


def get_best_move_lichess(fen, turn='w'):
    """Fallback: Get best move from Lichess cloud eval API (only works for known positions)."""
    try:
        full_fen = f"{fen} {turn} - - 0 1"
        url = f"https://lichess.org/api/cloud-eval?fen={requests.utils.quote(full_fen)}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'pvs' in data and data['pvs']:
                moves = data['pvs'][0].get('moves', '')
                if moves:
                    return moves.split(' ')[0]
        return None
    except Exception as e:
        print(f"[Lichess API] Error: {e}")
        return None


def detect_board_orientation(fen_positions):
    """
    Detect how to orient the board so white pieces are at the bottom.
    Returns rotation needed: 0, 90, 180, or 270 degrees.
    
    Standard FEN orientation: white at bottom (rows 6-7), black at top (rows 0-1)
    """
    # Count pieces by position
    white_rows = []
    black_rows = []
    white_cols = []
    black_cols = []
    
    for row_idx, row in enumerate(fen_positions):
        for col_idx, cell in enumerate(row):
            if cell != '1':
                if cell.isupper():  # White piece
                    white_rows.append(row_idx)
                    white_cols.append(col_idx)
                else:  # Black piece
                    black_rows.append(row_idx)
                    black_cols.append(col_idx)
    
    if not white_rows or not black_rows:
        print("[Orientation] Not enough pieces to determine orientation")
        return 0
    
    avg_white_row = sum(white_rows) / len(white_rows)
    avg_black_row = sum(black_rows) / len(black_rows)
    avg_white_col = sum(white_cols) / len(white_cols)
    avg_black_col = sum(black_cols) / len(black_cols)
    
    print(f"[Orientation] White avg: row={avg_white_row:.1f}, col={avg_white_col:.1f}")
    print(f"[Orientation] Black avg: row={avg_black_row:.1f}, col={avg_black_col:.1f}")
    
    # Calculate differences
    row_diff = avg_white_row - avg_black_row  # Positive = white below black (correct)
    col_diff = avg_white_col - avg_black_col  # For side views
    
    # Determine best orientation based on piece positions
    # We want white at bottom (high row numbers) in final result
    
    if abs(row_diff) > abs(col_diff):
        # Pieces are separated by rows (normal or 180° view)
        if row_diff > 0:
            print("[Orientation] Normal view - no rotation needed")
            return 0
        else:
            print("[Orientation] Inverted view - 180° rotation needed")
            return 180
    else:
        # Pieces are separated by columns (side view - 90° or 270°)
        if col_diff > 0:
            print("[Orientation] Left side view - 90° rotation needed")
            return 90
        else:
            print("[Orientation] Right side view - 270° rotation needed")
            return 270


def rotate_board(fen_positions, degrees):
    """Rotate the board by specified degrees (0, 90, 180, 270)."""
    if degrees == 0:
        return fen_positions
    elif degrees == 180:
        # Flip vertically and horizontally
        return [row[::-1] for row in reversed(fen_positions)]
    elif degrees == 90:
        # Rotate 90° clockwise
        n = len(fen_positions)
        return [[fen_positions[n-1-j][i] for j in range(n)] for i in range(n)]
    elif degrees == 270:
        # Rotate 90° counter-clockwise
        n = len(fen_positions)
        return [[fen_positions[j][n-1-i] for j in range(n)] for i in range(n)]
    return fen_positions


def flip_board(fen_positions):
    """Flip the board 180 degrees (legacy, kept for compatibility)."""
    return rotate_board(fen_positions, 180)


def validate_and_correct_pieces(fen_positions, piece_detections):
    """
    Apply chess rules to correct obvious detection errors.
    - Enforce exactly 1 king per color
    - Enforce max piece limits
    - Convert excess pieces to next most likely alternative (don't just remove)
    Returns corrected fen_positions.
    """
    # Build a map of position -> (piece, confidence)
    position_info = {}
    for detection in piece_detections:
        row = detection.get('row')
        col = detection.get('col')
        if row is not None and col is not None:
            key = (row, col)
            position_info[key] = {
                'piece': detection['piece'],
                'confidence': detection.get('confidence', 0.5)
            }
    
    # Count pieces and track their positions with confidence
    piece_positions = {}  # piece -> list of (row, col, confidence)
    for row_idx, row in enumerate(fen_positions):
        for col_idx, cell in enumerate(row):
            if cell != '1':
                if cell not in piece_positions:
                    piece_positions[cell] = []
                # Get confidence from detection info if available
                conf = position_info.get((row_idx, col_idx), {}).get('confidence', 0.5)
                piece_positions[cell].append((row_idx, col_idx, conf))
    
    # Count pieces
    piece_counts = {p: len(positions) for p, positions in piece_positions.items()}
    print(f"[Validation] Piece counts: {piece_counts}")
    
    corrections_made = []
    
    # Enforce limits for each piece type
    for piece, positions in piece_positions.items():
        max_allowed = PIECE_LIMITS.get(piece, 10)
        
        if len(positions) > max_allowed:
            excess = len(positions) - max_allowed
            
            # Sort by confidence (lowest first) to convert least confident
            positions_sorted = sorted(positions, key=lambda x: x[2])
            
            # Convert excess pieces to next most likely alternative
            for i in range(excess):
                row, col, conf = positions_sorted[i]
                original_piece = fen_positions[row][col]
                
                # Get the most likely alternative from confusion matrix
                alternatives = LIKELY_CONFUSIONS.get(original_piece, [])
                new_piece = None
                
                if alternatives:
                    # Find first alternative that doesn't exceed its limit
                    for alt in alternatives:
                        alt_count = piece_counts.get(alt, 0)
                        alt_limit = PIECE_LIMITS.get(alt, 10)
                        if alt_count < alt_limit:
                            new_piece = alt
                            piece_counts[alt] = alt_count + 1
                            break
                
                if new_piece:
                    fen_positions[row][col] = new_piece
                    corrections_made.append(f"{original_piece}->{new_piece} at ({row},{col}) conf={conf:.2f}")
                    print(f"[Validation] Converted {original_piece} to {new_piece} at ({row},{col})")
                else:
                    # No valid alternative, just remove
                    fen_positions[row][col] = '1'
                    corrections_made.append(f"Removed {original_piece} at ({row},{col}) conf={conf:.2f}")
            
            piece_counts[piece] = max_allowed
            print(f"[Validation] {piece}: {len(positions)} -> {max_allowed}")
    
    # Pieces that could be confused with king (in order of likelihood)
    KING_CANDIDATES = {
        'K': ['Q', 'R', 'B'],  # White pieces that might be a white king
        'k': ['q', 'r', 'b']   # Black pieces that might be a black king
    }
    
    # Ensure exactly 1 king per color - if missing, promote the best candidate
    for king in ['K', 'k']:
        # Recount after previous corrections
        king_count = sum(1 for row in fen_positions for cell in row if cell == king)
        
        if king_count == 0:
            color = "white" if king == 'K' else "black"
            print(f"[Validation] No {color} king detected - looking for candidate to promote")
            
            # Find the best candidate to promote to king
            candidates = KING_CANDIDATES[king]
            best_candidate = None
            best_pos = None
            
            for candidate in candidates:
                if candidate in piece_positions and piece_positions[candidate]:
                    # Get the highest confidence piece of this type
                    positions_sorted = sorted(piece_positions[candidate], key=lambda x: x[2], reverse=True)
                    for row, col, conf in positions_sorted:
                        # Check if this position still has this piece (not already converted)
                        if fen_positions[row][col] == candidate:
                            best_candidate = candidate
                            best_pos = (row, col, conf)
                            break
                    if best_candidate:
                        break
            
            if best_candidate and best_pos:
                row, col, conf = best_pos
                fen_positions[row][col] = king
                corrections_made.append(f"{best_candidate}->{king} at ({row},{col}) conf={conf:.2f} (king required)")
                print(f"[Validation] Promoted {best_candidate} to {king} at ({row},{col}) - king is required!")
            else:
                print(f"[Validation] WARNING: Could not find a candidate to promote to {color} king!")
    
    if corrections_made:
        print(f"[Validation] Corrections: {corrections_made}")
    else:
        print("[Validation] All piece counts valid")
    
    return fen_positions


def preprocess(image_bytes):
    """Preprocess image bytes to numpy array for YOLO inference."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    return img


def order_corners(points):
    """
    Order 4 corner points as: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    This is critical for correct perspective transformation.
    """
    points = np.array(points, dtype=np.float32)
    
    # Sort by y-coordinate (top to bottom)
    sorted_by_y = points[np.argsort(points[:, 1])]
    
    # Top two points (smaller y)
    top_points = sorted_by_y[:2]
    # Bottom two points (larger y)
    bottom_points = sorted_by_y[2:]
    
    # Sort top points by x (left to right)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x (left to right)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def get_4_corners_from_polygon(polygon_points):
    """
    Extract exactly 4 corners from a polygon (which may have more points).
    Uses convex hull and selects the 4 extreme corners.
    """
    points = np.array(polygon_points, dtype=np.float32)
    
    if len(points) == 4:
        return order_corners(points)
    
    if len(points) < 4:
        return None
    
    # Use convex hull to get outer points
    hull = cv2.convexHull(points)
    hull_points = hull.reshape(-1, 2)
    
    if len(hull_points) < 4:
        return None
    
    if len(hull_points) == 4:
        return order_corners(hull_points)
    
    # If more than 4 points, find the 4 extreme corners
    # Method: Find centroid, then find points in each quadrant furthest from center
    centroid = np.mean(hull_points, axis=0)
    
    # Categorize points by quadrant relative to centroid
    top_left_candidates = []
    top_right_candidates = []
    bottom_left_candidates = []
    bottom_right_candidates = []
    
    for pt in hull_points:
        if pt[0] < centroid[0] and pt[1] < centroid[1]:
            top_left_candidates.append(pt)
        elif pt[0] >= centroid[0] and pt[1] < centroid[1]:
            top_right_candidates.append(pt)
        elif pt[0] < centroid[0] and pt[1] >= centroid[1]:
            bottom_left_candidates.append(pt)
        else:
            bottom_right_candidates.append(pt)
    
    def get_extreme(candidates, is_top, is_left):
        if not candidates:
            return None
        candidates = np.array(candidates)
        # For corners, we want the point furthest from centroid in that direction
        if is_top and is_left:
            return candidates[np.argmin(candidates[:, 0] + candidates[:, 1])]
        elif is_top and not is_left:
            return candidates[np.argmax(candidates[:, 0] - candidates[:, 1])]
        elif not is_top and is_left:
            return candidates[np.argmax(candidates[:, 1] - candidates[:, 0])]
        else:  # bottom right
            return candidates[np.argmax(candidates[:, 0] + candidates[:, 1])]
    
    tl = get_extreme(top_left_candidates, True, True)
    tr = get_extreme(top_right_candidates, True, False)
    bl = get_extreme(bottom_left_candidates, False, True)
    br = get_extreme(bottom_right_candidates, False, False)
    
    if any(x is None for x in [tl, tr, bl, br]):
        # Fallback: just take 4 points with largest area
        return order_corners(hull_points[:4])
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def extract_board_polygon(board_results):
    """
    Extract the chessboard polygon from YOLO segmentation results.
    Returns ordered 4 corners or None if not detected.
    """
    masks = board_results[0].masks
    if masks is None or masks.data is None:
        return None
    
    masks_data = masks.data
    
    # Merge all masks
    merged_mask = torch.any(masks_data, dim=0).int()
    merged = merged_mask.cpu().numpy().astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (should be the board)
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon_points = approx.reshape(-1, 2).tolist()
    
    # Get exactly 4 corners
    corners = get_4_corners_from_polygon(polygon_points)
    
    return corners


def FEN_extract(pieces, board_corners):
    """
    Extract FEN string from piece detections and board corners.
    """
    if not pieces:
        return "8/8/8/8/8/8/8/8"
    
    # Debug: show what classes we're getting
    print(f"[FEN Debug] Piece classes: {[p['class'] for p in pieces]}")
    print(f"[FEN Debug] PIECE_MAP keys: {list(PIECE_MAP.keys())}")
    
    # Get piece positions (using bottom-center of bounding box for better accuracy)
    piece_positions = []
    for piece in pieces:
        x1, y1, x2, y2 = piece['bbox']
        center_x = (x1 + x2) / 2
        # Use bottom of bounding box (minus small offset) - pieces stand on squares
        bottom_y = y2 - 10
        piece_char = PIECE_MAP.get(piece['class'])
        if piece_char:
            piece_positions.append({
                'x': center_x,
                'y': bottom_y,
                'piece': piece_char,
                'confidence': piece['confidence']
            })
        else:
            print(f"[FEN Debug] Unknown class: {piece['class']}")
    
    # Initialize empty board
    fen_positions = [['1' for _ in range(8)] for _ in range(8)]
    
    if board_corners is not None and len(board_corners) == 4:
        # Use perspective transform to map pieces to board squares
        src_pts = board_corners.astype(np.float32)
        dst_pts = np.array([
            [0, 0], 
            [INPUT_SIZE, 0], 
            [INPUT_SIZE, INPUT_SIZE], 
            [0, INPUT_SIZE]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        for pos in piece_positions:
            pt = np.array([[[pos['x'], pos['y']]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt, M)[0][0]
            
            # Map to 8x8 grid
            col = int(warped_pt[0] // (INPUT_SIZE / 8))
            row = int(warped_pt[1] // (INPUT_SIZE / 8))
            
            # Clamp to valid range
            col = max(0, min(7, col))
            row = max(0, min(7, row))
            
            # Handle collisions: keep piece with higher confidence
            current = fen_positions[row][col]
            if current == '1':
                fen_positions[row][col] = pos['piece']
                # Store row/col for validation later
                pos['row'] = row
                pos['col'] = col
    else:
        # Fallback: estimate board from piece positions
        print("[FEN] No board corners, using fallback based on piece positions")
        
        if len(piece_positions) < 2:
            return "8/8/8/8/8/8/8/8"
        
        # Find bounding box of all pieces
        xs = [p['x'] for p in piece_positions]
        ys = [p['y'] for p in piece_positions]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add some margin
        margin = 20
        min_x = max(0, min_x - margin)
        max_x = min(INPUT_SIZE, max_x + margin)
        min_y = max(0, min_y - margin)
        max_y = min(INPUT_SIZE, max_y + margin)
        
        width = max_x - min_x
        height = max_y - min_y
        
        for pos in piece_positions:
            norm_x = (pos['x'] - min_x) / width if width > 0 else 0.5
            norm_y = (pos['y'] - min_y) / height if height > 0 else 0.5
            
            col = int(norm_x * 8)
            row = int(norm_y * 8)
            
            col = max(0, min(7, col))
            row = max(0, min(7, row))
            
            if fen_positions[row][col] == '1':
                fen_positions[row][col] = pos['piece']
                pos['row'] = row
                pos['col'] = col
    
    # Validate and correct pieces based on chess rules
    fen_positions = validate_and_correct_pieces(fen_positions, piece_positions)
    
    # Detect and fix board orientation (white should be at bottom = high row indices)
    rotation = detect_board_orientation(fen_positions)
    if rotation != 0:
        fen_positions = rotate_board(fen_positions, rotation)
        print(f"[FEN] Board rotated {rotation}° to correct orientation")
    
    # Convert to FEN string
    # FEN is always from white's perspective: rank 8 (top) to rank 1 (bottom)
    # After orientation fix, row 0 = rank 8, row 7 = rank 1
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
    
    return '/'.join(fen_rows)


@app.post("/detect_fen")
async def detect_fen(file: UploadFile, turn: str = Query(default="w", regex="^[wb]$")):
    """
    Detect chess pieces and board from an image, return FEN notation and best move.
    
    Args:
        file: Image file
        turn: Whose turn to play - 'w' for white, 'b' for black (default: 'w')
    """
    try:
        img_bytes = await file.read()
        img = preprocess(img_bytes)
        
        # Create debug image copy
        debug_img = img.copy()

        # Run piece detection
        results = model.predict(img, verbose=False, conf=0.25)
        
        # Run board detection
        board_results = board_model.predict(img, verbose=False)

        # Extract pieces
        pieces = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pieces.append({
                    "class": CLASSES[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                # Log detailed piece info
                print(f"[Piece] {CLASSES[cls_id]} at ({x1},{y1}) conf={conf:.2f}")

                # Draw piece bounding box (green)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(debug_img, f"{CLASSES[cls_id]}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # Verify low-confidence pieces with secondary classifier
        verified_count = 0
        if piece_verifier and pieces:
            pieces = piece_verifier.verify_detections(img, pieces)
            verified_count = sum(1 for p in pieces if 'verification' in p)
            if verified_count > 0:
                print(f"[Classifier] Verified {verified_count} low-confidence pieces")

        # Extract board corners
        board_corners = extract_board_polygon(board_results)
        
        # Also get raw polygon for comparison
        raw_polygon = None
        masks = board_results[0].masks
        if masks is not None and masks.data is not None:
            merged_mask = torch.any(masks.data, dim=0).int()
            merged = merged_mask.cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                raw_polygon = approx.reshape(-1, 2).tolist()
                
                # Draw raw polygon (cyan) - what YOLO detected
                cv2.polylines(debug_img, [np.array(raw_polygon, dtype=np.int32)], 
                             isClosed=True, color=(255, 255, 0), thickness=2)
                print(f"[Debug] Raw polygon points: {len(raw_polygon)}")
        
        # Draw extracted 4 corners (red circles) if detected
        if board_corners is not None:
            for i, corner in enumerate(board_corners):
                pt = (int(corner[0]), int(corner[1]))
                cv2.circle(debug_img, pt, 8, (255, 0, 0), -1)
                cv2.putText(debug_img, f"{i}", (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # Draw 4-corner polygon (blue)
            corners_int = board_corners.astype(np.int32)
            cv2.polylines(debug_img, [corners_int], isClosed=True, color=(0, 0, 255), thickness=3)
        
        board_detected = board_corners is not None
        print(f"[Detection] Pieces: {len(pieces)}, Board detected: {board_detected}")
        if board_corners is not None:
            print(f"[Debug] 4 corners: {board_corners.tolist()}")
        
        # Save debug image
        debug_path = "debug_server_detection.jpg"
        debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(debug_path, debug_img_bgr)
        print(f"[Debug] Saved to {debug_path}")
        
        # Extract FEN
        fen = FEN_extract(pieces, board_corners)
        print(f"[FEN] {fen}")
        
        # Get best move from Lichess API
        best_move = get_best_move(fen, turn)
        turn_name = "White" if turn == 'w' else "Black"
        if best_move:
            print(f"[Best Move] {turn_name}: {best_move}")
        
        return {
            "fen": fen,
            "pieces_count": len(pieces),
            "board_detected": board_detected,
            "best_move": best_move,
            # Enhanced detection info
            "detection_mode": "ensemble" if USE_ENSEMBLE else "single",
            "pieces_verified": verified_count,
            "avg_confidence": round(sum(p['confidence'] for p in pieces) / len(pieces), 3) if pieces else 0,
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))  # Default 7860 for Hugging Face
    print(f"🚀 Starting Chess Detection Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
