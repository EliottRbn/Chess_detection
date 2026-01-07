# Chess Vision AI

A complete chess board digitization solution using Artificial Intelligence. Take a photo of your physical chessboard and instantly get the digital FEN position and the best move to play.

**Unified Monorepo Architecture:**
- **Frontend:** Mobile Application (React Native / Expo)
- **Backend:** AI Inference Server (Python / FastAPI / YOLOv8)

---

## Key Features

- **Instant Detection**: Analyzes a chessboard photo in < 1 second.
- **Hybrid AI (Level 3)**:
  - Object Detection (YOLO8m/11n) to locate pieces.
  - Segmentation (YOLO-seg) to extract the board.
  - Specialized Classifier (MobileNetV3) to validate ambiguous pieces (e.g., King vs Queen).
  - Logic Validation enforcing chess rules (max 1 king, max pawns, etc.).
- **Move Calculation**: Integration of Stockfish to suggest the optimal move.
- **Mobile Interface**: Modern UX, 2D visualization, turn management (White/Black).

---

## Project Structure

```
Chess/
├── ChessExpo/           # Mobile Application (React Native)
│   ├── app/             # Pages (Expo Router)
│   ├── assets/          # Images & Icons
│   └── src/             # Components & Business Logic
│
└── ChessServer/         # AI Backend (Python)
    ├── src/             # Server Source Code
    │   ├── server_fen.py   # Main FastAPI Entrypoint
    │   └── scripts/        # Tools for training/debugging
    ├── models/          # AI Model Weights (.pt/.onnx)
    └── assets/          # Test Data
```

---

## Installation & Quick Start

### 1. Prerequisites
- Node.js & npm
- Python 3.9+
- Expo Go (on your mobile device)

### 2. Start AI Server (Backend)

```bash
cd ChessServer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start server
cd src
python server_fen.py
```
*The server will start on `http://0.0.0.0:7860`*

### 3. Start Mobile App (Frontend)

```bash
cd ChessExpo

# Install dependencies
npm install

# Start Expo
npx expo start
```
*Scan the QR code with the "Expo Go" app on your phone.*

**Note:** Ensure your phone and computer are on the same WiFi network. Update the server IP in `ChessExpo/app/(tabs)/index.tsx` if needed.

---

## Technical Details (AI Pipeline)

The recognition pipeline consists of 5 stages:

1.  **Board Segmentation**: A YOLOv8-seg model detects the 4 corners of the chessboard and applies a perspective transformation (Homography) to obtain a flat 640x640 view.
2.  **Piece Detection**: A YOLOv8m (or YOLO11n) model detects all pieces on the rectified board.
3.  **Filtering (Confidence & IoU)**: Removes duplicate or overlapping detections.
4.  **Secondary Classification (Validation)**: If a detection has low confidence (< 65%), a fast CNN classifier (MobileNetV3) verifies the piece identity (e.g., confirms it is a King, not a Queen).
5.  **Chess Logic**: A validation algorithm corrects impossible states (e.g., "3 Kings detected" -> keeps the most probable one).
6.  **FEN & Engine**: Converts the detected 8x8 grid into a FEN string, then sends it to Stockfish for analysis.

---

## Authors & Credits

Project realized as part of the "Systems Design - Rapid Prototyping" course at IPSA.

**Datasets used:**
- Roboflow Universe (Chess Pieces by Bruno Leonardo)
- Synthetic Data Generation (Data Augmentation)

---
© 2026 - Chess Vision Team