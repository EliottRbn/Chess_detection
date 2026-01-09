FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models (ensemble: best.pt primary + ONNX confirmation)
COPY src/runs/best.pt ./models/
COPY src/runs/detect_pieces_1.onnx ./models/
COPY src/runs/chessboard_detection.pt ./models/

# Copy app code
COPY src/server_fen.py ./app.py

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
