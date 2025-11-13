from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle PyTorch
MODEL_PATH = "src/runs/yolo8-20epoch.pt"
model = YOLO(MODEL_PATH)
INPUT_SIZE = 640

# Extract class names from the model
CLASSES = model.names

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    return img

@app.post("/detect")
async def detect(file: UploadFile):
    try:
        img_bytes = await file.read()
        img = preprocess(img_bytes)
        cv2.imwrite("debug_preprocessed_image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Debug: save the preprocessed image

        # Effectuer l'inférence avec le modèle YOLO
        results = model.predict(img)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class": CLASSES[cls_id],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

        return {"detections": detections}

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
