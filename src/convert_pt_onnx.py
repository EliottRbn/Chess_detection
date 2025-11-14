from ultralytics import YOLO

model = YOLO("src/runs/best.pt")
model.export(format="tfjs",optimize=True,device="cpu")
print("Model exported to TensorFlow.js format.")