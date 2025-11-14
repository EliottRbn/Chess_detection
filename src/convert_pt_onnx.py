from ultralytics import YOLO

model = YOLO("src/runs/best.pt")
model.export(format="onnx",optimize=True)
print("Model exported to ONNX format.")