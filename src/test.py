from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torch
import time
# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "src/runs/chessboard_detection.pt"
OUTPUT_PATH = "output_board.jpg"

MODEL_PATH_2 = "src/runs/chessboard_detection_online.pt"

def main(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Image introuvable : {image_path}")
        return
    print("✅ Loading YOLO board detector...")
    model = YOLO(MODEL_PATH)
    model_2 = YOLO(MODEL_PATH_2)
    print("🔍 Running inference...")
    results = model(str(image_path))
    results_2 = model_2(str(image_path))

    r = results[0]
    r_2 = results_2[0]

    if r.masks is None or len(r.masks) == 0:
        print("⚠️ Aucun masque détecté.")
        return
    
    if r_2.masks is None or len(r_2.masks) == 0:
        print("⚠️ Aucun masque détecté par le second modèle.")
        return

    # 1. Extraire le masque binaire
    masks = results[0].masks.data if results[0].masks is not None else None

    masks_2 = results_2[0].masks.data if results_2[0].masks is not None else None

    # 2. Trouver le contour et le rectangle minimal
    if masks is not None:
        merged_mask = torch.any(masks, dim=0).int()
        merged = merged_mask.cpu().numpy()
        contours, _ = cv2.findContours(merged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            polygon_points = approx.reshape(-1, 2).tolist()

    if masks_2 is not None:
        merged_mask_2 = torch.any(masks_2, dim=0).int()
        merged_2 = merged_mask_2.cpu().numpy()
        contours_2, _ = cv2.findContours(merged_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_2:
            contour_2 = max(contours_2, key=cv2.contourArea)
            approx_2 = cv2.approxPolyDP(contour_2, 0.02 * cv2.arcLength(contour_2, True), True)
            polygon_points_2 = approx_2.reshape(-1, 2).tolist()

    print(f"Polygon points: {polygon_points}")

    # 3. Charger l'image et dessiner le rectangle
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Impossible de charger l'image.")
        return
    cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.polylines(image, [np.array(polygon_points_2)], isClosed=True, color=(255, 0, 0), thickness=3)
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"✅ Saving annotated result → {OUTPUT_PATH}")

if __name__ == "__main__":
    for i in range(1, 6):
        main(f"{i}.jpg")
        time.sleep(3)  # Pause d'une seconde entre les tests