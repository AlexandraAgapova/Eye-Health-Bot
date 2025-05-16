from config import MODEL_PATH
import torch
from pathlib import Path

model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

def predict_image_class(image_path: str) -> int:
    results = model(image_path)

    # Здесь логика обработки результатов YOLO
    return 0
