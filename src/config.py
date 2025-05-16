import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/YOLOv11/yolov11_weights.pt")

CLASS_NAMES = {
    -1: "❌ Лицо не обнаружено",
     0: "✅ Глаза выглядят здоровыми",
     1: "⚠️ Мешки под глазами",
     2: "⚠️ Тёмные круги под глазами",
}
