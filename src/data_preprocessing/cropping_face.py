import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Константы
FACE_LANDMARKS = [10, 109, 67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 138, 135, 136, 169, 
                  150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 364, 397, 435, 401, 
                  366, 447, 366, 389, 251, 284, 332, 297, 338]

# Пути к данным
INPUT_PATHS = {
    'bags': '../../data/raw/dataset_classify/bags',
    'dark_circles': '../../data/raw/dataset_classify/darkCycles'
}

OUTPUT_PATHS = {
    'bags': '../../data/processed/for_YOLO_unhealthy/bags',
    'dark_circles': '../../data/processed/for_YOLO_unhealthy/dark_Cycles'
}

# Создаем выходные директории, если их нет
for path in OUTPUT_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def process_image(image_path, output_path):
    """Обрабатывает одно изображение и сохраняет результат"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Конвертируем в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обработка лица
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for idx in FACE_LANDMARKS:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
            
            if len(points) > 2:
                points_array = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points_array], 255)
                
                # Находим ограничивающий прямоугольник
                x, y, w_rect, h_rect = cv2.boundingRect(points_array)
                
                # Вырезаем область лица с небольшим запасом
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w, x + w_rect + padding)
                y2 = min(h, y + h_rect + padding)
                
                cropped = image[y1:y2, x1:x2]
                
                # Сохраняем обрезанное изображение
                output_filename = os.path.join(output_path, os.path.basename(image_path))
                cv2.imwrite(output_filename, cropped)
                return True
    
    return False

def process_folder(input_folder, output_folder):
    """Обрабатывает все изображения в папке"""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]
    
    for filename in tqdm(files, desc=f"Processing {os.path.basename(input_folder)}"):
        input_path = os.path.join(input_folder, filename)
        process_image(input_path, output_folder)

# Обрабатываем все папки
for category in ['bags', 'dark_circles']:
    process_folder(INPUT_PATHS[category], OUTPUT_PATHS[category])

# Освобождаем ресурсы
face_mesh.close()
print("Обработка завершена!")