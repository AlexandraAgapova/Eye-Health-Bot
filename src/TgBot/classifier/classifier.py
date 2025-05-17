import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


FACE_LANDMARKS = [10, 109, 67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 138, 135, 136, 169, 
                  150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 364, 397, 435, 401, 
                  366, 447, 366, 389, 251, 284, 332, 297, 338]

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def process_image(image_path) -> tuple:
    """
    Обрабатывает одно изображение, подготавливая его к inference и сохраняет результат
    If image not found -> (None, 0)
    If face not found -> (stock_image, 0)
    If founded -> (cropped_image, 1)
    """
    image = cv2.imread(image_path)
    if image is None:
        return (None, 0)
    
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
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w, x + w_rect + padding)
                y2 = min(h, y + h_rect + padding)
                
                cropped = image[y1:y2, x1:x2]
                
                # Возвращаем полученное изображение
                return (cropped, 1)
    
    return (image, 0)



path_to_h_u_model = "../models/YOLOv11/healthy_unhealthy.pt"
path_to_b_dc_model = "../models/YOLOv11/bags_circles.pt"
health_un_model = YOLO(path_to_h_u_model)
bags_dc_model = YOLO(path_to_b_dc_model)

models_ = [health_un_model, bags_dc_model]

class Classifier:
    def __init__(self, models_):
        self.models = models_

    def predict(self, image : str)->int:
        """
        not found face -> -1    
        healthy -> 0    
        bags -> 1   
        dark_circles -> 2   
        bags and dark_circles -> 3
        """
        ready_to_classify_tuple = process_image(image)

        # отсекаем все изображения без лица
        if (ready_to_classify_tuple[1] == 0):
            return -1

        results_hu = models_[0].predict(ready_to_classify_tuple[0])

        healthy_conf = results_hu[0].probs.data[0].item()

        if (healthy_conf > 0.7):
            return 0
        else:
            results_bc = models_[1].predict(ready_to_classify_tuple[0])
            dark_circles_conf = results_bc[0].probs.data[0].item()
            bags_conf = results_bc[0].probs.data[1].item()
            if (dark_circles_conf > 0.4 and bags_conf > 0.4):
                return 3
            if (dark_circles_conf > 0.5):
                return 2
            else:
                return 1



# Example of using
"""
orchestre = Classifier(models_) 

result = orchestre.predict("/home/m.sukhanov1/ML/Eye_bags_detection/ML-Project/src/classifier/a294bba19001ec335a702476a3430594.jpg")

if (result == -1):
    print("Лицо не распознано")
if (result == 0):
    print("лицо выглядит здоровым")
if (result == 1):
    print("похоже, что у вас мешки под глазами")
if (result == 2):
    print("Кажется, у вас черные круги")
if (result == 3):
    print("дела плохи. и черные круги и мешки")
"""