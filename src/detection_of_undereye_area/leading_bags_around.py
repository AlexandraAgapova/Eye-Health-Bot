import cv2
import mediapipe as mp

# Инициализация Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True)

# Индексы точек под глазами в Mediapipe
UNDER_EYE_LANDMARKS = {
    "left": [33, 145, 159, 130],  # Точки от внешнего угла глаза до нижней области
    "right": [263, 374, 386, 359]  # Точки от внешнего угла глаза до нижней области
}

# Захват видео с камеры устройства
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения в формат RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            for eye, landmarks in UNDER_EYE_LANDMARKS.items():
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for idx in landmarks:
                    point = face_landmarks.landmark[idx]
                    x, y = int(point.x * w), int(point.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Расширяем область вниз и в стороны для охвата носовых пазух
                y_min += 10
                y_max += 40
                x_min -= 10
                x_max += 10
                
                # Рисуем прямоугольник вокруг расширенной зоны под глазами
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Отображение результата
    cv2.imshow("Under Eye Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
