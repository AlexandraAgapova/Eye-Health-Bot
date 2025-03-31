import cv2
import mediapipe as mp
import numpy as np
from collections import deque

def count_relation_of_mean_face_and_eye():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True)

    EYE_LANDMARKS = {
        "left": [417, 441, 442, 443, 444, 445, 342, 265, 340, 280, 330, 329, 277, 343, 412, 465],
        "right": [193, 221, 222, 223, 224, 225, 113, 35, 111, 50, 101, 100, 47, 114, 188, 245]
    }
    
    FACE_LANDMARKS = [10, 109, 67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 138, 135, 136, 169, 
                      150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 364, 397, 435, 401, 
                      366, 447, 366, 389, 251, 284, 332, 297, 338]
    
    avg_bgr_values = {"left": deque(maxlen=25), "right": deque(maxlen=25), "face": deque(maxlen=25)}

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Обработка глаз
                for eye, landmarks in EYE_LANDMARKS.items():
                    points = [
                        (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
                        for idx in landmarks if idx < len(face_landmarks.landmark)
                    ]
                    if len(points) < len(landmarks):  # Проверяем, все ли точки есть
                        print(f"Warning: Недостаточно точек для {eye} глаза")
                        continue

                    points = np.array(points, np.int32)
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)
                    mean_bgr = cv2.mean(frame, mask=mask)[:3]
                    
                    avg_bgr_values[eye].append(mean_bgr)
                    if len(avg_bgr_values[eye]) == 25:
                        mean_over_25 = np.mean(avg_bgr_values[eye], axis=0)
                        text_pos = tuple(points[0])
                        cv2.putText(frame, f"BGR: {tuple(map(int, mean_over_25))}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Обработка лица
                face_points = [
                    (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
                    for idx in FACE_LANDMARKS if idx < len(face_landmarks.landmark)
                ]
                if len(face_points) < len(FACE_LANDMARKS):
                    print("Warning: Недостаточно точек для построения лица")
                    continue
                
                face_points = np.array(face_points, np.int32)
                cv2.polylines(frame, [face_points], isClosed=True, color=(255, 0, 0), thickness=2)
                
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [face_points], 255)
                mean_bgr_face = cv2.mean(frame, mask=mask)[:3]
                
                avg_bgr_values["face"].append(mean_bgr_face)
                if len(avg_bgr_values["face"]) == 25:
                    mean_over_25_face = np.mean(avg_bgr_values["face"], axis=0)
                    cv2.putText(frame, f"Face BGR: {tuple(map(int, mean_over_25_face))}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Eye Bags Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    count_relation_of_mean_face_and_eye()

if __name__ == '__main__':
    main()
