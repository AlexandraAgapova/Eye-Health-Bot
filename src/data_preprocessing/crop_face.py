import cv2
import mediapipe as mp
import numpy as np

FACE_LANDMARKS = [10, 109, 67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 138, 135, 136, 169, 
                  150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 364, 397, 435, 401, 
                  366, 447, 366, 389, 251, 284, 332, 297, 338]

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            mask = np.zeros((h, w), dtype=np.uint8)

            for face_landmarks in results.multi_face_landmarks:
                points = []
                for idx in FACE_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        points.append((x, y))

                if len(points) > 1:
                    points_array = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points_array], 255)  # Белая маска внутри контура

            # Применяем маску
            frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

            # Конвертируем в 3 канала, чтобы отобразить корректно
            frame = frame_masked

        cv2.imshow("Masked Face Region", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
