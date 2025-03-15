import cv2
import mediapipe as mp
import numpy as np

def detect_eye_bags():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True)

    # Индексы точек вокруг глаз и под глазами
    EYE_LANDMARKS = {
        "left": [417, 441, 442, 443, 444, 445, 342, 265, 340, 280, 330, 329, 277, 343, 412, 465],
        "right": [193, 221, 222, 223, 224, 225, 113, 35, 111, 50, 101, 100, 47, 114, 188, 245]
    }

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
                for eye, landmarks in EYE_LANDMARKS.items():
                    points = []
                    for idx in landmarks:
                        point = face_landmarks.landmark[idx]
                        x, y = int(point.x * w), int(point.y * h)
                        points.append((x, y))

                    # Создаем замкнутый контур вокруг глаза и увеличенной области под ним
                    points = np.array(points, np.int32)
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Eye Bags Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_eye_bags()

if __name__ == '__main__':
    main()
