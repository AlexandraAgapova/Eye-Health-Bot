import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

RIGHT_LOWER_EYELID = [453, 452, 451, 450, 449, 448, 261]
LEFT_LOWER_EYELID = [35, 31, 228, 229, 230, 231, 232, 233]

RIGHT_LOWER_BROW = [285, 295, 282, 283, 276]
LEFT_LOWER_BROW = [70, 53, 52, 65, 55]

VISIBLE_POINTS = RIGHT_LOWER_EYELID + LEFT_LOWER_EYELID + RIGHT_LOWER_BROW + LEFT_LOWER_BROW

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = []
            
            for idx in VISIBLE_POINTS:
                if idx >= len(face_landmarks.landmark):
                    continue  

                x, y = int(face_landmarks.landmark[idx].x * frame.shape[1]), int(face_landmarks.landmark[idx].y * frame.shape[0])
                points.append((x, y))

            # контур области
            if len(points) > 2:
                hull = cv2.convexHull(np.array(points))
                
                # маска
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [hull], 255)  
                
                black_frame = np.zeros_like(frame) 
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask) 
                frame = cv2.addWeighted(masked_frame, 1, black_frame, 0, 0)  

    cv2.imshow("Filtered Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()