import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from collections import deque

def process_image(image_path, label):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    EYE_LANDMARKS = {
        "left": [417, 441, 442, 443, 444, 445, 342, 265, 340, 280, 330, 329, 277, 343, 412, 465],
        "right": [193, 221, 222, 223, 224, 225, 113, 35, 111, 50, 101, 100, 47, 114, 188, 245]
    }
    FACE_LANDMARKS = [10, 109, 67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 138, 135, 136, 169, 
                      150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 364, 397, 435, 401, 
                      366, 447, 366, 389, 251, 284, 332, 297, 338]
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        print(f"Warning: No face detected in {image_path}")
        return None
    
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    
    avg_bgr_values = {}
    
    for eye, landmarks in EYE_LANDMARKS.items():
        points = [
            (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            for idx in landmarks if idx < len(face_landmarks.landmark)
        ]
        if len(points) < len(landmarks):
            print(f"Warning: Недостаточно точек для {eye} глаза в {image_path}")
            return None

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
        avg_bgr_values[eye] = cv2.mean(frame, mask=mask)[:3]
    
    face_points = [
        (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
        for idx in FACE_LANDMARKS if idx < len(face_landmarks.landmark)
    ]
    if len(face_points) < len(FACE_LANDMARKS):
        print(f"Warning: Недостаточно точек для построения лица в {image_path}")
        return None
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(face_points, np.int32)], 255)
    avg_bgr_values["face"] = cv2.mean(frame, mask=mask)[:3]
    
    return {
        "ID": os.path.basename(image_path),
        "mean_R": avg_bgr_values["face"][2],
        "mean_G": avg_bgr_values["face"][1],
        "mean_B": avg_bgr_values["face"][0],
        "R_eye_mean_R": avg_bgr_values["right"][2],
        "R_eye_mean_G": avg_bgr_values["right"][1],
        "R_eye_mean_B": avg_bgr_values["right"][0],
        "L_eye_mean_R": avg_bgr_values["left"][2],
        "L_eye_mean_G": avg_bgr_values["left"][1],
        "L_eye_mean_B": avg_bgr_values["left"][0],
        "is_Good": label
    }

def process_dataset():
    dataset_dirs = {"../dataset/disease": 0, "../dataset/good": 1}
    results = []
    
    for dir_path, label in dataset_dirs.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue
        
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            result = process_image(file_path, label)
            if result:
                results.append(result)
    
    df = pd.DataFrame(results)
    os.makedirs("../dataset/csv", exist_ok=True)
    csv_path = "../dataset/csv/eye_disease.csv"
    df.to_csv(csv_path, sep=';', index=False)
    print(f"Results saved to {csv_path}")

def main():
    process_dataset()

if __name__ == '__main__':
    main()
