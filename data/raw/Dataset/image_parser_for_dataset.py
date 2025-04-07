import os
import requests
from duckduckgo_search import DDGS
import mediapipe as mp
import cv2
import numpy as np

# Dictionary of queries and corresponding folders for saving images
queries = {
    "Swelling_of_the_eyelids": "Swelling_of_the_eyelids",
    "Bags_under_the_eyes": "Bags_under_the_eyes",
    "Dark_circles_under_the_eyes": "Dark_circles_under_the_eyes",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.7
)

# Create folders to save images if there are none
for folder in queries.values():
    os.makedirs(folder, exist_ok=True)

# A function to check for exactly one face in an image
def face_check(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return False
    results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results.detections and len(results.detections) == 1

# A function for downloading images on request and checking for the presence of a face
def download_images(query, folder, num_images=2000):
    count = 0   
    with DDGS() as ddgs:
        for img in ddgs.images(query, max_results=num_images):
            try:
                img_resp = requests.get(img["image"], headers=headers, timeout=10)
                if img_resp.status_code == 200 and face_check(img_resp.content):
                    ext = img["image"].split(".")[-1].lower()
                    if ext == "avif":
                        ext = "jpg"
                    filename = os.path.join(folder, f"{query}_{count}.{ext}")
                    with open(filename, "wb") as f:
                        f.write(img_resp.content)
                    count += 1
                    print(f"Saved: {filename}")

                if count >= num_images:
                    break

            except Exception as e:
                print(f"Error downloading {img['image']}: {e}")


if __name__ == "__main__":
    for ru_query, folder in queries.items():
        print(f"Downloading images for: {ru_query}")
        download_images(ru_query, folder, num_images=200)

    print("Download completed!")
