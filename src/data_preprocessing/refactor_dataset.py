import os
import shutil
import cv2
import mediapipe as mp

# Инициализация модели распознавания лиц
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Пути к папкам
source_dirs = ["../../data/raw/Dataset/Bags_under_the_eyes", "../../data/raw/Dataset/Dark_circles_under_the_eyes", "../../data/raw/Dataset/Swelling_of_the_eyelids"]
healthy_dir = "../../data/raw/Dataset/Healthy"

# Куда копировать файлы
output_good = "../../data/processed/healthy"
output_disease = "../../data/processed/disease"

def clear_directory(directory):
    """ Очищает указанную директорию перед сохранением новых файлов. """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Удаляем папку со всеми файлами
    os.makedirs(directory)  # Создаём её заново

def detect_single_face(image_path):
    """Определяет, есть ли ровно 1 лицо на изображении."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Ошибка загрузки: {image_path}")
        return False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections is None or len(results.detections) != 1:
        print(f"🚫 Пропущено: {image_path} (лиц: {len(results.detections) if results.detections else 0})")
        return False

    return True

def copy_and_rename_images(source_folder, destination_folder, start_index=0):
    """ Копирует и переименовывает файлы, если на них найдено 1 лицо. """
    index = start_index
    for filename in sorted(os.listdir(source_folder)):  # Сортируем для стабильности
        file_path = os.path.join(source_folder, filename)

        if not os.path.isfile(file_path):
            continue  # Пропускаем папки

        if not detect_single_face(file_path):  # Проверяем, есть ли ровно 1 лицо
            continue

        ext = os.path.splitext(filename)[1]  # Получаем расширение файла
        new_filename = f"image{index}{ext}"  # Новый формат имени
        new_path = os.path.join(destination_folder, new_filename)

        shutil.copy2(file_path, new_path)  # Копируем файл с сохранением метаданных
        print(f"✅ Копировано: {new_filename}")
        index += 1

    return index  # Возвращаем следующий доступный индекс

# Очищаем папки перед началом работы
clear_directory(output_good)
clear_directory(output_disease)

# Обрабатываем Healthy -> ../good
next_index = copy_and_rename_images(healthy_dir, output_good)

# Обрабатываем disease-категории -> ../disease
for folder in source_dirs:
    next_index = copy_and_rename_images(folder, output_disease, next_index)

print("✅ Все подходящие файлы скопированы!")
