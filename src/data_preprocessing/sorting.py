import os
import shutil
import cv2

SOURCE_FOLDER = "to_sort"
HEALTHY_FOLDER = "healthy_rem"
UNHEALTHY_FOLDER = "unhealthy"
TRASH_FOLDER = "trash"

os.makedirs(HEALTHY_FOLDER, exist_ok=True)
os.makedirs(UNHEALTHY_FOLDER, exist_ok=True)
os.makedirs(TRASH_FOLDER, exist_ok=True)

# Получаем список всех изображений в папке
images = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

print(f"Найдено {len(images)} изображений для сортировки.")

for img_name in images:
    img_path = os.path.join(SOURCE_FOLDER, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Не удалось загрузить изображение: {img_path}")
        continue

    cv2.imshow("Сортировка изображения", image)
    print(f"Текущее изображение: {img_name} (нажми 'h' – healthy, 'u' – unhealthy, 't' – trash, 'q' – выход)")

    key = cv2.waitKey(0) & 0xFF

    if key == ord('h'):
        shutil.move(img_path, os.path.join(HEALTHY_FOLDER, img_name))
        print(f"-> {img_name} перемещено в {HEALTHY_FOLDER}")
    elif key == ord('u'):
        shutil.move(img_path, os.path.join(UNHEALTHY_FOLDER, img_name))
        print(f"-> {img_name} перемещено в {UNHEALTHY_FOLDER}")
    elif key == ord('t'):
        shutil.move(img_path, os.path.join(TRASH_FOLDER, img_name))
        print(f"-> {img_name} перемещено в {TRASH_FOLDER}")
    elif key == ord('q'):
        print("Выход из сортировки.")
        break
    else:
        print("Нажата неизвестная клавиша. Пропускаем изображение.")

    cv2.destroyAllWindows()

cv2.destroyAllWindows()

