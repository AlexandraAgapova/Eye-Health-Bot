import os
import shutil

os.mkdir("train")
os.mkdir("val")
os.mkdir("train/healthy")
os.mkdir("train/unhealthy")
os.mkdir("val/healthy")
os.mkdir("val/unhealthy")

healthy_images = list(os.walk("healthy/"))[-1][-1]
unhealthy_images = list(os.walk("unhealthy/"))[-1][-1]

size_of_health = len(healthy_images)
size_of_unhealth = len(unhealthy_images)


for i in range(size_of_health):
    if (i < 20 * size_of_health / 100):
        shutil.copy(f"data/processed/for_YOLO/healthy/{healthy_images[i]}",f"data/processed/for_YOLO/val/healthy/{healthy_images[i]}")
    else:
        shutil.copy(f"healthy/{healthy_images[i]}",f"train/healthy/{healthy_images[i]}")

for i in range(size_of_unhealth):
    if (i < 20 * size_of_unhealth / 100):
        shutil.copy(f"unhealthy/{unhealthy_images[i]}",f"val/unhealthy/{unhealthy_images[i]}")
    else:
        shutil.copy(f"unhealthy/{unhealthy_images[i]}",f"train/unhealthy/{unhealthy_images[i]}")