import os
import shutil

# os.mkdir("train")
# os.mkdir("val")
# os.mkdir("train/bags")
# os.mkdir("train/darkCircles")
# os.mkdir("train/healthy")
# os.mkdir("val/healthy")
# os.mkdir("val/darkCircles")
# os.mkdir("val/bags")


healthy_images = list(os.walk("healthy/"))[-1][-1]
bags_images = list(os.walk("bags/"))[-1][-1]
dark_circles_images = list(os.walk("darkCycles/"))[-1][-1]

size_of_health = len(healthy_images)
size_of_bags = len(bags_images)
size_of_dc = len(dark_circles_images)


for i in range(size_of_health):
    if (i < 20 * size_of_health / 100):
        shutil.copy(f"healthy/{healthy_images[i]}",f"val/healthy/{healthy_images[i]}")
    else:
        shutil.copy(f"healthy/{healthy_images[i]}",f"train/healthy/{healthy_images[i]}")

for i in range(size_of_bags):
    if (i < 20 * size_of_bags / 100):
        shutil.copy(f"bags/{bags_images[i]}",f"val/bags/{bags_images[i]}")
    else:
        shutil.copy(f"bags/{bags_images[i]}",f"train/bags/{bags_images[i]}")

for i in range(size_of_dc):
    if (i < 20 * size_of_dc / 100):
        shutil.copy(f"darkCycles/{dark_circles_images[i]}",f"val/darkCircles/{dark_circles_images[i]}")
    else:
        shutil.copy(f"darkCycles/{dark_circles_images[i]}",f"train/darkCircles/{dark_circles_images[i]}")