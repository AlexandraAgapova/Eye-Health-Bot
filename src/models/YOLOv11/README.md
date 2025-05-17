# YOLOv11 Model Directory

This folder contains the trained YOLOv11 model used for classifying under-eye conditions from user-submitted images.

## Contents

- `healthy_unhealty.pt` and `bags_circles.pt`  
  The PyTorch weights file of the YOLOv11 model. This file is loaded at runtime by the bot for inference.

## Model Purpose

The model is trained to:

- Detect human faces in a photo.
- Focus on the under-eye region.
- Classify the visual condition into one of the following categories:
  - `-1` – No face detected.
  - `0` – Eyes appear healthy.
  - `1` – Visible under-eye bags.
  - `2` – Noticeable dark circles.
