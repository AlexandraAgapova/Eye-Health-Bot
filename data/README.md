# User Image Storage

This folder contains the latest photos uploaded by Telegram bot users.

## How it works

- Each user photo is saved with a **unique filename** based on their Telegram `user_id`: 123456789.jpg
- Only **one image per user** is stored at a time. If a user sends a new photo, it **overwrites** the previous one.
- The goal is to keep the storage clean and focused on the **most recent submission** from each user.

## Used by the Model

- The classification model (based on YOLOv11) automatically loads the image from this directory.
- It analyzes the area under the eyes and returns a class:
- `-1` – no face detected
- `0` – healthy eyes
- `1` – eye bags
- `2` – dark circles

The result is then sent back to the user via the Telegram bot along with personalized recommendations.

---

> ⚠️ Do not modify or delete files here manually during runtime, as this directory is managed automatically by the bot.

