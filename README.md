# 👁️ Eye Health Detector Bot

Telegram bot that detects and classifies under-eye conditions (eye bags, dark circles) using computer vision (YOLOv11) and provides instant feedback to users.

---

## How It Works

1. User sends a selfie via Telegram.
2. The bot saves the image locally using the user's Telegram ID.
3. YOLOv11 model analyzes the under-eye area.
4. The bot replies with a classification and recommendation:
   - `-1` – No face detected  
   - `0` – Healthy eyes  
   - `1` – Eye bags  
   - `2` – Dark circles  

---

## Project Structure

```
ML-Project/
│
├── data/                    # User images (named by Telegram user_id)
│   └── README.md
│
├── models/
│   └── YOLOv11/             # YOLOv11 weights + model-specific README
│
├── src/
│   ├── TgBot/               # Telegram bot logic
│   │   └── README.md
│   ├── bot.py               # Main bot handler
│   ├── model.py             # Prediction logic (YOLOv11)
│   ├── config.py            # Paths, class names, constants
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Excluded files/folders
└── README.md                # (you are here)
```

---

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/your-username/eye-health-bot.git
cd eye-health-bot
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your Telegram bot token**

In `src/bot.py`:

```python
bot = telebot.TeleBot("YOUR_TELEGRAM_BOT_TOKEN")
```

5. **Download and place model weights**

Place the `yolov11_weights.pt` file in:

```
models/YOLOv11/yolov11_weights.pt
```

6. **Run the bot**

```bash
python src/bot.py
```

---

## 🛡 Recommendations

- Never push real API tokens or model weights to public repositories.
- Use `.env` and `python-dotenv` to manage secrets (optional).
- Keep image storage lightweight — only one image per user is saved (`{user_id}.jpg`).

---

## 📌 Authors & Contributors

This project was crafted with care and curiosity by  
**Alexandra Agapova** and **Mikhail Sukhanov**.

We hope it helps bring a bit more clarity and confidence to those caring for their eye health. 💙
