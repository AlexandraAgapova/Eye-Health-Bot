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
   - `3` – Dark circles and Eye bags


---

## Project Structure

```
Eye-Health-Bot/
├── data/                              # User images (named by Telegram user_id)
│   └── README.md
├── src/
│   └── TgBot/                         # Telegram bot logic
│       ├── bot.py                     # Main bot handler
│       ├── README.md
│       ├── classifier/
│       │   ├── classifier.py          # Prediction logic (YOLOv11)
│       │   ├── README.md   
│       └── models/
│           └── YOLOv11/               # YOLOv11 weights
│               ├── bags_circles.pt
│               ├── healthy_unhealthy.pt
│               ├── testing.ipynb
│               └── README.md
├── .gitignore                          # Excluded files
├── README.md                           # (you are here)
└── requirements.txt                    # Python dependencies

```

---

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/AlexandraAgapova/Eye-Health-Bot
cd Eye-Health-Bot
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

In `src/TgBot/bot.py`:

```python
bot = telebot.TeleBot("YOUR_TELEGRAM_BOT_TOKEN")
```

5. **Run the bot**

```bash
python src/TgBot/bot.py
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
