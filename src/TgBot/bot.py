import telebot
import uuid
import re
import os
from model import predict_image_class

bot = telebot.TeleBot("YOUR_TELEGRAM_BOT_TOKEN")
helloVarious = {"привет", "Привет", "здарова", "приветик", "Приветик", "Здарова"}
DATA_DIR = "data"

@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(
        message.chat.id,
        "Привет! Давай я помогу понять, есть ли у тебя проблемы в области глаз, и дам рекомендации?",
    )


@bot.message_handler(content_types=["text"])
def get_text_messages(message):
    if message.text in helloVarious:
        bot.send_message(message.from_user.id, "Привет, отправляй свое фото, и я проанализирую 👁✨")
    elif message.text == "/help":
        bot.send_message(
            message.from_user.id,
            "Давай определим, есть ли у тебя проблемы в области глаз? Просто отправь мне свое фото!",
        )
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю:( Напиши /help.")


@bot.message_handler(content_types=["photo"])
def download_photo(message):
    user_id = message.from_user.id
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    filename = f"{user_id}.jpg"
    file_path = os.path.join(DATA_DIR, filename)

    with open(file_path, "wb") as new_file:
        new_file.write(downloaded_file)

    prediction = predict_image_class(file_path)

    response = {
        -1: "❌ Не удалось распознать лицо. Попробуйте другое фото.",
         0: "✅ Всё в порядке! Глаза выглядят здоровыми.",
         1: "⚠️ Обнаружены мешки под глазами. Рекомендую высыпаться и уменьшить потребление соли.",
         2: "⚠️ Обнаружены тёмные круги. Проверь режим сна и снизьте стресс.",
    }

    bot.send_message(user_id, response.get(prediction, "Произошла ошибка при анализе."))



bot.polling(none_stop=True, interval=0)
