from dotenv import load_dotenv
load_dotenv()
import os
import telebot

BOT_TOKEN = os.environ.get('BOT_API_KEY')

bot = telebot.TeleBot(BOT_TOKEN)
@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    text="you mean me?"
    bot.reply_to(message, text)

bot.infinity_polling()