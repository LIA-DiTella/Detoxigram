import sys
sys.path.append('..')
import time
import asyncio
import os
from dotenv import main
import telebot
from telebot import types
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
from model_evaluation_scripts.classifiers_classes_api.perspective_classifier import perspective_classifier
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.gpt_classifier import gpt_classifier
from bot_functions.Formater import Formatter
from bot_functions.Summarizer import Summarizer
from bot_functions.ChannelAnalyzer import ChannelAnalyzer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import numpy as np

main.load_dotenv()
print(os.getcwd())

# Load environment variables
BOT_TOKEN = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM = os.environ.get('API_ID')
API_HASH_TELEGRAM = os.environ.get('API_HASH')
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_CLOUD_API_KEY = os.environ['GOOGLE_CLOUD_API_KEY']

# Initialize the classifiers
perspective = perspective_classifier(GOOGLE_CLOUD_API_KEY, attributes=['TOXICITY'], verbosity=True)
bert = hate_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert', verbosity=True)
gpt = gpt_classifier('gpt-3.5-turbo', OPENAI_API_KEY, verbosity=True, templatetype='prompt_template_few_shot')

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

if bot:  # Check if the bot is running
    print('Bot token loaded, Detoxigram is live 🚀')

loop = asyncio.new_event_loop()  # Create a new event loop
client = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  # Create a new TelegramClient

# Global variables
last_analyzed_toxicity = {}
last_channel_analyzed = None

# Initialize formatter, summarizer, and channel analyzer
formatter = Formatter(client)
summarizer = Summarizer(bot, loop, formatter, bert, gpt, ChatOpenAI(model='gpt-3.5-turbo', temperature=0), StrOutputParser())
channel_analyzer = ChannelAnalyzer(bot, loop, formatter, bert)

'''Helper function'''

def helper(message):
    markup = types.InlineKeyboardMarkup(row_width=1)
    summarize_button = types.InlineKeyboardButton('Summarize a Channel 📝', callback_data='summarize')
    explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
    analyze = types.InlineKeyboardButton('Analyze a Channel 🔍', callback_data='analyze')
    go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
    help_text = (
        "Here's how you can use Detoxigram:\n\n"
        "1. **Analyze a Channel:** Tap the 'Analyze a Channel 🔍' button and provide the @ChannelName to analyze the toxicity of the channel's messages.\n\n"
        "2. **Summarize a Channel:** Tap the 'Summarize a Channel 📝' button and provide the @ChannelName to get a summary of the channel's messages and an evaluation of its toxicity level.\n\n"
        "3. **Understanding Toxicity Levels:** Tap the 'Explain me toxicity levels 👀' button to learn more about the different levels of toxicity.\n\n"
        "If you have any questions or need further assistance, feel free to ask!"
        "You can reach us out at malbaposse@mail.utdt.edu\n\n"
    )
    markup.add(analyze, summarize_button, explainer, go_back)
    bot.reply_to(message, help_text)


'''Handlers'''
@bot.message_handler(commands=['start'])
def send_welcome(message):
    username = message.from_user.first_name
    markup = types.InlineKeyboardMarkup(row_width=1)
    analyze = types.InlineKeyboardButton('Analyze a Channel 🔍', callback_data='analyze')
    summarize = types.InlineKeyboardButton('Summarize a Channel 📝', callback_data='summarize')
    help = types.InlineKeyboardButton('Help, please! 🛟', callback_data='help')
    markup.add(analyze, summarize, help)
    bot.send_message(message.chat.id, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n''', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def answer(callback):
    markup = types.InlineKeyboardMarkup(row_width=1)
    summarize_button = types.InlineKeyboardButton('Summarize a Channel 📝', callback_data='summarize')
    explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
    analyze = types.InlineKeyboardButton('Analyze a Channel 🔍', callback_data='analyze')
    go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
    new_analyze = types.InlineKeyboardButton('Analyze another Channel 🔍', callback_data='analyze')

    if callback.message:
        if callback.data == 'analyze':
            bot.send_message(callback.message.chat.id, "Great! Just for you to know, when we evaluate the toxicity, we'll only consider the last 50 messages of the channel ⚠️ Now, please provide the @ChannelName you would like to analyze 🤓")
            bot.register_next_step_handler(callback.message, channel_analyzer.analyze_channel_bert) 
        elif callback.data == 'summarize':
            bot.register_next_step_handler(callback.message, summarizer.summarize) 
        elif callback.data == 'help':
            helper(callback.message)
        elif callback.data == 'end':
            bot.send_message(callback.message.chat.id, "Goodbye! 👋 If you need anything else, just type /start")
        elif callback.data == 'restart':
            markup.add(analyze, summarize_button, explainer)
            bot.send_message(callback.message.chat.id, "Ok! Let's see, what would you like to do now? 🤔")
        elif callback.data == 'explain':
            toxicity_level = last_analyzed_toxicity.get(callback.message.chat.id, None)
            if toxicity_level is not None:
                if 0 <= toxicity_level < 0.99:
                    explanation = "🟢 Non-toxic: Messages that foster a positive, respectful, and inclusive atmosphere, promoting kindness and mutual understanding. They value diverse opinions and contribute to constructive dialogue without offensive content. ✨\n"
                elif 0.99 <= toxicity_level < 1.5:
                    explanation = "🟡 Slightly Toxic: Messages that, while mostly respectful, may include criticism or disagreements in a passive-aggressive manner. They lack full appreciation for differing viewpoints but don't directly attack individuals or groups. 🤔\n"
                elif 1.5 <= toxicity_level < 2.5:
                    explanation = "🟡 Moderately Toxic: Messages with an aggressive or disrespectful tone, often containing sarcasm, irony, or derogatory language towards specific groups. They seek to hurt, ridicule, or belittle others, showing a rejection of diversity. 😠\n\n"
                elif 2.5 <= toxicity_level < 3.2:
                    explanation = "🔴 Highly Toxic: Messages demonstrating clear rejection and contempt for individuals or groups, using insults, degrading language, or references based on gender, ethnicity, etc. They aim to intimidate, exclude, or incite hatred, with explicit intent to cause harm. 😡\n"
                elif 3.2 <= toxicity_level < 4:
                    explanation = "🔴 Extremely Toxic: Messages explicitly aggressive and disrespectful, containing threats or calls to violent action. They promote hostility, incite hatred, and suggest harmful consequences, endangering the safety and well-being of others. ⚠️ \n 😡\n"

                message_text = explanation + "\nUnderstanding these levels helps maintain a healthy and respectful online environment. Let's strive for constructive dialogue and mutual respect. 🌐💬"
                markup.add(new_analyze, summarize_button, go_back)
                bot.send_message(callback.message.chat.id, message_text)
            else:
                bot.send_message(callback.message.chat.id, "No toxicity level has been analyzed yet for this chat! ")

bot.infinity_polling()

