
import sys
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
from bot_classes.Formater import Formatter
from bot_classes.Summarizer import Summarizer
from bot_classes.ChannelAnalyzer import ChannelAnalyzer

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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



bot.infinity_polling()

