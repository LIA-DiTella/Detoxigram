import sys
sys.path.append('..')
import time
import asyncio
import os
import re
import requests
from dotenv import main
import telebot
from telebot import types
from telebot.types import Message, CallbackQuery
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from langchain_core.output_parsers import StrOutputParser
from requests.exceptions import ReadTimeout, ConnectionError
from transformers import pipeline
from toxicity.Explainer import Explainer
from toxicity.Analyzer import Analyzer
from toxicity.Detoxifier import Detoxifier
from toxicity.Dataviz import ToxicityDataviz
from user_management.Detoxigramer import Telegram_Detoxigramer

# Variables de Ambiente

BOT_TOKEN:str = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM:str = os.environ.get('API_ID')
API_HASH_TELEGRAM:str = os.environ.get('API_HASH')
MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']

# Modelos

hatebert:hate_bert_classifier = hate_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert', verbosity=True)
multibert:multi_bert_classifier = multi_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/multibert', verbosity=True, toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/multibert_distribution.json',calculate_toxicity_distribution=False)
mistral:mistral_classifier = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', verbosity=True, toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/mistral_distribution.json', calculate_toxicity_distribution=False)

# Cliente de Telegram

client = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  


# Clases

explainer = Explainer()
analyzer = Analyzer()
detoxifier = Detoxifier()
dataviz = ToxicityDataviz()
detoxigramer = Telegram_Detoxigramer()


