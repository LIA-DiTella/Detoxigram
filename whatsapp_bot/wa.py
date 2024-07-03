# import sys
# sys.path.append('..')
# import time
# import asyncio
# import os
# import re
# import requests
from pywa import WhatsApp, filters
from pywa.types import Message, CallbackButton, Button, CallbackData
from fastapi import FastAPI, Request
import logging
from pywa import filters as fil
from pywa.types import SectionList, Section, SectionRow

# from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
# from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
# from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
# from bot_functions.formater import formater
# from bot_functions.explainer import explainer
# from bot_functions.group_toxicity_distribution.group_toxicity_distribution import group_toxicity_distribution
# from detoxigram_bot.bot_functions.channel_analyzer import channel_analyzer
# from detoxigram_bot.bot_functions.user_management import user_management
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# import numpy as np
# from requests.exceptions import ReadTimeout, ConnectionError
# from transformers import pipeline
# import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fastapi_app = FastAPI()

#aca va wa

@wa.on_message(fil.startswith("Hi", "Hello", "Hola", "Holis", "Buenas", ignore_case=True))
def hello(client: WhatsApp, msg: Message):
    logger.info(f"Received message: {msg.text}")
    print("\n")
    print("\n")
    # msg.react("👋")
    wa.send_text(to=''
        , text=f'Hola {msg.from_user.name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus mensajes y grupos, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',buttons=[
        Button(title='Analize a message', callback_data='msj'),
        Button(title='Analize a group', callback_data='grp')
    ])

@wa.on_callback_button(filters.startswith("id"))
def click_me(client: WhatsApp, clb: CallbackButton):

    if clb.data == "id:100":
        client.send_message(
            chat_id=clb.chat_id,
            text="Por favor, envíame el mensaje que deseas detoxificar."
        )
    elif clb.data == "id:101":
        client.send_message(
            chat_id=clb.chat_id,
            text="Por favor, envíame la conversación que deseas analizar."
        )
    elif clb.data == "id:102":
        client.send_message(
            chat_id=clb.chat_id,
            text="Dame un momento, estoy calculando las dimensiones de toxicidad.")

@fastapi_app.get("/")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode == "subscribe" and token == wa.verify_token:
        return challenge
    else:
        raise HTTPException(status_code=403, detail="Invalid verification token")

if __name__ == "__main__":
    wa.run()
