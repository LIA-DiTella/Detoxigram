#@Make it work, then make it pretty. Pendiente de refactor

import logging
from typing import Literal
from fastapi import FastAPI, Request, HTTPException
from pywa import WhatsApp, filters
from pywa.types import Message, CallbackButton, Button, Document
from dotenv import main, load_dotenv
load_dotenv()
import sys
import os 
sys.path.append('..')
from toxicity.Analyzer import WhatsApp_Analyzer
from toxicity.Detoxifier import Whatsapp_Detoxifier
from toxicity.Explainer import Whatsapp_Explainer
from toxicity.Dataviz import ToxicityDataviz
from user_management.Detoxigramer import WhatsApp_Detoxigramer
from user_management.ManagementDetoxigramers import ManagementDetoxigramers_Whatsapp
from utilities.Utilities import Utilities
from utilities.Fetcher import WhatsApp_Fetcher
from messager.messager import WhatsApp_Messager
from messager.messages import MESSAGES, BUTTONS
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from langchain_core.output_parsers import StrOutputParser
from random import randint
# Variables de Ambiente
PHONE_ID = os.environ.get('PHONE_ID')
TOKEN_WPP = os.environ.get('TOKEN_WPP')
CALLBACK_URL = os.environ.get('CALLBACK_URL')
VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')
APP_ID = os.environ.get('APP_ID')
APP_SECRET = os.environ.get('APP_SECRET')
TESTING_NUMBER = os.environ.get('TESTING_NUMBER')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')


# Inicializo clases auxiliares

management_detoxigramers = ManagementDetoxigramers_Whatsapp()
hatebert = hate_bert_classifier('tomh/toxigen_hatebert', verbosity=True)
multibert = multi_bert_classifier(
    '../model_evaluation_scripts/classifiers_classes_api/multibert',
    verbosity=True,
    toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/multibert_distribution.json',
    calculate_toxicity_distribution=False
)
mistral = mistral_classifier(
    mistral_api_key=MISTRAL_API_KEY,
    templatetype='prompt_template_few_shot',
    toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/mistral_distribution.json',
    calculate_toxicity_distribution=False,
    verbosity=True
)

fastapi_app = FastAPI()
main.load_dotenv()
utils = Utilities()
str_parser = StrOutputParser()
analyzer = WhatsApp_Analyzer(hatebert, mistral, management_detoxigramers)
detoxifier = Whatsapp_Detoxifier(mistral,str_parser, management_detoxigramers, analyzer)
fetcher = WhatsApp_Fetcher()

print(f"PHONE_ID: {PHONE_ID}")
print(f"TOKEN_WPP: {TOKEN_WPP}")
print(f"CALLBACK_URL: {CALLBACK_URL}")
print(f"VERIFY_TOKEN: {VERIFY_TOKEN}")

# Inicializamos el client de WhatsApp
wa = WhatsApp(
    phone_id=PHONE_ID,
    token=TOKEN_WPP,
    server=fastapi_app,
    callback_url=CALLBACK_URL,
    verify_token=VERIFY_TOKEN,
    app_id=APP_ID,
    app_secret=APP_SECRET
)



# Incializamos clase para mandar mensajes más facil
messager = WhatsApp_Messager(wa)

@wa.on_message()
def greeting(client: WhatsApp, msg: Message):
    user_id = msg.from_user.wa_id
    detoxigramer = WhatsApp_Detoxigramer()
    management_detoxigramers.set_detoxigramer(user_id, detoxigramer)
    user = management_detoxigramers.get_detoxigramer(user_id)

    Greet = utils.greeting_detection(msg.text)
    
    # Send greeting based on the user's language
    if user.global_language == "ES":
        if Greet != "GREETING":
            messager.send_message(MESSAGES['NO_GREETING_SP'])
        else: 
            messager.send_message_with_buttons(MESSAGES['GREETING_SP'].format(name=msg.from_user.name), TESTING_NUMBER, BUTTONS['GREETING_ES'])

    elif user.global_language == "EN":
        if Greet != "GREETING":
            messager.send_message(MESSAGES['NO_GREETING_EN'])
        else: 
            messager.send_message_with_buttons(MESSAGES['GREETING_SP'].format(name=msg.from_user.name), TESTING_NUMBER, BUTTONS['GREETING_EN'])

@wa.on_callback_button(filters.startswith("id"))
def click_me(client: WhatsApp, clb: CallbackButton):
    conversation_id = 0
    user_id = clb.from_user.wa_id
    user = management_detoxigramers.get(user_id)

    if user.global_language == "ES":
        if clb.data == "id:000":
            messager.send_message(MESSAGES["WAITING_FOR_MSG_ES"])
            user.set_status('DETOX')
        elif clb.data == "id:001":
            messager.send_message(MESSAGES["WAITING_FOR_FILE_ES"])
            user.set_status('ANALIZE')
        elif clb.data == "id:002":
            output = Whatsapp_Explainer.explain_es(user.store_conversation, conversation_id)
            user.send_message(output)
        elif clb.data == "id:003":
            user.send_message("distribución!!")
        
    else:
        if clb.data == "id:000":
            messager.send_message(MESSAGES["WAITING_FOR_MSG_EN"])
            user.set_status('DETOX')
        elif clb.data == "id:001":
            messager.send_message(MESSAGES["WAITING_FOR_FILE_EN"])
            user.set_status('ANALIZE')
        elif clb.data == "id:002":
            output = Whatsapp_Explainer.explain_en(user.store_conversation,user.id*(len(user.store_conversation[0])+len(user.store_conversation[1])+len(user.store_conversation[2])))
            user.send_message(output)
        elif clb.data == "id:003":
            user.send_message("distribución!!")

@wa.on_message(filters.regex(".*")) 
def handle_user_response(client: WhatsApp, msg: Message):
    user_id = msg.from_user.wa_id
    user = management_detoxigramers.get(user_id)

    if user.status == 'DETOX':
        if utils.language_detection(msg.text) == "ES":
            msg_detoxified = detoxifier.detoxify_single_message_es(msg.text)
            messager.send_message(msg_detoxified)
        elif utils.language_detection(msg.text) == "EN":
            msg_detoxified = detoxifier.detoxify_single_message_en(msg.text)
            messager.send_message(msg_detoxified)

@wa.on_message(filters.document)  
def handle_user_file(client: WhatsApp, msg: Message):
    user_id = msg.from_user.wa_id
    user = management_detoxigramers.get(user_id)
    
    if user.status == 'ANALIZE':
        document_url = msg.document.get_media_url()
        conversation = fetcher.fetch(document_url)
        analisis = analyzer.conversation_classifier(str(randint()), conversation)
        user.messages_per_conversation[user.id * (len(conversation[0]) + len(conversation[1]) + len(conversation[2]))] = conversation
        user.store_conversation = conversation

        if user.global_language == 'ES':
            resp = "La conversacion que enviaste resulto ser " + analisis + "."
            messager.send_message(resp)
            messager.send_message_with_buttons(MESSAGES['POST_ANALISIS_ES'], BUTTONS['POST_ANALISIS_ES'])
        elif user.global_language == 'EN':
            resp = "The conversation you sent appears to be " + analisis + "."
            messager.send_message(resp)
            messager.send_message_with_buttons(MESSAGES['POST_ANALISIS_EN'], BUTTONS['POST_ANALISIS_EN'])


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