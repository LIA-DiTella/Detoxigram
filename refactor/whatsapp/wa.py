#@Make it work, then make it pretty. Pendiente de refactor

import logging
from typing import Literal
from fastapi import FastAPI, Request, HTTPException
from pywa import WhatsApp, filters
from pywa.types import Message, CallbackButton, Button, Document
from dotenv import main
import os 
from toxicity.Analyzer import Analyzer
from toxicity.Detoxifier import Detoxifier
from toxicity.Explainer import Explainer
from toxicity.Dataviz import ToxicityDataviz
from user_management.Detoxigramer import Detoxigramer
from user_management.ManagementDetoxigramers import ManagementDetoxigramers
from utilities import Utilities
from refactor.messager.messager import WhatsApp_Messager
from refactor.messager.messages import MESSAGES, BUTTONS

# Inicializo clases auxiliares
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fastapi_app = FastAPI()
main.load_dotenv()
utils = Utilities()
user = Detoxigramer()


# Variables de Ambiente
PHONE_ID = os.environ.get('PHONE_ID')
TOKEN_WPP = os.environ.get('TOKEN_WPP')
CALLBACK_URL = os.environ.get('CALLBACK_URL')
VERIFY_TOKEN = os.environ['VERIFY_TOKEN']
APP_ID = os.environ['APP_ID']
APP_SECRET = os.environ['APP_SECRET']
TESTING_NUMBER = os.environ['TESTING_NUMBER']

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

# Primero detectamos el idioma en el que nos estan hablando.
@wa.on_message()
def greeting(client: WhatsApp, msg: Message):
    user.global_language = utils.language_detection(msg.text)
    Greet = utils.greeting_detection(msg.text)
    logger.info(f"Received message: {msg.text}")

    if user.global_language == "ES":
        if Greet != "GREETING":
            messager.send_message(MESSAGES['NO_GREETING_SP'])
        else: 
            messager.send_message_with_buttons(MESSAGES['GREETING_SP'].format(name=msg.from_user.name), TESTING_NUMBER, BUTTONS['GREETING_ES'])

    if user.global_language == "EN":
        if Greet != "GREETING":
            messager.send_message(MESSAGES['NO_GREETING_EN'])
        else: 
            messager.send_message_with_buttons(MESSAGES['GREETING_SP'].format(name=msg.from_user.name), TESTING_NUMBER, BUTTONS['GREETING_EN'])

@wa.on_callback_button(filters.startswith("id"))
def click_me(client: WhatsApp, clb: CallbackButton):
    if user.global_language == "ES":
        if clb.data == "id:000":
            messager.send_message(MESSAGES["WAITING_FOR_MSG_ES"])
        elif clb.data == "id:001":
            messager.send_message(MESSAGES["WAITING_FOR_FILE_ES"])
    else:
        if clb.data == "id:000":
            messager.send_message(MESSAGES["WAITING_FOR_MSG_EN"])
        elif clb.data == "id:001":
            messager.send_message(MESSAGES["WAITING_FOR_FILE_EN"])
        

@wa.on_message(filters.regex(".*")) 
def handle_user_response(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    if user_state.get(msg.from_user) == "waiting_for_message":
        user_response_content = msg.txt  
        logger.info(f"User response saved: {user_response_content}")

        wa.send_text(
            to=TESTING_NUMBER, 
            text="Analizando la toxicidad del mensaje..."
        )
        user_state[msg.from_user] = "idle"


@wa.on_message(filters.document)  
def handle_user_file(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    if user_state.get(msg.from_user) == "waiting_for_file" and msg.document.mime_type == "text/plain":
        document_url = msg.document.get_media_url()
        user_response_content = document_url 
        logger.info(f"User file saved: {user_response_content}")

        wa.send_text(
            to=TESTING_NUMBER, 
            text="Analizando la toxicidad de la conversación..."
        )
        user_state[msg.from_user] = "idle"

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