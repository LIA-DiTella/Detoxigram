#@Make it work, then make it pretty. Pendiente de refactor

import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fastapi_app = FastAPI()
main.load_dotenv()

# Variables de Ambiente
PHONE_ID:str = os.environ.get('PHONE_ID')
TOKEN_WPP:str = os.environ.get('TOKEN_WPP')
CALLBACK_URL:str = os.environ.get('CALLBACK_URL')
VERIFY_TOKEN:str = os.environ['VERIFY_TOKEN']
APP_ID:str = os.environ['APP_ID']
APP_SECRET:str = os.environ['APP_SECRET']
TESTING_NUMBER:str = os.environ['TESTING_NUMBER']

# Inicializamos WhatsApp
wa = WhatsApp(
    phone_id=PHONE_ID,
    token=TOKEN_WPP,
    server=fastapi_app,
    callback_url=CALLBACK_URL,
    verify_token=VERIFY_TOKEN,
    app_id=APP_ID,
    app_secret=APP_SECRET
)

# user_response_content = ""
# user_state = {}

@wa.on_message(filters.startswith("Hi", "Hello", "Hola", "Holis", "Buenas", ignore_case=True))
def hello(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    logger.info(f"Received message: {msg.text}")

    user_state[msg.from_user] = "waiting_for_selection"

    wa.send_text(
        to=TESTING_NUMBER,  
        text=f'Hola {msg.from_user.name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus conversaciones, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
        buttons=[
                Button(title='Deto un mensaje 📧', callback_data='id:000'),
                Button(title='Analizar una conversación 💬', callback_data='id:001')
    ])

@wa.on_callback_button(filters.startswith("id"))
def click_me(client: WhatsApp, clb: CallbackButton):
    global user_response_content, user_state
    if clb.data == "id:000":
        client.send_message(
            to=TESTING_NUMBER,  
            text="Por favor, envíame el mensaje que deseas detoxificar."
        )
    elif clb.data == "id:002":
        user_state[clb.from_user] = "waiting_for_file"
        client.send_message(
            to=TESTING_NUMBER,
            text="Por favor, envíame el archivo .txt de la conversación que deseas analizar."
        )
        

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