import logging
from fastapi import FastAPI, Request, HTTPException
from pywa import WhatsApp, filters
from pywa.types import Message, CallbackButton, Button, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fastapi_app = FastAPI()

# Initialize WhatsApp
wa = WhatsApp(
    phone_id="388552330998694",
    token="EAAGRBKRIRZCsBOZBFcH56w32Cn4YsbHbGqkTeR2CWiBZBJ9qKPHzn3u0eyfezmWDS8fpKGjcHhrjbTUacJPZBZC7NYwZAvUJF4CywFZAOBjsZCDLBo5M3hvDAnaNh83hJhqdZCqdtfqZBozZCcJgxGsCcZBTuV9ZCnSbVlrzw8bBhWzMPQgPZA72T3YFANiy0yoDpmMqyWQmCBNHGJuPPwRTkRDXYZD",
    server=fastapi_app,
    callback_url="https://detoxigram.serveo.net",
    verify_token="xyz123",
    app_id=440924098807803,
    app_secret="9a2566e13a11988b4912dbce7bbeee46"
)

# Global variable to store the response content and user state
user_response_content = ""
user_state = {}

recipient = '+54111532986313' # Temporal

@wa.on_message(filters.startswith("Hi", "Hello", "Hola", "Holis", "Buenas", ignore_case=True))
def hello(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    logger.info(f"Received message: {msg.text}")

    # Reset user state
    user_state[msg.from_user] = "waiting_for_selection"

    # Send initial message with buttons
    wa.send_text(
        to=recipient,  # Use specific number for now
        text=f'Hola {msg.from_user.name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus mensajes y grupos, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
        buttons=[
                Button(title='Analizar un mensaje', callback_data='id:100'),
                Button(title='Analizar un grupo', callback_data='id:101')
    ])

@wa.on_callback_button(filters.startswith("id"))
def click_me(client: WhatsApp, clb: CallbackButton):
    global user_response_content, user_state
    if clb.data == "id:100":
        user_state[clb.from_user] = "waiting_for_message"
        client.send_message(
            to=recipient,  # Use specific number for now
            text="Por favor, envíame el mensaje que deseas detoxificar."
        )
    elif clb.data == "id:101":
        user_state[clb.from_user] = "waiting_for_file"
        client.send_message(
            to=recipient,  # Use specific number for now
            text="Por favor, envíame el archivo .txt de la conversación que deseas analizar."
        )

@wa.on_message(filters.regex(".*"))  # This captures all messages for analysis
def handle_user_response(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    if user_state.get(msg.from_user) == "waiting_for_message":
        user_response_content = msg.text  # Save the content of the response message
        logger.info(f"User response saved: {user_response_content}")

        # Optionally, respond to the user to confirm receipt of their message
        wa.send_text(
            to=recipient,  # Use specific number for now
            text="Analizando la toxicidad del mensaje..."
        )
        # Reset user state
        user_state[msg.from_user] = "idle"


@wa.on_message(filters.document)  # This captures incoming documents
def handle_user_file(client: WhatsApp, msg: Message):
    global user_response_content, user_state
    if user_state.get(msg.from_user) == "waiting_for_file" and msg.document.mime_type == "text/plain":
        document_url = msg.document.get_media_url()
        user_response_content = document_url  # Save the URL of the document
        logger.info(f"User file saved: {user_response_content}")

        # Optionally, respond to the user to confirm receipt of their file
        wa.send_text(
            to=recipient,  # Use specific number for now
            text="Analizando la toxicidad de la conversación..."
        )
        # Reset user state
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
