import flask  # pip3 install flask
from pywa import WhatsApp
from pywa.types import Message

flask_app = flask.Flask(__name__)

wa = WhatsApp(
    phone_id='...',
    token='...',
    server=flask_app,
    verify_token='xyzxyz',
)

@wa.on_message()
def hello(_: WhatsApp, msg: Message):
    msg.react('👋')
    msg.reply(f'Hello {msg.from_user.name}!')

# Run the server
flask_app.run()
