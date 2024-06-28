import flask  # pip3 install flask
from pywa import WhatsApp
from pywa.types import Message

flask_app = flask.Flask(__name__)

wa = WhatsApp(
    phone_id='388552330998694',
    token='EAAGRBKRIRZCsBOyDNlxXOs7sJkf8MnYRVIdVuC4laEng4eIy2A3Ui6gNNFO2ZBYSNGx6nuZC4JRd5tNZASQlZBLldVZCZAfpt8XqXBr9iduZAlxxOflrxf4638WZAwDQXA4ntPhaVWGQ5OQgeQL1KL7U6U1anM5vWTo548Kp67SQnJBqH1bo0S5zZBo6ZBR8FOQ70bYZBGblVcprbHE6gDZAHIn4ZD',
    server=flask_app,
    verify_token='xyzxyz',
)

@wa.on_message()
def hello(_: WhatsApp, msg: Message):
    msg.react('👋')
    msg.reply(f'Hello {msg.from_user.name}!')

# Run the server
flask_app.run()