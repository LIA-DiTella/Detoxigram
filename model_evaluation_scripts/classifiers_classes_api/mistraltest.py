import os
from dotenv import load_dotenv
load_dotenv()


import getpass
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

chat = ChatMistralAI(mistral_api_key= os.getenv('MISTRAL_API_KEY'))


messages = [HumanMessage(content="who are you?")]
print(chat.invoke(messages))

