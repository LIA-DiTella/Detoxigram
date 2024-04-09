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
from model_evaluation_scripts.classifiers_classes_api.perspective_classifier import perspective_classifier
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from bot_functions.Formater import Formatter
from bot_functions.Explainer import Explainer
from bot_functions.group_toxicity_distribution.group_toxicity_distribution import GroupToxicityDistribution
from bot_functions.ChannelAnalyzer import ChannelAnalyzer
from bot_functions.UserManagement import UserManagement
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import numpy as np

main.load_dotenv()
print(os.getcwd())

BOT_TOKEN:str = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM:str = os.environ.get('API_ID')
API_HASH_TELEGRAM:str = os.environ.get('API_HASH')
OPENAI_API_KEY:str = os.environ['OPENAI_API_KEY']
GOOGLE_CLOUD_API_KEY:str = os.environ['GOOGLE_CLOUD_API_KEY']
MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']

bert:hate_bert_classifier = hate_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert', verbosity=True)
multibert:multi_bert_classifier = multi_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/multibert', verbosity=True)
mistral:mistral_classifier = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', verbosity=True)

client:TelegramClient = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  

group_toxicity_distribution:GroupToxicityDistribution = GroupToxicityDistribution()
formatter:Formatter = Formatter(client)
user_management:UserManagement = UserManagement()
loop:asyncio = asyncio.new_event_loop()  

bot:telebot = telebot.TeleBot(BOT_TOKEN)
if bot:  # Check if the bot is running
    print('Bot token loaded, Detoxigram is live 🚀')



greetings = ['holis', 'holaaaa', 'hey', 'hi', 'hello', 'good morning', 'hellooo', 'good afternoon', 'good evening', 'good night', 'good day', r'ho+la+', r'he+llo+', r'he+y+', r'a+lo+', r'ho+li+', r'ho+lo+', r'ho+li+']

markup:types = types.InlineKeyboardMarkup(row_width=1)
explanation:types = types.InlineKeyboardButton('Explain me 👀', callback_data='explainer')
analyze:types = types.InlineKeyboardButton('Analyze 🔍', callback_data='analyze')
go_back:types = types.InlineKeyboardButton('Restart! 🔄', callback_data='restart')
analyze:types = types.InlineKeyboardButton('Analyze 🔍', callback_data='analyze')
help:types = types.InlineKeyboardButton('Help 🛟', callback_data='help')
more:types = types.InlineKeyboardButton('Show more 👇', callback_data='more')
single_message:types = types.InlineKeyboardButton('Detoxify 📩', callback_data='single_message')

async def main():

    channel_analyzer:ChannelAnalyzer = ChannelAnalyzer(bot, loop, formatter, multibert=multibert, mistral=mistral, user_management=user_management)
    explainer:Explainer = Explainer(bot, loop, formatter, mistral, bert,  StrOutputParser(), user_management = user_management)
    '''Handlers'''
    #Esto está medio hardcodeado por ahora jaja
    @bot.message_handler(func=lambda message: message.text is not None and (re.search(r'ho+la+', message.text.lower()) or any(greeting in message.text.lower() for greeting in greetings) or (re.search(r'he+llo+', message.text.lower())) or (re.search(r'he+y+', message.text.lower()) )))
    def handle_greeting(message):
        username = message.from_user.first_name
        markup_start_hello = types.InlineKeyboardMarkup(row_width=1)
        markup_start_hello.add(analyze, help, more)
        bot.reply_to(message, f'''Hello {username} and welcome to Detoxigram! 👋 \n
    I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n
    What would you like to do?''', reply_markup=markup_start_hello)

    @bot.message_handler(commands=['start'])
    def send_welcome(message:Message) -> None:
        username = message.from_user.first_name
        markup_start:types = types.InlineKeyboardMarkup(row_width=1)
        markup_start.add(analyze, help, more)
        bot.send_message(message.chat.id, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n
What would you like to do?''', reply_markup=markup_start)

    @bot.callback_query_handler(func=lambda call: True)
    def answer(callback:CallbackQuery) -> None:
        if callback.message:
            user_id = callback.message.chat.id
            state = user_management.get_user_state(user_id) 
            if callback.data == 'analyze':
                bot.send_message(callback.message.chat.id, '''Great! \n 
Just for you to know, when we evaluate the toxicity, we'll only consider the last 50 messages of the channel ⚠️\n 
Now, please provide the @ChannelName you would like to analyze 🤓''')
                bot.register_next_step_handler(callback.message, channel_analyzer.ChannelClassifier)
            
            elif callback.data == 'explainer':
                print(state.last_channel_analyzed)
                if state.last_channel_analyzed != None: 
                    bot.send_message(callback.message.chat.id, f"Let's evaluate the content of {state.last_channel_analyzed}... We saw that the toxicity level of this channel is {state.last_analyzed_toxicity}, now I will explain you why. It may take a few seconds 🕣")
                    explainer.explain(callback.message)
                else:
                    bot.send_message(callback.message.chat.id, "I'm sorry, I don't have any channel to explain. Please analyze a channel first!")


            elif callback.data == 'help':
                help_text:str = (
                    "Here's how you can use Detoxigram:\n\n"
                    "1. **Analyze a Channel:** Tap the 'Analyze a Channel 🔍' button and provide the @ChannelName to analyze the toxicity of the channel's messages.\n\n"
                    "2. **Explain why** Tap the 'Explain why📝' button and provide the @ChannelName to get a summary of the channel's messages and an evaluation of its toxicity level.\n\n"
                    "If you have any questions or need further assistance, feel free to ask!"
                    "You can reach us out at malbaposse@mail.utdt.edu\n\n"
                )
                help_markup = markup.add(analyze, explainer, go_back, more)
                bot.send_message(help_text, reply_markup=help_markup, parse_mode='Markdown')

            elif callback.data == 'learn_more':
                bot.send_message(callback.message.chat.id, "So... we just clasified the channel you sent. But, how toxic is in in relation with other channels we have analized? Let's see...")
                toxicity_vector = multibert.get_group_toxicity_distribution(state.last_chunk_of_messages)
                
                #Hardcodeado, random mensajes en el medio
                random_number = np.random.randint(0, 20)
                if random_number < 10:
                    bot.send_message(callback.message.chat.id, "Mmm... it may contain quite sarcastic messages 😅")
                if random_number > 10 and random_number < 20:
                    bot.send_message(callback.message.chat.id, "Oh! Those are some strong messages haha")

                toxicity_graphic = group_toxicity_distribution.get_toxicity_graph(state.last_channel_analyzed, toxicity_vector)

                if os.path.exists(toxicity_graphic) and os.access(toxicity_graphic, os.R_OK):
                    bot.send_photo(callback.message.chat.id, open(toxicity_graphic, 'rb'))
                else:
                    bot.send_message(callback.message.chat.id, "Oh! Something went wrong... I couldn't get the toxicity distribution of the channel 😔")
                
                os.remove(toxicity_graphic)
            
            elif callback.data == 'single_message':
                bot.send_message(callback.message.chat.id, "Great! ⚠️ Now, please write a message you would like to detoxify 🤓")
                bot.register_next_step_handler(callback.message, explainer.detoxify_single_message)
            
            elif callback.data == 'more':
                more_markup = types.InlineKeyboardMarkup(row_width=1)
                more_markup.add(single_message, go_back)
                bot.send_message(callback.message.chat.id, "Here are some more options! 🤓", reply_markup=more_markup)
            
            elif callback.data == 'restart':
                restart_markup = types.InlineKeyboardMarkup(row_width=1)
                restart_markup.add(analyze, explanation, more)
                bot.send_message(callback.message.chat.id, "Ok! Let's see, what would you like to do now? 🤔", reply_markup=restart_markup)

            elif callback.data == 'end' or callback.data == 'exit' or callback.data == 'goodbye' or callback.data == 'bye' or callback.data == 'stop' or callback.data == 'cancel' or callback.data == 'quit':
                bot.send_message(callback.message.chat.id, "Goodbye! 👋 If you need anything else, just type /start")
    while True:
        try:
            await bot.polling()
        except requests.exceptions.ConnectionError:
            print("Connection lost... retrying")
            time.sleep(5)

if __name__ == '__main__':
    
    asyncio.run(main())

