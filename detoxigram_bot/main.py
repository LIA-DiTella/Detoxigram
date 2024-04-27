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
from bot_functions.formater import formater
from bot_functions.explainer import explainer
from bot_functions.group_toxicity_distribution.group_toxicity_distribution import group_toxicity_distribution
from detoxigram_bot.bot_functions.channel_analyzer import channel_analyzer
from detoxigram_bot.bot_functions.user_management import user_management
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from requests.exceptions import ReadTimeout, ConnectionError

main.load_dotenv()
print(os.getcwd())

BOT_TOKEN:str = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM:str = os.environ.get('API_ID')
API_HASH_TELEGRAM:str = os.environ.get('API_HASH')
OPENAI_API_KEY:str = os.environ['OPENAI_API_KEY']
GOOGLE_CLOUD_API_KEY:str = os.environ['GOOGLE_CLOUD_API_KEY']
MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']

bert:hate_bert_classifier = hate_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert', verbosity=True)
multibert:multi_bert_classifier = multi_bert_classifier('../model_evaluation_scripts/classifiers_classes_api/multibert', verbosity=True, toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/multibert_distribution.json',calculate_toxicity_distribution=False)
mistral:mistral_classifier = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', verbosity=True)

client:TelegramClient = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  
group_toxicity_distribution_:group_toxicity_distribution = group_toxicity_distribution()
formatter:formater = formater(client)
user_manage:user_management = user_management()
loop:asyncio = asyncio.new_event_loop()  

bot:telebot = telebot.TeleBot(BOT_TOKEN)
if bot:  # Check if the bot is running
    print('Bot token loaded, Detoxigram is live 🚀')

greetings = ['holis', 'holaaaa', 'hey', 'hi', 'hello', 'good morning', 'hellooo', 'good afternoon', 'good evening', 'good night', 'good day', r'ho+la+', r'he+llo+', r'he+y+', r'a+lo+', r'ho+li+', r'ho+lo+', r'ho+li+']

markup:types = types.InlineKeyboardMarkup(row_width=1)
explanation:types = types.InlineKeyboardButton('Explain me 👀', callback_data='explainer')
analyze:types = types.InlineKeyboardButton('Analyze a channel 🔍', callback_data='analyze')
go_back:types = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
help:types = types.InlineKeyboardButton('What can I do? 🛟', callback_data='help')
more:types = types.InlineKeyboardButton('More options 👇', callback_data='more')
detoxify:types = types.InlineKeyboardButton('Detoxify a message 📩', callback_data='detoxify')

async def main():

    retry_delays = [10, 20, 40, 80] 
    attempt = 0

    channel_analyzer_:channel_analyzer = channel_analyzer(bot, loop, formatter, multibert=multibert, mistral=mistral, user_management=user_manage)
    explainer_:explainer = explainer(bot, loop, formatter, mistral, bert,  StrOutputParser(), user_management = user_manage)
    '''Handlers'''

    @bot.message_handler(func=lambda message: message.text is not None and (re.search(r'ho+la+', message.text.lower()) or any(greeting in message.text.lower() for greeting in greetings) or (re.search(r'he+llo+', message.text.lower())) or (re.search(r'he+y+', message.text.lower()) )))
    def handle_greeting(message):
        username = message.from_user.first_name
        markup_start_hello = types.InlineKeyboardMarkup(row_width=1)
        markup_start_hello.add(analyze, detoxify, help)
        bot.reply_to(message, '''Hello {username} and welcome to Detoxigram! 👋 \n
I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n
What would you like to do?'''.format(username = username), reply_markup=markup_start_hello)

    @bot.message_handler(commands=['start'])
    def send_welcome(message:Message) -> None:
        username = message.from_user.first_name
        markup_start:types = types.InlineKeyboardMarkup(row_width=1)
        markup_start.add(analyze, help, more)
        bot.send_message(message.chat.id, '''Hello {username} and welcome to Detoxigram! 👋 \n
I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n
What would you like to do?'''.format(username = username), reply_markup=markup_start)

    @bot.callback_query_handler(func=lambda call: True)
    def answer(callback:CallbackQuery) -> None:
        if callback.message:
            user_id = callback.message.chat.id
            state = user_manage.get_user_state(user_id) 
            if callback.data == 'analyze':
                bot.send_message(callback.message.chat.id, '''Great! \n 
Just for you to know, when we evaluate the toxicity, we'll only consider the last 50 messages of the channel ⚠️\n 
Now, please provide the @ChannelName or the invite link of the channel you would like to analyze 🤓''')
                bot.register_next_step_handler(callback.message, channel_analyzer_.channel_classifier)
            
            elif callback.data == 'explainer':
                print(state.last_channel_analyzed)
                if state.last_channel_analyzed != None: 
                    bot.send_message(callback.message.chat.id, "After evaluating the content of {last_channel_analyzed}, we saw that this channel is {last_toxicity}. Now I will explain you why, it will take a few seconds 🕣".format(last_channel_analyzed = state.last_channel_analyzed, last_toxicity = state.last_toxicity ))                    
                    
                    explainer_.explain(callback.message)
                else:
                    bot.send_message(callback.message.chat.id, "I'm sorry, I don't have any channel to explain. Please analyze a channel first!")


            elif callback.data == 'help':
                help_text:str = (
                    "Welcome to Detoxigram! 🌟 Here's how you can use our bot to make your Telegram experience safer:\n\n"
                    "1. **Analyze a Channel:** To start analyzing a channel for toxic content, simply tap the 'Analyze a Channel 🔍' button. Then, enter the @ChannelName or send the invite to the channel. I'll check the last 50 messages and let you know how toxic the conversations are.\n\n"
                    "2. **Explain Toxicity:** Curious about why a channel was rated a certain way? Tap the 'Explain why 📝' button after analyzing a channel. I'll provide you with a summary of the channel's messages, highlighting specific examples of toxicity. This helps you understand the context and specifics of the content I've analyzed.\n\n"
                    "3. **Detoxify a Message:** Want to clean up a specific message? Use the 'Detoxify a message📩' option to send me a message you think is problematic. I'll offer a less toxic version, providing a cleaner, more respectful alternative.\n\n"
                    "Need more help or have any questions? Don't hesitate to reach out. You can contact us directly at malbaposse@mail.utdt.edu. We're here to help make your digital spaces safer! 🛡️\n\n"
                )

                help_markup = markup.add(analyze, explainer_, go_back, more)
                bot.send_message(help_text, reply_markup=help_markup, parse_mode='Markdown')

            elif callback.data == 'learn_more':

                bot.send_message(callback.message.chat.id, "So... we've just classified the channel you sent. But, how does its toxicity stack up against the others we've analyzed?")

                toxicity_vector = multibert.get_group_toxicity_distribution(state.last_chunk_of_messages)
                
                keys_order = ['sarcastic', 'antagonize', 'generalisation', 'dismissive']
                
                toxicity_vector = [toxicity_vector[key] for key in keys_order]

                toxicity_graphic = group_toxicity_distribution_.get_toxicity_graph(state.last_channel_analyzed, toxicity_vector)

                if os.path.exists(toxicity_graphic) and os.access(toxicity_graphic, os.R_OK):
                    markupLearnMore = types.InlineKeyboardMarkup(row_width=1)
                    markupLearnMore.add(explanation, go_back)
                    bot.send_photo(callback.message.chat.id, open(toxicity_graphic, 'rb'))
                else:
                    bot.send_message(callback.message.chat.id, "Oh! Something went wrong... I couldn't get the toxicity distribution of the channel 😔")
                
                os.remove(toxicity_graphic)
            
            elif callback.data == 'detoxify':
                bot.send_message(callback.message.chat.id, "Great! ⚠️ Now, please write a message you would like to detoxify 🤓")
                bot.register_next_step_handler(callback.message, explainer_.detoxify_single_message)
            
            elif callback.data == 'more':
                more_markup = types.InlineKeyboardMarkup(row_width=1)
                more_markup.add(detoxify, go_back)
                bot.send_message(callback.message.chat.id, "Here are some more options! 🤓", reply_markup=more_markup)
            
            elif callback.data == 'restart':
                restart_markup = types.InlineKeyboardMarkup(row_width=1)
                restart_markup.add(analyze, more)
                bot.send_message(callback.message.chat.id, "Ok! Let's see, what would you like to do now? 🤔", reply_markup=restart_markup)

            elif callback.data == 'end' or callback.data == 'exit' or callback.data == 'goodbye' or callback.data == 'bye' or callback.data == 'stop' or callback.data == 'cancel' or callback.data == 'quit':
                bot.send_message(callback.message.chat.id, "Goodbye! 👋 If you need anything else, just say hi!")
    
    while True:
        try:
            await bot.polling(none_stop=True)
        except ReadTimeout:
            if attempt < len(retry_delays):
                time_to_wait = retry_delays[attempt]
                print(f"Request timed out. Retrying in {time_to_wait} seconds.")
                await asyncio.sleep(time_to_wait)
                attempt += 1
            else:
                print("Request failed after maximum attempts. Retrying again after a delay.")
                attempt = 0
                await asyncio.sleep(retry_delays[-1]) 
        except ConnectionError:
            print("Connection lost... retrying in 5 seconds")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}.≥ Retrying in 10 seconds...")
            await asyncio.sleep(10)

if __name__ == '__main__':
    
    asyncio.run(main())

