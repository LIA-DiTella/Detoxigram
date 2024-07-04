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
from transformers import pipeline
import json


main.load_dotenv()
print(os.getcwd())

BOT_TOKEN:str = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM:str = os.environ.get('API_ID')
API_HASH_TELEGRAM:str = os.environ.get('API_HASH')
MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']

hatebert:hate_bert_classifier = hate_bert_classifier('../classifiers/classifiers_classes_api/toxigen_hatebert', verbosity=True)
multibert:multi_bert_classifier = multi_bert_classifier('../classifiers/classifiers_classes_api/multibert', verbosity=True, toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/multibert_distribution.json',calculate_toxicity_distribution=False)
mistral:mistral_classifier = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', verbosity=True, toxicity_distribution_path='../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/mistral_distribution.json', calculate_toxicity_distribution=False)

client:TelegramClient = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  
group_toxicity_distribution_:group_toxicity_distribution = group_toxicity_distribution()
formatter:formater = formater(client)
user_manage:user_management = user_management()
loop:asyncio = asyncio.new_event_loop()  

bot:telebot = telebot.TeleBot(BOT_TOKEN)
if bot:  # Check if the bot is running
    print('Bot token loaded, Detoxigram is live 🚀')

classifier = pipeline("text-classification", model="Falconsai/intent_classification")

def is_greeting(message):
    greetings = ['holis', 'hola', 'hey', r'ho+la+', r'he+llo+', r'he+y+', r'a+lo+', r'ho+li+', r'ho+li+']
    result = classifier(message)
    return (result[0]['label'] == 'speak to person' or any(greeting in message.lower() for greeting in greetings))

markup:types = types.InlineKeyboardMarkup(row_width=1)
explanation:types = types.InlineKeyboardButton('Explain me 👀', callback_data='explainer')
analyze:types = types.InlineKeyboardButton('Analyze a channel 🔍', callback_data='analyze')
go_back:types = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
help:types = types.InlineKeyboardButton('What can I do? 🛟', callback_data='help')
more:types = types.InlineKeyboardButton('More options 👇', callback_data='more')
detoxify:types = types.InlineKeyboardButton('Detoxify a message 📩', callback_data='detoxify')


def write_cache(user_id, channel_name, toxicity_vector, cache_dir=os.path.dirname(os.path.abspath(__file__))):
        cache_file_path = os.path.join(cache_dir, "channel_analyzer_cache.json")
        try:
            with open(cache_file_path, 'r') as cache_file:
                cache = json.load(cache_file)
        except FileNotFoundError:
            cache = {}

        cache[channel_name] = {'channel_name': channel_name, 'toxicity vector': toxicity_vector}

        with open(cache_file_path, 'w') as cache_file:
            json.dump(cache, cache_file)
            print("Cache updated successfully!")

async def start_polling(bot, retry_delays):
    attempt = 0
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

async def main():

    retry_delays = [10, 20, 40, 80] 

    channel_analyzer_:channel_analyzer = channel_analyzer(bot, loop, formatter, hatebert=hatebert, mistral=mistral, user_management=user_manage)
    explainer_:explainer = explainer(bot, loop, formatter, mistral, hatebert,  StrOutputParser(), user_management = user_manage)
    
    @bot.message_handler(func = lambda message: message.text == "/testing")
    def activate_testing(message):
        user_id = message.chat.id
        state = user_manage.get_user_state(user_id)
        state.is_testing = not state.is_testing

        if (state.is_testing): bot.reply_to(message, f"Entering testing mode. I will now output some internal information, and i will now work on downloaded channels.")
        else: bot.reply_to(message, f"Leaving testing mode.")

    @bot.message_handler(func=lambda message: (is_greeting(message.text) is True) or message.text == '/start') 
    def handle_greeting(message):
        username = message.from_user.first_name
        markup_start_hello = types.InlineKeyboardMarkup(row_width=1)
        markup_start_hello.add(analyze, detoxify, help)
        bot.reply_to(message, '''Hello {username} and welcome to Detoxigram! 👋\n
I'm here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n
What would you like to do?
'''.format(username = username), reply_markup=markup_start_hello)
    
    @bot.message_handler(func=lambda message: (message.text is not None and message.text.startswith('/end')) or (message.text in r'bye+'))
    def handle_goodbye(message):
        state = user_manage.get_user_state(message.chat.id)
        state.is_analyzing = False
        state.is_explaining = False
        state.is_toxicity_distribution = False
        state.is_detoxifying = False
        bot.reply_to(message, "Goodbye! 👋 If you need anything else, just say hi!")
    
    @bot.message_handler(func=lambda message: message.text)
    def handle_random_message(message):
        username = message.from_user.first_name
        markup = types.InlineKeyboardMarkup(row_width=1)
        markup.add(analyze, detoxify, help)
        bot.send_message(message.chat.id, "Mmm... I'm not sure what that means, {username}. Would any of these options be helpful? 😁".format(username = username), reply_markup=markup)
    
    @bot.callback_query_handler(func=lambda call: True)
    def answer(callback:CallbackQuery) -> None:
        if callback.message:
            user_id = callback.message.chat.id
            state = user_manage.get_user_state(user_id) 

            if callback.data == 'analyze' and state.is_analyzing == False:
                if state.is_explaining == True or state.is_analyzing == True or state.is_toxicity_distribution == True or state.is_detoxifying == True:
                    bot.send_message(callback.message.chat.id, "I'm sorry, I'm still working on your last request! 🕣")
                else:
                    state.is_analyzing = True
                    bot.send_message(callback.message.chat.id, '''Great!\n\n
Just so you know, when we evaluate the toxicity, we'll only consider the last 50 messages of the channel ⚠️\n\n
Now, please provide the @ChannelName you would like to analyze 🤓''')
                    bot.register_next_step_handler(callback.message, channel_analyzer_.channel_classifier)
                
            elif callback.data == 'explainer' and state.is_explaining == False:
                                if state.is_analyzing == True or state.is_toxicity_distribution == True or state.is_detoxifying == True:
                                    bot.send_message(callback.message.chat.id, "I'm sorry, I'm still working on your last request! 🕣")
                                else:
                                    state.is_explaining = True
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

            elif callback.data == 'learn_more' and state.is_toxicity_distribution == False:
                    if state.is_explaining == True or state.is_analyzing == True or state.is_detoxifying == True:
                        bot.send_message(callback.message.chat.id, "I'm sorry, I'm still working on your last request! 🕣")
                    else: 
                        state.is_toxicity_distribution = True

                        bot.send_message(callback.message.chat.id, "We've just classified the channel you sent. I will send you a more detailed analysis of the channel shortly 📊")

                        toxicity_vector = multibert.get_group_toxicity_distribution(state.last_chunk_of_messages)
                        
                        write_cache(user_id, state.last_channel_analyzed, toxicity_vector)
                        
                        keys_order = ['sarcastic', 'antagonize', 'generalisation', 'dismissive']

                        toxicity_vector = [toxicity_vector[key] for key in keys_order]

                        toxicity_graphic = group_toxicity_distribution_.get_toxicity_graph(state.last_channel_analyzed, toxicity_vector, state.last_analyzed_toxicity)

                        if os.path.exists(toxicity_graphic) and os.access(toxicity_graphic, os.R_OK):
                            markupLearnMore = types.InlineKeyboardMarkup(row_width=1)
                            markupLearnMore.add(explanation, go_back)
                            bot.send_photo(callback.message.chat.id, open(toxicity_graphic, 'rb'), reply_markup=markupLearnMore)
                            state.is_toxicity_distribution = False

                        else:

                            bot.send_message(callback.message.chat.id, "Oops! Something went wrong... I couldn't get the toxicity distribution of the channel 😔 Why don't we try again?")
                            markupLearnMore = types.InlineKeyboardMarkup(row_width=1)
                            markupLearnMore.add(go_back)
                            state.is_toxicity_distribution = False
                        os.remove(toxicity_graphic)

            elif callback.data == 'detoxify' and state.is_detoxifying == False:
                if state.is_explaining == True or state.is_analyzing == True or state.is_toxicity_distribution == True:
                    bot.send_message(callback.message.chat.id, "I'm sorry, I'm still working on your last request! 🕣")
                else:
                    state.is_detoxifying = True
                    bot.send_message(callback.message.chat.id, "Great! ⚠️ Now, please write a message you would like to detoxify 🤓")
                    bot.register_next_step_handler(callback.message, explainer_.detoxify_single_message)
            
            elif callback.data == 'more':
                more_markup = types.InlineKeyboardMarkup(row_width=1)
                more_markup.add(detoxify, go_back)
                bot.send_message(callback.message.chat.id, "Here are some more options! 🤓", reply_markup=more_markup)
            
            elif callback.data == 'restart':
                state.is_analyzing = False
                state.is_explaining = False
                state.is_toxicity_distribution = False
                state.is_detoxifying = False

                restart_markup = types.InlineKeyboardMarkup(row_width=1)
                restart_markup.add(analyze, more)
                bot.send_message(callback.message.chat.id, "Alright! What would you like to do now? 🤔", reply_markup=restart_markup)
            elif (callback.data == 'analyze' or callback.data == 'explainer' or callback.data == 'learn_more' or callback.data == 'detoxify') and (state.is_analyzing == True or state.is_explaining == True or state.is_toxicity_distribution == True or state.is_detoxifying == True):
                bot.send_message(callback.message.chat.id, "I'm sorry, I'm still working on your last request! 🕣")
    await start_polling(bot, retry_delays)

if __name__ == '__main__':
    
    asyncio.run(main())


