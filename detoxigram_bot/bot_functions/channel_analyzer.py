import time
import telebot
from telebot import types
from urllib.parse import parse_qs, urlparse
import json 
import os
class channel_analyzer:

    def __init__(self, bot, loop, formatter, hatebert, mistral, user_management):

        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.hatebert = hatebert
        self.mistral = mistral
        self.user_management = user_management
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))

    def write_cache(self, chat_id, channel_name, average_toxicity_score):
        cache_file_path = os.path.join(self.cache_dir, "channel_analyzer_cache.json")
        try:
            with open(cache_file_path, 'r') as cache_file:
                cache = json.load(cache_file)
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

        cache[chat_id] = {
            'channel_name': channel_name,
            'average_toxicity_score': average_toxicity_score
        }

        with open(cache_file_path, 'w') as cache_file:
            json.dump(cache, cache_file)
            
    def _parse_channel_from_url(self, url):

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc in ['t.me', 'telegram.me']:
                channel_path = parsed_url.path.lstrip('/').lstrip('+')
                return channel_path
            else:
                return None
        except Exception as e:
            print('Had the error:', e)
            return None

    def obtain_channel_name(self, message):

        channel_name = message.strip()

        if channel_name.startswith('http'):
            parsed_name = self._parse_channel_from_url(channel_name)
            if parsed_name:
                return parsed_name
            else:
                self._ask_for_new_channel_name(message)
                return None

        elif channel_name.startswith('@'):
            return channel_name

        else:
            self._ask_for_new_channel_name(message)
            return None

    def _ask_for_new_channel_name(self, message):

        self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
        self.user_management.set_user_state(message.chat.id, 'awaiting_channel_name')
        self.bot.register_next_step_handler(message, self.handle_next_channel_name)

    def channel_classifier(self, message):

        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        markup = self._get_base_markup('basic')
        markup_2 = self._get_base_markup('exception')

        try:
            channel_name = self.obtain_channel_name(message.text)
            self._analyze_channel_messages(message, channel_name, state, markup)

        except (IndexError, ValueError) as e:
            self.bot.reply_to(message, "Oops! That is not a valid channel name. Try again! 🫣", reply_markup=markup_2)
            print(e)
        except Exception as e:
            self.bot.reply_to(message, "Oops! Something went wrong 😞 Let's start over!", reply_markup=markup_2)
            print(e)
    
    def _analyze_channel_messages(self, message, channel_name, state, markup):
        
        self.bot.reply_to(message, f"Got it! I will analyze {channel_name}... Please wait a moment 🙏")
        messages, response_time = self._fetch_and_process_messages(channel_name, state)
        state.update_channel_analysis(channel_name, messages)
        print(messages)
        self._reply_based_on_response_time(message, response_time)

        if messages: 
            self._classify_messages(messages, state, message, channel_name, markup)
        else:
            self.bot.reply_to(message, "No messages found in the specified channel! Why don't we start again?", reply_markup=markup)

    def _fetch_and_process_messages(self, channel_name, state):

        start = time.time()
        messages = self.formatter.process_messages(self.loop.run_until_complete(self.formatter.fetch(channel_name)))

        end = time.time()
        state.last_chunk_of_messages = messages
        return messages, end - start

    def _reply_based_on_response_time(self, message, response_time):

        if response_time > 10:
            self.bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me a second to check for toxicity in the channel 🕣")
        else:
            self.bot.reply_to(message, "I've got the messages! Now give me a few seconds to check for toxicity in the channel... 🕵️‍♂️")

    def _classify_messages(self, messages, state, message, channel_name, markup):
        filtered_messages = self.hatebert.get_most_toxic_messages(messages)
        average_toxicity_score = self.hatebert.predict_average_toxicity_score(filtered_messages)
        state.last_analyzed_toxicity = average_toxicity_score
        self._send_toxicity_response(message, channel_name, average_toxicity_score, markup)
        self.write_cache(message.chat.id, channel_name, average_toxicity_score)
        


    def _send_toxicity_response(self, message, channel_name, toxicity_score, markup):
        print('The toxicity score for', channel_name,' is:', toxicity_score)
        if toxicity_score < 1.5:
            self._send_response_message(message, '🟢', channel_name, "isn't toxic at all!\n\nDo you want to learn more about our analysis? Click on the buttons below!", markup)
        elif 1 <= toxicity_score < 2.5:
            self._send_response_message(message, '🟡', channel_name, "has quite toxic content!\n\nDo you want to learn more about our analysis? Click on the buttons below!", markup)
        elif 2.5 <= toxicity_score:
            self._send_response_message(message, '🔴', channel_name, "has really toxic content!\n\nDo you want to learn more about our analysis? Click on the buttons below!", markup)

    def _send_response_message(self, message, emoji, channel_name, assessment, markup):
        response_text = f"{emoji} Well, {channel_name} {assessment}"
        self.bot.send_message(message.chat.id, response_text, reply_markup=markup)

    def _get_base_markup(self, marktype):
        if marktype == 'basic':
            markup = types.InlineKeyboardMarkup(row_width=2)
            explain_toxicity = types.InlineKeyboardButton('Toxicity dimensions 📊', callback_data='learn_more')
            explain = types.InlineKeyboardButton('Explain me 👀', callback_data='explainer')
            restart = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
            markup.add(explain_toxicity, explain, restart)
        
        elif marktype == 'exception':
            markup = types.InlineKeyboardMarkup(row_width=2)
            analyze = types.InlineKeyboardButton('Analyze again 🔄', callback_data='analyze')
            restart = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
            markup.add(analyze, restart)
        return markup
    

