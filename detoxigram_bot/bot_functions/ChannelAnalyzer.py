import time
import telebot
from telebot import types
from urllib.parse import parse_qs, urlparse

class ChannelAnalyzer:
    def __init__(self, bot, loop, formatter, bert, gpt, last_channel_analyzed, last_toxicity):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.bert = bert
        self.last_channel_analyzed = last_channel_analyzed
        self.last_toxicity = last_toxicity
        self.gpt = gpt

    def resolve_invite_link(self, invite_link):
        chat = self.bot.get_chat(invite_link)
        return chat.username

    def analyze_channel_bert(self, message):
        markup = types.InlineKeyboardMarkup(row_width=2)
        summarize = types.InlineKeyboardButton('Summarize 📝', callback_data='summarize')
        explainer = types.InlineKeyboardButton('Explain me 👀', callback_data='explain')
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        global last_channel_analyzed

        try:
            channel_name = message.text
            
            if channel_name.startswith('http'):
                url = urlparse(channel_name)
                if url.netloc == 't.me':
                    path = url.path.lstrip('/')
                    if path.startswith('+'):
                        try:
                            chat = self.bot.get_chat(channel_name)
                            channel_name = chat.username
                        except Exception as e:
                            print(f"Error getting chat: {e}")
                    else:
                        channel_name = path
            last_channel_analyzed = channel_name
            
            print(f"Channel name: {channel_name}")
            print(f"Last channel analyzed: {last_channel_analyzed}")

            if channel_name:
                self.bot.reply_to(message, f"Got it! We will analyze {channel_name}... Please wait a moment 🙏")
                start = time.time()
                messages = self.loop.run_until_complete(self.formatter.fetch(channel_name))
                processed_messages = self.formatter.process_messages(messages)
                end = time.time()
                if processed_messages:
                    if end - start > 10:
                        self.bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me one second to predict the toxicity of the channel... 🕣")
                    else:
                        self.bot.reply_to(message, "I've got the messages! Now give me some seconds to predict the toxicity of the channel... 🕣")
                if len(processed_messages) > 0:
                    data = processed_messages[:100]
                    total_toxicity_score = 0
                    for msg in data:
                        toxicity_result = self.bert.predictToxicity(msg['message'])
                        _, numeric_toxicity = toxicity_result
                        total_toxicity_score += numeric_toxicity
                    average_toxicity_score = total_toxicity_score / len(messages)
                    self.last_toxicity = average_toxicity_score
                    if 0 <= average_toxicity_score < 0.99:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all!''', reply_markup=markup)
                    elif 0.99 <= average_toxicity_score < 2.1:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content''', reply_markup=markup)
                    elif 2.1 <= average_toxicity_score < 4.1:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🔴 Be careful... {channel_name} seems to have a highly toxic content''', reply_markup=markup)
                else:
                    markup.add(go_back)
                    self.bot.reply_to(message, "No messages found in the specified channel! Why don't we start again?", reply_markup=markup)
            else:
                self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")  
        except IndexError:
            self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
        except Exception as e:
            print( f"Oh no! An error occurred: {str(e)}")

    def ChannelClassifier(self, message):
        markup = types.InlineKeyboardMarkup(row_width=2)
        summarize = types.InlineKeyboardButton('Summarize 📝', callback_data='summarize')
        explainer = types.InlineKeyboardButton('Explain me 👀', callback_data='explain')
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        global last_channel_analyzed

        try:
            channel_name = message.text
            
            if channel_name.startswith('http'):
                url = urlparse(channel_name)
                if url.netloc == 't.me':
                    path = url.path.lstrip('/')
                    if path.startswith('+'):
                        try:
                            chat = self.bot.get_chat(channel_name)
                            channel_name = chat.username
                        except Exception as e:
                            print(f"Error getting chat: {e}")
                    else:
                        channel_name = path
            last_channel_analyzed = channel_name
            
            print(f"Channel name: {channel_name}")
            print(f"Last channel analyzed: {last_channel_analyzed}")

            if channel_name:
                self.bot.reply_to(message, f"Got it! We will analyze {channel_name}... Please wait a moment 🙏")
                start = time.time()
                messages = self.loop.run_until_complete(self.formatter.fetch(channel_name))
                processed_messages = self.formatter.process_messages(messages)
                end = time.time()
                if processed_messages:
                    if end - start > 10:
                        self.bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me one second to predict the toxicity of the channel... 🕣")
                    else:
                        self.bot.reply_to(message, "I've got the messages! Now give me some seconds to predict the toxicity of the channel... 🕣")
                if len(processed_messages) > 0:
                    data = processed_messages[:100]
                    data = self.bert.filter_toxic_messages(data)
                    total_toxicity_score = 0
                    
                for msg in data:
                    message_text = msg['message']
                    if not isinstance(message_text, str):
                        print(f"Message is not a string: {message_text}")
                        continue  # Skip this message
                    toxicity_result = self.gpt.predictToxicity(message_text)
                    _, numeric_toxicity = toxicity_result
                    total_toxicity_score += numeric_toxicity
                    self.last_toxicity = total_toxicity_score
                    
                    if 0 <= average_toxicity_score < 0.99:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all!''', reply_markup=markup)
                    elif 0.99 <= average_toxicity_score < 2.1:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content''', reply_markup=markup)
                    elif 2.1 <= average_toxicity_score < 4.1:
                            markup.add(explainer, summarize, go_back)
                            self.bot.send_message(message.chat.id, f'''🔴 Be careful... {channel_name} seems to have a highly toxic content''', reply_markup=markup)
                else:
                    markup.add(go_back)
                    self.bot.reply_to(message, "No messages found in the specified channel! Why don't we start again?", reply_markup=markup)
            else:
                self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")  
        except IndexError:
            self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
        except Exception as e:
            print( f"Oh no! An error occurred: {str(e)}")
