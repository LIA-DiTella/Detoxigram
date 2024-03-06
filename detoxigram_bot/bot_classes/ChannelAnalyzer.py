import time
import telebot
from telebot import types

class ChannelAnalyzer:
    def __init__(self, bot, loop, formatter, bert):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.bert = bert

    def analyze_channel_bert(self, message):
        try:
            channel_name = message.text
            last_channel_analyzed = channel_name
            if channel_name:
                self.bot.reply_to(message, f"Got it! We will analyze {channel_name}... Please wait a moment 🙏")
                start = time.time()
                messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
                processed_messages = self.formatter.process_messages(messages)
                end = time.time()

                if processed_messages:
                    if end - start > 10:
                        self.bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me one second to predict the toxicity of the channel... 🕣")
                    else:
                        self.bot.reply_to(message, "I've got the messages! Now give me some seconds to predict the toxicity of the channel... 🕣")
                markup = types.InlineKeyboardMarkup(row_width=1)
                summarize = types.InlineKeyboardButton('Summarize a Channel 📝', callback_data='summarize')
                explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
                go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
                new_analyze = types.InlineKeyboardButton('Analyze another Channel 🔍', callback_data='analyze')
                if len(processed_messages) > 0:
                    data = processed_messages[:50]
                    total_toxicity_score = 0
                    for msg in data:
                        toxicity_result = self.bert.predictToxicity(msg['message'])
                        _, numeric_toxicity = toxicity_result
                        total_toxicity_score += numeric_toxicity
                    average_toxicity_score = total_toxicity_score / len(messages)
                    if 0 <= average_toxicity_score < 0.99:
                            markup.add(explainer, summarize, new_analyze, go_back)
                            self.bot.send_message(message.chat.id, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all!''', reply_markup=markup)
                    elif 0.99 <= average_toxicity_score < 2.1:
                            markup.add(explainer, summarize, new_analyze, go_back)
                            self.bot.send_message(message.chat.id, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content''', reply_markup=markup)
                    elif 2.1 <= average_toxicity_score < 4.1:
                            markup.add(explainer, summarize, new_analyze, go_back)
                            self.bot.send_message(message.chat.id, f'''🔴 Be careful... {channel_name} seems to have a highly toxic content''', reply_markup=markup)
                else:
                    self.bot.reply_to(message, "No messages found in the specified channel! Why don't we try again with another channel? Type /start and we will start all over again :)", reply_markup=markup)
            else:
                self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")
        except IndexError:
            self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
        except Exception as e:
            self.bot.reply_to(message, f"Oh no! An error occurred: {str(e)}")
