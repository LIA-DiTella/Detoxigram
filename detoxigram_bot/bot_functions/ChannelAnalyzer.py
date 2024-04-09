import time
import telebot
from telebot import types
from urllib.parse import parse_qs, urlparse

class ChannelAnalyzer:
    def __init__(self, bot, loop, formatter, multibert, mistral, user_management):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.multibert = multibert
        self.mistral = mistral
        self.user_management = user_management


    def resolve_invite_link(self, invite_link):
        '''
        Pre: invite_link is a string
        Post: returns the username of the chat
        '''
        chat = self.bot.get_chat(invite_link)
        return chat.username

    def obtain_channel_name(self, channel_name):
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
            return channel_name
    
    def ChannelClassifier(self, message):
        '''
        Pre: message is a string with the name of a channel
        Post: returns the toxicity of the channel
        '''
        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        
        markup = types.InlineKeyboardMarkup(row_width=2)
        explainer = types.InlineKeyboardButton('Explain me 👀', callback_data='explainer')
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        learn_more = types.InlineKeyboardButton('Learn more 📚', callback_data='learn_more')
        average_toxicity_score = 0
        total_toxicity_score = 0
        try:
            channel_name = self.obtain_channel_name(message.text)
            state.last_channel_analyzed = channel_name
            if channel_name:
                self.bot.reply_to(message, f"Got it! I will analyze {channel_name}... Please wait a moment 🙏")
                start = time.time()
                state.last_chunk_of_messages = self.formatter.process_messages(self.loop.run_until_complete(self.formatter.fetch(channel_name)))
                end = time.time()
                
                if state.last_chunk_of_messages:
                    if end - start > 10:
                        self.bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me one second to understand the toxicity of the channel... 🕣")
                    else:
                        self.bot.reply_to(message, "I've got the messages! Now give me some seconds to understand the toxicity of the channel... 🕣")
                
                if len(state.last_chunk_of_messages) > 0:
                    filtered_messages = self.multibert.get_most_toxic_messages([item['message'] for item in state.last_chunk_of_messages])
                    state.last_chunk_of_messages = filtered_messages
                    
                    for msg in filtered_messages:
                        toxicity_result = self.mistral.predictToxicity(msg)
                        numeric_toxicity = 0

                        if isinstance(toxicity_result, tuple) and len(toxicity_result) > 1:

                            if isinstance(toxicity_result[1], int):
                                numeric_toxicity = toxicity_result[1]
                                total_toxicity_score += numeric_toxicity
                            else:
                                print("El segundo valor no es un entero:", toxicity_result[1])
                                pass
                        else:
                            print("Resultado no es una tupla o no tiene suficientes elementos:", toxicity_result)
                            pass                                        
                        
                        total_toxicity_score += numeric_toxicity
                        print(total_toxicity_score)
                    
                    average_toxicity_score = total_toxicity_score / len(filtered_messages)
                    state.last_analyzed_toxicity = average_toxicity_score
                    
                    if 0 <= average_toxicity_score < 0.80:
                            markup.add(learn_more, explainer, go_back)
                            self.bot.send_message(message.chat.id, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all!''', reply_markup=markup)
                    elif 0.80 <= average_toxicity_score < 2.1:
                            markup.add(learn_more, explainer, go_back)
                            self.bot.send_message(message.chat.id, f'''🟡 Watch out! {channel_name} seems to have quite toxic content...''', reply_markup=markup)
                    elif 2.1 <= average_toxicity_score < 4.1:
                            markup.add(learn_more, explainer, go_back)
                            self.bot.send_message(message.chat.id, f'''🔴 Be careful... {channel_name} seems to have really toxic content!''', reply_markup=markup)
                
                else:
                    markup.add(go_back)
                    self.bot.reply_to(message, "No messages found in the specified channel! Why don't we start again?", reply_markup=markup)
            
            else:
                self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")  
        
        except IndexError:
            self.bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
        
        except Exception as e:
            print( f"Oh no! An error occurred: {str(e)}")