'''
-> Avisar q se toman los ultimos 50 mensajes ?  ✅
-> Ir para atras y elegir random ? 
       - Ultimas horas o toda la historia del canal ?
- limite cuando se deje de sentir q es en tiempo real.
- cuando este cerrado chequear tiempos de respuesta
-> WExplain me the toxicity level -> cambiar wording
-> show me an example -> mediana del canal -> would you like to see a detoxify version -> show me another example -> desues de 3 nada más. 
->  show more -> detoxify my messages. 
'''
import sys
import time
sys.path.append('..')
import os
from dotenv import load_dotenv
import telebot
from telebot import types
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import os
from model_evaluation_scripts.classifiers_classes_api.perspective_classifier import perspective_classifier
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.gpt_classifier import gpt_classifier
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import numpy as np

# Load environment variables
print(os.getcwd())
load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
API_ID_TELEGRAM = os.environ.get('API_ID')
API_HASH_TELEGRAM = os.environ.get('API_HASH')
#OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_CLOUD_API_KEY = os.environ['GOOGLE_CLOUD_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0)
output_parser = StrOutputParser()

#Initialize classifiers
# perspective = perspective_classifier(GOOGLE_CLOUD_API_KEY ,attributes=["TOXICITY"], verbosity=True)
bert = hate_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert", verbosity = True)
# gpt = gpt_classifier("gpt-3.5-turbo", OPENAI_API_KEY, verbosity = True)
#Check the token
print("Token:", BOT_TOKEN)

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize event loop
loop = asyncio.new_event_loop()

# Initialize the Telethon client with MemorySession
client = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)

#Some global variables
last_analyzed_toxicity = {}

#Use the telegram API for fetching the last 50 messages of a channel (this will need verification)
async def fetch_last_50_messages(channel_name) -> list:
    '''
    Requiere: channel_name
    Retorna: lista de mensajes
    '''
    await client.start()
    channel_entity = await client.get_entity(channel_name)
    posts = await client(GetHistoryRequest(
        peer=channel_entity,
        limit=50,
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))
    await client.disconnect()
    return posts.messages

def transform_data_to_expected_format(data):
    transformed_data = [("user", item["message"]) for item in data]
    return transformed_data

#process_messages will return a list of messages with the format {'message': 'message', 'timestamp': 'timestamp'}
def process_messages(messages) -> list:
    processed_messages = []
    for msg in messages:
        if msg.message:
            processed_message = {
                'message': msg.message,
                'timestamp': msg.date.strftime('%Y-%m-%d %H:%M:%S')
            }
            processed_messages.append(processed_message)
    return processed_messages
def summarizor_gpt(data, channelname) -> str:
    transformed_data = transform_data_to_expected_format(data)
    print(transformed_data)
    if channelname:
            start = time.time()
            messages = loop.run_until_complete(fetch_last_50_messages(channelname))
            end = time.time()
            print(f"Time to fetch messages: {end - start}")
            bot.reply_to(channelname, "I've got the messages! Now give me one second to summarize them... 🕣")
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                data = processed_messages[:20]
                message_gpt = 0
                message_perspective = 0
                message_bert = 0
                for msg in data:
                    toxicity_result = gpt.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_gpt += numeric_toxicity
                average_gpt = message_gpt / len(data)
                print(average_gpt)
                for msg in data:
                    toxicity_result = perspective.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_perspective += numeric_toxicity
                average_perspective = message_perspective / len(data)
                print(average_perspective)
                for msg in data:
                    toxicity_result = bert.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_bert += numeric_toxicity
                average_bert = message_bert / len(data)
                print(average_bert)
                toxicity = (average_gpt + average_perspective + average_bert) / 3
    prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    Your role as a moderator involves two key tasks:

                    Summarize the content of the following messages in a concise paragraph, limiting your summary to no more than 50 words. The summary should remain objective and exclude toxic content.
                    Evaluate the channel's toxicity level based on the content you've summarized, providing a brief explanation for your assessment according to the toxicity scale provided.
                    Toxicity Scale:

                    0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting **kindness** and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.
                    1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.
                    2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, **it seeks to hurt, ridicule, or belittle others**, showing a rejection towards the diversity of opinions and people.
                    3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, **exclude**, or **incite** hatred, showing an explicit intention to cause harm.
                    4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    
                    Instructions:

                    For Non-toxic content, include it directly in the summary.
                    For Slightly to Moderately Toxic content, incorporate it carefully, filtering out any bias or offensive elements.
                    For Highly to Extremely Toxic content, exclude the offensive details and focus on objective information. Additionally, provide a warning about the content's potential impact.
                    After summarizing, conclude with your evaluation of the channel's overall toxicity. Consider the prevalence of toxic messages and their severity to explain whether the channel is generally toxic or not and why.

                    Remember: Your complete response, including the summary and toxicity evaluation, should be concise and informative, helping readers understand the channel's nature without exceeding the word limit for the summary.
                     
                    ### EXAMPLES
                     Understood, here are conversation examples for each level of toxicity:

                        1. **Non-toxic:**
                        - Conversation:
                            ```
                            User1: I think renewable energy is a great solution for combating climate change.
                            User2: I do not agree with you, but you may have a point! It may be important to transition to sustainable energy sources.

                            Output:
                            It seems like the channel's toxicity level is 🟢 Non-toxic. The conversation encourages respectful dialogue and values diverse viewpoints, contributing to a constructive exchange of ideas.      The conversation focuses on constructive discussion about renewable energy, promoting a positive and inclusive environment.

                            ```

                        2. **Slightly Toxic:**
                        - Conversation:
                            ```
                            User1: I prefer Android phones over iPhones because they offer more customization options.
                            User2: Well, that's your opinion, but iPhones have better security features.

                            Output:
                            It seems like the channel's toxicity level is 🟡 Slightly Toxic. While the conversation maintains a mostly respectful tone, there's a subtle lack of appreciation for the opposing viewpoint, potentially hindering constructive discourse.
                        While the conversation addresses differing viewpoints, User2's response comes off as slightly dismissive, implying superiority in their preference.

                                            ```

                        3. **Moderately Toxic:**
                        - Conversation:
                            ```
                            User1: I'm considering becoming a vegetarian for ethical reasons.
                            User2: Don't be ridiculous. Humans are meant to eat meat. You'll just end up malnourished.
                            Output:
                            It seems like the channel's toxicity level is 🟡 Moderately Toxic. User2's response shows a rejection towards User1's perspective and lacks appreciation for their decision, potentially discouraging open dialogue.
                            User2's response dismisses User1's decision and exhibits a disrespectful attitude towards their dietary choice.

                                            ```

                        4. **Highly Toxic:**
                        - Conversation:
                            ```
                            User1: I support equal rights for all genders.
                            User2: You're just a feminazi pushing your agenda. Men are the real victims here.
                            Output:
                            It seems like the channel's toxicity level is 🔴 Highly Toxic. User2's response demonstrates a clear rejection of User1's viewpoint, using offensive language and inciting hatred towards a specific group.
                                User2's response contains derogatory language and dismisses User1's stance on gender equality, exhibiting clear contempt and hostility.

                                            ```

                        5. **Extremely Toxic:**
                        - Conversation:
                            ```
                            User1: I believe in freedom of speech and expression.
                            User2: Shut your mouth, you fascist pig. Your words have consequences, and I hope you suffer for them.
                            Output:
                            It seems like the channel's toxicity level is 🔴 Extremely Toxic. User2's response promotes violence and poses a severe danger to User1's well-being, crossing ethical and moral boundaries.
                                User2's response contains explicit aggression and threats towards User1, violating ethical principles and endangering their safety.

                      ```
                     """),

                    ("user", " Reply with the summary of the following messages (which is the first 50 message I attach here) and the toxicity level of the channel, which is the second number I attach \n"), 
                ]+ transformed_data + [toxicity])

    chain = prompt_template | llm | output_parser
    # Batch input for classification
    output = chain.batch([{}])
    return output
def helper(message):
    markup = types.InlineKeyboardMarkup(row_width=1)
    summarize = types.InlineKeyboardButton('Summarize a group chat 📝', callback_data='summarize')
    explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
    analyze = types.InlineKeyboardButton('Analyze a group chat 🔍', callback_data='analyze')
    go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
    help_text = (
        "Here's how you can use Detoxigram:\n\n"
        "1. **Analyze a Group Chat:** Tap the 'Analyze a group chat 🔍' button and provide the @ChannelName to analyze the toxicity of the channel's messages.\n\n"
        "2. **Summarize a Group Chat:** Tap the 'Summarize a group chat 📝' button and provide the @ChannelName to get a summary of the channel's messages and an evaluation of its toxicity level.\n\n"
        "3. **Understanding Toxicity Levels:** Tap the 'Explain me toxicity levels 👀' button to learn more about the different levels of toxicity.\n\n"
        "If you have any questions or need further assistance, feel free to ask!"
        "You can reach us out at malbaposse@mail.utdt.edu\n\n"
    )
    markup.add(analyze, summarize, explainer, go_back)
    bot.reply_to(message, help_text)

def analyze_channel_bert(message): 
    try:
        channel_name = message.text
        if channel_name:
            bot.reply_to(message, f"Got it! We will analyze {channel_name}... Please wait a moment 🙏")
            start = time.time()
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            end = time.time()

            if processed_messages:
                if end - start > 10:
                    bot.reply_to(message, "It took a while to get the messages, but I've got them! Now give me one second to predict the toxicity of the channel... 🕣")
                else:
                    bot.reply_to(message, "I've got the messages! Now give me some seconds to predict the toxicity of the channel... 🕣")
            markup = types.InlineKeyboardMarkup(row_width=1)
            summarize = types.InlineKeyboardButton('Summarize a group chat 📝', callback_data='summarize')
            explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
            go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
            new_analyze = types.InlineKeyboardButton('Analyze another group chat 🔍', callback_data='analyze')
            if len(processed_messages) > 0:
                data = processed_messages[:50]
                total_toxicity_score=0
                for msg in data:
                    toxicity_result = bert.predictToxicity(msg['message'])
                    toxicity_score, numeric_toxicity = toxicity_result
                    total_toxicity_score += numeric_toxicity
                average_toxicity_score = total_toxicity_score / len(messages)
                last_analyzed_toxicity[message.chat.id] = average_toxicity_score
                if 0 <= average_toxicity_score < 0.99:
                        markup.add(explainer, summarize, new_analyze, go_back)
                        bot.send_message(message.chat.id, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all!''', reply_markup=markup)
                elif 0.99 <= average_toxicity_score < 2.1:
                        markup.add(explainer, summarize, new_analyze, go_back)
                        bot.send_message(message.chat.id, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content''', reply_markup=markup)
                elif 2.1 <= average_toxicity_score < 4.1:
                        markup.add(explainer, summarize, new_analyze, go_back)
                        bot.send_message(message.chat.id, f'''🔴 Be careful... {channel_name} seems to have a highly toxic content''', reply_markup=markup)
            else:
                bot.reply_to(message, "No messages found in the specified channel! Why don't we try again with another channel? Type /start and we will start all over again :)", reply_markup=markup)
        else:
            bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")
    except IndexError:
        bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
    except Exception as e:
        bot.reply_to(message, f"Oh no! An error occurred: {str(e)}")

def summarize(message):
    try:    
        channel_name = message.text
        if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                bot.reply_to(message, "Give me one sec... 🤔")
                data = processed_messages[:20]
                output = summarizor_gpt(data, channel_name)
                print(output)
                bot.reply_to(message, f'{output[0]}')
            else:
                bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            bot.reply_to(message, "Please provide a channel name!")

    except:
        bot.reply_to(message, "An error occurred! Please try again!")
#start is the command used for saying hi to the user and giving him/her instructions
@bot.message_handler(commands=['start'])
def send_welcome(message):
    username = message.from_user.first_name
    markup = types.InlineKeyboardMarkup(row_width=1)
    analyze = types.InlineKeyboardButton('Analyze a group chat 🔍', callback_data='analyze')
    summarize = types.InlineKeyboardButton('Summarize a group chat 📝', callback_data='summarize')
    help = types.InlineKeyboardButton('Help, please! 🛟', callback_data='help' )
    markup.add(analyze, summarize, help)
    bot.send_message(message.chat.id, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I\'m here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n''', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def answer(callback):
    markup = types.InlineKeyboardMarkup(row_width=1)
    summarize = types.InlineKeyboardButton('Summarize a group chat 📝', callback_data='summarize')
    explainer = types.InlineKeyboardButton('Explain me the toxicity levels 👀', callback_data='explain')
    analyze = types.InlineKeyboardButton('Analyze a group chat 🔍', callback_data='analyze')
    go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
    new_analyze = types.InlineKeyboardButton('Analyze another group chat 🔍', callback_data='analyze')
            
    if callback.message:
        if callback.data == 'analyze':
            bot.send_message(callback.message.chat.id, "Great! Just for you to know, when we evaluate the toxicity, we'll only consider the last 50 messages of the channel ⚠️ Now, please provide the @ChannelName you would like to analyze 🤓")
            bot.register_next_step_handler(callback.message, analyze_channel_bert)
        elif callback.data == 'summarize':
            bot.send_message(callback.message.chat.id, "Let's do this! Please provide the @ChannelName you would like to summarize 🤓")
            bot.register_next_step_handler(callback.message, summarize)
        elif callback.data == 'help':
            helper(callback.message)
        elif callback.data == 'end':
            bot.send_message(callback.message.chat.id, "Goodbye! 👋 If you need anything else, just type /start")
        elif callback.data == 'restart':

            markup.add(analyze, summarize, explainer)
            bot.send_message(callback.message.chat.id, "Ok! Let's see, what would you like to do now? 🤔")
        elif callback.data == 'explain':
            toxicity_level = last_analyzed_toxicity.get(callback.message.chat.id, None)
            if toxicity_level is not None:
                if 0 <= toxicity_level < 0.99:
                    explanation = "🟢 Non-toxic: Messages that foster a positive, respectful, and inclusive atmosphere, promoting kindness and mutual understanding. They value diverse opinions and contribute to constructive dialogue without offensive content. ✨\n"
                elif 0.99 <= toxicity_level < 1.5:
                    explanation = "🟡 Slightly Toxic: Messages that, while mostly respectful, may include criticism or disagreements in a passive-aggressive manner. They lack full appreciation for differing viewpoints but don't directly attack individuals or groups. 🤔\n"
                elif 1.5 <= toxicity_level < 2.5:
                    explanation = "🟡 Moderately Toxic: Messages with an aggressive or disrespectful tone, often containing sarcasm, irony, or derogatory language towards specific groups. They seek to hurt, ridicule, or belittle others, showing a rejection of diversity. 😠\n\n"
                elif 2.5 <= toxicity_level < 3.2:
                    explanation = "🔴 Highly Toxic: Messages demonstrating clear rejection and contempt for individuals or groups, using insults, degrading language, or references based on gender, ethnicity, etc. They aim to intimidate, exclude, or incite hatred, with explicit intent to cause harm. 😡\n"
                elif 3.2 <= toxicity_level < 4:
                    explanation = "🔴 Extremely Toxic: Messages explicitly aggressive and disrespectful, containing threats or calls to violent action. They promote hostility, incite hatred, and suggest harmful consequences, endangering the safety and well-being of others. ⚠️ \n 😡\n"

                message_text = explanation + "\nUnderstanding these levels helps maintain a healthy and respectful online environment. Let's strive for constructive dialogue and mutual respect. 🌐💬"
                markup.add(new_analyze, summarize, go_back)
                bot.send_message(callback.message.chat.id, message_text)
            else:
                bot.send_message(callback.message.chat.id, "No toxicity level has been analyzed yet for this chat! ")
bot.infinity_polling()