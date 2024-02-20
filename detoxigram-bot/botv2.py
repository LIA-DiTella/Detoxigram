import sys
sys.path.append('..')
import os
from dotenv import load_dotenv
import telebot
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import os
from model_evaluation_scripts.classifiers_classes_api.perspective_classifier import perspective_classificator
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classificator
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
perspective = perspective_classificator(GOOGLE_CLOUD_API_KEY ,attributes=["TOXICITY"], verbosity=True)
bert = hate_bert_classificator("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert", verbosity = True)
gpt = gpt_classifier("gpt-3.5-turbo", OPENAI_API_KEY, verbosity = True)
#Check the token
print("Token:", BOT_TOKEN)

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize event loop
loop = asyncio.new_event_loop()

# Initialize the Telethon client with MemorySession
client = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)

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

def summarizor_gpt(data) -> str:
    transformed_data = transform_data_to_expected_format(data)
    prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    Your role as a moderator involves two key tasks:

                    Summarize the content of the following messages in a concise paragraph, limiting your summary to no more than 50 words. Your summary should remain objective and exclude toxic content.
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
                     """),

                    ("user", " Reply with the summary of the following messages")
                ]+ transformed_data)
    chain = prompt_template | llm | output_parser
    # Batch input for classification
    output = chain.batch([{}])
    return output

#start is the command used for saying hi to the user and giving him/her instructions
@bot.message_handler(commands=['start'])
def send_welcome(message):
    username = message.from_user.first_name
    bot.reply_to(message, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I\'m here to help you to identify toxicity in you telegram channels, so you can make an informed choice in the information you consume and share 🤖\n 
To start, use the command /analyze plus the username of the channel you would like to access. For example: /analyze @ChannelName 👀 \n''')
               
#analyze_p will analyze the toxicity of a channel using the perspective API
@bot.message_handler(commands=['analyze_p'])
def analyze_channel_perspective(message):
    try:
            channel_name = message.text.split(' ')[1]
            if channel_name:
                messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
                processed_messages = process_messages(messages)
                if len(processed_messages) > 0:
                    total_toxicity_score = 0
                    data = processed_messages[:20]
                    for msg in data:
                        toxicity_result = perspective.predictToxicity(msg['message'])
                        toxicity_score, numeric_toxicity = toxicity_result
                        total_toxicity_score += numeric_toxicity

                    average_toxicity_score = total_toxicity_score / len(messages)
                    bot.reply_to(message, f'''The toxicity score of this channel is {average_toxicity_score} 👀''')
                else:
                    bot.reply_to(message, "No messages found in the specified channel.")
            else:
                bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")
    except IndexError:
        bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
    except Exception as e:
        bot.reply_to(message, f"Oh no! An error occurred: {str(e)}")

#analyze w/bert
@bot.message_handler(commands=['analyze_b'])
def analyze_channel_bert(message): 
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                data = processed_messages[:20]
                total_toxicity_score=0
                for msg in data:
                    toxicity_result = bert.predictToxicity(msg['message'])
                    toxicity_score, numeric_toxicity = toxicity_result
                    total_toxicity_score += numeric_toxicity
                average_toxicity_score = total_toxicity_score / len(messages)
                bot.reply_to(message, f'''The toxicity score of this channel is {average_toxicity_score} 👀''')
            else:
                bot.reply_to(message, "No messages found in the specified channel.")
        else:
            bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")
    except IndexError:
        bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
    except Exception as e:
        bot.reply_to(message, f"Oh no! An error occurred: {str(e)}")
    

#analyze w/gpt
@bot.message_handler(commands=['analyze_gpt'])
def analyze_channel_gpt(message):
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                data = processed_messages[20:]
                average = gpt.predictToxicity(data) # Ver la nueva clase que hizo @Santi (hay que cambiar antes del merge)
                bot.reply_to(message, f'The average toxicity of the channel: {average}')
            else:
                bot.reply_to(message, f'Failed! Try with another channel')
        
        else:
            bot.reply_to(message, "Please provide a channel name.")
    except IndexError:
        bot.reply_to(message, "Usage: /analyze @channelname")


# To-Do: analyze w/knowledge of crowds
@bot.message_handler(commands=['analyze_crowd'])
def knowledge_of_crowds(message):
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                data = processed_messages[:20]
                average_gpt = gpt.predictToxicity(data)
                print(average_gpt)
                message_perspective = 0
                message_bert = 0
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
                average_crowd = (average_gpt + average_perspective + average_bert) / 3
                bot.reply_to(message, f'Average toxicity of the channel: {average_crowd}')
            else:
                bot.reply_to(message, 'No messages to process...')
        else:
            bot.reply_to(message, f'Failed! Try with another channel')
    except IndexError:
        bot.reply_to(message, "Ups, try again")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")
#summarize 
@bot.message_handler(commands=['summarize'])
def summarize(message):
    try:    
        channel_name = message.text.split(' ')[1]
        if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                bot.reply_to(message, "Let's see... 🤔")
                data = processed_messages[:20]
                output = summarizor_gpt(data)
                print(output)
                bot.reply_to(message, f'So, here you have the summary of {channel_name}: {output[0]}')
            else:
                bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            bot.reply_to(message, "Please provide a channel name!")
    except IndexError:
        bot.reply_to(message, "Usage: /summarize @channelname")


@bot.message_handler(commands=['help'])
def helper(message):
    bot.reply_to(message, "You are a curious one! 🤓 This is our list of commands:\n - /start: This command will say hi to you and start the bot!\n -/analyze: When you want to analyze the toxicity of a channel, you should write /analyze @\n -/learn: Did you get interested in how is this bot made? Write this command and obtain information about toxicity!")


bot.infinity_polling()


