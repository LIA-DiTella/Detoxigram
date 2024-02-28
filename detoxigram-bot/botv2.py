'''
-> Avisar q se toman los ultimos 50 mensajes ?
-> Ir para atras y elegir random ? 
       - Ultimas horas o toda la historia del canal ?
- limite cuando se deje de sentir q es en tiempo real.
- cuando este cerrado chequear tiempos de respuesta
'''
import sys
sys.path.append('..')
import os
from dotenv import load_dotenv
import telebot
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
perspective = perspective_classifier(GOOGLE_CLOUD_API_KEY ,attributes=["TOXICITY"], verbosity=True)
bert = hate_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert", verbosity = True)
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

def summarizor_gpt(data, channelname) -> str:
    transformed_data = transform_data_to_expected_format(data)
    print(transformed_data)
    channel_name = channelname.text.split(' ')[1]
    if channel_name:
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
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

#start is the command used for saying hi to the user and giving him/her instructions
@bot.message_handler(commands=['start'])
def send_welcome(message):
    username = message.from_user.first_name
    bot.reply_to(message, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I\'m here to help you to identify toxicity in your telegram channels, so you can make an informed choice in the content you consume and share 🤖\n 
You can use the following commands: \n
                 - /analyze_p, *_b, *_gpt or *_crowd @ChannelName: to analyze the toxicity of a channel using Perspective API\n
                 - /summarize @ChannelName: to summarize the content of a channel and understand if it is toxic and why\n
                 - /help: to get more information about the commands available 📚\n
Watch out! This bot is still in development, so if you find any bugs or have any suggestions, please let us know! 🐛🔍
''')
               
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
                    if 0 <= average_toxicity_score < 0.99:
                        bot.reply_to(message, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all! \n In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                    elif 0.99 <= average_toxicity_score < 2.1:
                        bot.reply_to(message, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content. \n In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                    elif 2.1 <= average_toxicity_score < 4.1:
                        bot.reply_to(message, f'''🔴 You should be careful... {channel_name} seems to have a highly toxic content. \n In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
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
                if 0 <= average_toxicity_score < 0.99:
                        bot.reply_to(message, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all! \nIn order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 0.99 <= average_toxicity_score < 2.1:
                        bot.reply_to(message, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content. \nIn order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 2.1 <= average_toxicity_score < 4.1:
                        bot.reply_to(message, f'''🔴 You should be careful... {channel_name} seems to have a highly toxic content. \nIn order to summarize it's content, you can use the command /summarize @channelname 🤓''')

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
                data = processed_messages[:20]
                total_toxicity_score=0
                for msg in data:
                    toxicity_result = gpt.predictToxicity(msg['message'])
                    toxicity_score, numeric_toxicity = toxicity_result
                    total_toxicity_score += numeric_toxicity
                average_toxicity_score = total_toxicity_score / len(messages)
                if 0 <= average_toxicity_score < 0.99:
                        bot.reply_to(message, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all! \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 0.99 <= average_toxicity_score < 2.1:
                        bot.reply_to(message, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content. \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 2.1 <= average_toxicity_score < 4.1:
                        bot.reply_to(message, f'''🔴 You should be careful... {channel_name} seems to have a highly toxic content. \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
            else:
                bot.reply_to(message, "No messages found in the specified channel.")
        else:
            bot.reply_to(message, "Ups! That is not a valid channel name. Try again!🫣")
    except IndexError:
        bot.reply_to(message, "Ups! That is not a valid channel name. Try again! 🫣")
    except Exception as e:
        bot.reply_to(message, f"Oh no! An error occurred: {str(e)}")
    

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
                average_crowd = (average_gpt + average_perspective + average_bert) / 3
                if 0 <= average_crowd < 0.99:
                        bot.reply_to(message, f'''🟢 Well, {channel_name} doesn't seem to be toxic at all! \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 0.99 <= average_crowd < 2.1:
                        bot.reply_to(message, f'''🟡 Watch out! {channel_name} seems to have a moderately toxic content. \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')
                elif 2.1 <= average_crowd < 4.1:
                        bot.reply_to(message, f'''🔴 You should be careful... {channel_name} seems to have a highly toxic content. \n
In order to summarize it's content, you can use the command /summarize @channelname 🤓''')

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
                output = summarizor_gpt(data, message)
                print(output)
                bot.reply_to(message, f'{output[0]}')
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


