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
classifier_instance = perspective_classificator(GOOGLE_CLOUD_API_KEY ,attributes=["TOXICITY"], verbosity=True)
classifier_instance2 = hate_bert_classificator("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert", verbosity = True)

#Check the token
print("Token:", BOT_TOKEN)

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize event loop
loop = asyncio.new_event_loop()

# Initialize the Telethon client with MemorySession
client = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)

#Function for checking if a string is a number, for the gpt-3 toxicity
async def is_number(s) -> bool:
    try:
        float(s) 
        return True
    except ValueError:
        return False

#Use the telegram API for fetching the last 50 messages of a channel (this will need verification)
#To-Do: manage verification with telegram user.
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

#function for analizying with gpt

def gpt_classificator(data) -> float:
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are moderator of toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and agressiveness. Reply only with a number from 0 to 4, no additional characters. Use the following scale:

                    0- **Non-Toxic: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    1- **Slightly Toxic**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    2- **Moderately Toxic**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    3- **Highly Toxic**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    4- **Extremely Toxic**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.


                ## EXAMPLES
                user: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.\n\nAnna Sevidova was injured when her home was shelled in Ukraine. Now she\u2019s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.\n\nAs Ukrainian refugees flood into Moldova, one of Europe\u2019s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova\u2019s population.\n\nThe New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more\n\n@nytimes
                assistant: 0

                user: Classify the following message: With #LloydAustin out of commission, Joe Biden for several days had no idea who his Secretary of Defense was. But then Americans right now don’t know who their President is. The country is being run by an unelected junta operating through a complicit semi-retarded figurehead 
                assistant: 3
             
                user: Classify the following message: These people are so much stoopiderer than I ever thought possible.!
                assistant: 3


                user: Classify the following message: They're not happy in Gaza .. \nThey're not happy in Egypt .. \nThey're not happy in  Libya .. \nThey're not happy in Morocco .. \nThey're not happy in Iran .. \nThey're not happy in Iraq .. \nThey're not happy in Yemen ... \nThey're not happy in Afghanistan ... \nThey're not happy in Pakistan .. \nThey're not happy in Syria .. \nThey're not happy in Lebanon ...\n SO.. WHERE ARE THEY HAPPY?\n They're happy in Australia .. \nThey're happy in Canada .. \nThey're happy in England .. \nThey're happy in Ireland...\nThey're happy in France .. \nThey're happy in Italy .. \nThey're happy in Germany .. \nThey're happy in Sweden .. \nThey're happy in the USA .. \nThey're happy in Norway .. \nThey're happy in Holland .. \nThey're happy in Denmark ..\n Basically, they're happy in every country that is not Muslim and unhappy in every country that is!\n AND WHO DO THEY BLAME?\n Not Islam. \nNot their leadership. \nNot themselves\n THEY BLAME THE COUNTRIES THEY ARE HAPPY IN !\n AND THEN- They want to change those countries to be like, THE COUNTRY THEY CAME FROM WHERE THEY WERE UNHAPPY!\n Excuse me, but I can't help wondering... \nHow frigging dumb can you get? \nEveryone seems to be wondering why Muslim Terrorists are so quick to commit suicide.\n Lets have a look at the evidence: \n- No Christmas \n- No television \n- No nude women \n- No football \n- No pork chops \n- No hot dogs \n- No burgers \n- No beer \n- No bacon \n- Rags for clothes \n- Towels for hats \n- Constant wailing from some bloke in a tower \n- More than one wife \n- More than one mother-in-law \n- You can't shave \n- Your wife can't shave \n- You can't wash off the smell of donkeys \n- You cook over burning camel shit \n- Your wife is picked by someone else for you \n- and your wife smells worse than your donkey \n- Then they tell them that \"when they die, it all gets better\"???\n Well No Shit Sherlock!.... \nIt's not like it could get much worse!
                assistant: 4


                    """),

                    ("user", "Classify the following message: {message}")
                ])
    chain = prompt_template | llm | output_parser
    # Batch input for classification
    batch_input = [{"message": msg["message"]} for msg in data]
    output = chain.batch(batch_input)
    # Filter out non-numeric items and convert to float
    filtered_output = [float(item) for item in output if is_number(item)]
    # Convert to a NumPy array
    filtered_output_array = np.array(filtered_output)
    # Calculate the average
    average = np.mean(filtered_output_array)
    return average

def summarizor_gpt(data) -> str:
    prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """You are moderator of toxic content in chat groups. Your task is to summarize the content of the following messages in a paragraph, no more than 50 words. The summary has to filter the toxic content and be as objective as possible. The classification of toxicity is the following:
                    0- **Non-Toxic: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    1- **Slightly Toxic**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    2- **Moderately Toxic**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    3- **Highly Toxic**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    4- **Extremely Toxic**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                  So, if the message is classified as 1, it is not toxic and you can include it in the summary. If it is classified as 2, you can include it but you have to be careful and start filtering information. If it is classified as 3, you should filter it and tell the information in an objective way. If it is classified as 4 or 5, you should do the same as 3, but also include a warning for the user and a recommendation regarding how to read that kind of information.
                    Remember to NOT OUTPUT a paragraph with more than 50 words. If you do so, the system will not be able to process your answer.
                     """),

                    ("user", " Reply with the summary of the following messages")
                ]+ data)
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
                        toxicity_result = classifier_instance.predictToxicity(msg['message'])
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
                    toxicity_result = classifier_instance2.predictToxicity(msg['message'])
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
                average = gpt_classificator(data)
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
                average_gpt = gpt_classificator(data)
                print(average_gpt)
                message_perspective = 0
                message_bert = 0
                for msg in data:
                    toxicity_result = classifier_instance.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_perspective += numeric_toxicity
                average_perspective = message_perspective / len(data)
                print(average_perspective)
                for msg in data:
                    toxicity_result = classifier_instance2.predictToxicity(msg['message'])
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
                bot.reply_to(message, "Channel analysis in progress...")
                data = processed_messages[:20]
                if data:
                    print(data)
                output = summarizor_gpt(data)
                max_length = 4096
                if len(output) > max_length:
                # Si la respuesta excede el máximo, divídela en partes
                    parts = [output[i:i + max_length] for i in range(0, len(output), max_length)]
                    for part in parts:
                        bot.reply_to(message, f'{part}')
            elif len(output) <= max_length:
                # Si no excede el máximo, envía la respuesta como está
                bot.reply_to(message, f'{output}')
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


