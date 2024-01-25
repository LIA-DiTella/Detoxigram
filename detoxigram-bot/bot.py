import os
from dotenv import load_dotenv
import telebot
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import numpy as np
import json 
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
print(os.getcwd())
load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
api_id = os.environ.get('API_ID')
api_hash = os.environ.get('API_HASH')
#OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

print("Token:", BOT_TOKEN)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0)
output_parser = StrOutputParser()
# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize the Telethon client with MemorySession
client = TelegramClient(sessions.MemorySession(), api_id, api_hash)

async def is_number(s):
    try:
        float(s)  # Attempt to convert to a float
        return True
    except ValueError:
        return False
    

async def fetch_last_50_messages(channel_name):
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

def process_messages(messages):
    processed_messages = []
    for msg in messages:
        if msg.message:  # Asegurarse de que el mensaje no esté vacío
            processed_message = {
                'message': msg.message,
                'timestamp': msg.date.strftime('%Y-%m-%d %H:%M:%S')
            }
            processed_messages.append(processed_message)
    return processed_messages

# Bienvenida
@bot.message_handler(commands=['start'])
def send_welcome(message):
    username = message.from_user.first_name
    bot.reply_to(message, f'''Hello {username} and welcome to Detoxigram! 👋 \n
I\'m here to help you to identify toxicity in you telegram channels, so you can make an informed choice in the information you consume and share 🤖\n 
To start, use the command /analyze plus the username of the channel you would like to access. For example: /analyze @ChannelName 👀 \n''')
               
#Pasaje de nombre del canal 
@bot.message_handler(commands=['analyze'])
def analyze_channel(message):
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                bot.reply_to(message, "Channel analysis in progress...")
                data = processed_messages[:20]
                print(data)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """You are moderator of polarizing and toxic content in chat groups. Your task is to Classify the following messages in a scale from 1 to 5 according to their level of toxicity and agressiveness. Reply only with a number from 1 to 5, no additional characters. Use the following scale:

                    1- **Non-Toxic and Non-Polarizing**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    2- **Slightly Toxic or Polarizing**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    3- **Moderately Toxic or Polarizing**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    4- **Highly Toxic or Polarizing**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    5- **Extremely Toxic and Polarizing**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.


                ## EXAMPLES
                user: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.\n\nAnna Sevidova was injured when her home was shelled in Ukraine. Now she\u2019s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.\n\nAs Ukrainian refugees flood into Moldova, one of Europe\u2019s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova\u2019s population.\n\nThe New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more\n\n@nytimes
                assistant: 1


                user: Classify the following message: These people are so much stoopiderer than I ever thought possible.!
                assistant: 3


                user: Classify the following message: They're not happy in Gaza .. \nThey're not happy in Egypt .. \nThey're not happy in  Libya .. \nThey're not happy in Morocco .. \nThey're not happy in Iran .. \nThey're not happy in Iraq .. \nThey're not happy in Yemen ... \nThey're not happy in Afghanistan ... \nThey're not happy in Pakistan .. \nThey're not happy in Syria .. \nThey're not happy in Lebanon ...\n SO.. WHERE ARE THEY HAPPY?\n They're happy in Australia .. \nThey're happy in Canada .. \nThey're happy in England .. \nThey're happy in Ireland...\nThey're happy in France .. \nThey're happy in Italy .. \nThey're happy in Germany .. \nThey're happy in Sweden .. \nThey're happy in the USA .. \nThey're happy in Norway .. \nThey're happy in Holland .. \nThey're happy in Denmark ..\n Basically, they're happy in every country that is not Muslim and unhappy in every country that is!\n AND WHO DO THEY BLAME?\n Not Islam. \nNot their leadership. \nNot themselves\n THEY BLAME THE COUNTRIES THEY ARE HAPPY IN !\n AND THEN- They want to change those countries to be like, THE COUNTRY THEY CAME FROM WHERE THEY WERE UNHAPPY!\n Excuse me, but I can't help wondering... \nHow frigging dumb can you get? \nEveryone seems to be wondering why Muslim Terrorists are so quick to commit suicide.\n Lets have a look at the evidence: \n- No Christmas \n- No television \n- No nude women \n- No football \n- No pork chops \n- No hot dogs \n- No burgers \n- No beer \n- No bacon \n- Rags for clothes \n- Towels for hats \n- Constant wailing from some bloke in a tower \n- More than one wife \n- More than one mother-in-law \n- You can't shave \n- Your wife can't shave \n- You can't wash off the smell of donkeys \n- You cook over burning camel shit \n- Your wife is picked by someone else for you \n- and your wife smells worse than your donkey \n- Then they tell them that \"when they die, it all gets better\"???\n Well No Shit Sherlock!.... \nIt's not like it could get much worse!
                assistant: 5


                    """),

                    ("user", "Classify the following message: {message}")
                ])
                chain = prompt_template | llm | output_parser
                # Batch input for classification
                batch_input = [{"message": msg["message"]} for msg in data]
                output = chain.batch(batch_input)

                # Process and display the output
                for msg, classification in zip(data, output):
                    print(f"Message: {msg['message']}\nClassified as: **{classification}**\n")

                print(output)

                # Filter out non-numeric items and convert to float
                filtered_output = [float(item) for item in output if is_number(item)]

                print(filtered_output)

                # Convert to a NumPy array
                filtered_output_array = np.array(filtered_output)

                # Calculate the average
                average = np.mean(filtered_output_array)

                bot.reply_to(message, f'Average toxicity of the channel: {average}')

            else:
                bot.reply_to(message, f'Failed! Try with another channel')
        
        else:
            bot.reply_to(message, "Please provide a channel name.")
    except IndexError:
        bot.reply_to(message, "Usage: /analyze @channelname")


bot.infinity_polling()
