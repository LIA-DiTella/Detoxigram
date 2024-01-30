'''
#NOTAS: 30/01/24
- ¿Cuántos mensajes le queremos pasar?
- Podemos tener una serie de ejemplos que sean average. Después, le podemos mostrar como Bert lo clasifica o como Perspective lo clasifica.
- ¿Al sesgarlo (GPT),cambia? ¿Cómo? 
- Q: ¿Comunicación entre modelos, los sesga?
- Hacerle explicar a GPT por qué tal score es de esa forma.

Tendríamos que seleccionar mensajes que tengan distintos tipos de toxicidad y encuestar entre muchas personas. 

To-Do: 
   [] Seleccionar mensajes, un listado de al menos 100
   [] Cargar a un formulario 
   [] Hacer la encuesta con la mayor cantidad de personas posibles

- Humanos / GPT / Perspective / Bert

Escala:
≥`
1- **Non-Toxic**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.    
2- **Slightly Toxic**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.    
3- **Moderately Toxic**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.    
4- **Highly Toxic**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.    
5- **Extremely Toxic**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

'''

import os
# from dotenv import load_dotenv
import telebot
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import os
from model_evaluation_scripts.classifiers_classes_api.perspective_classifier import perspective_classificator
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classificator
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import numpy as np


classifier_instance = perspective_classificator('AIzaSyCjj8W9VqY7eIfuwZ8cL7DCFXCloUruk68',attributes=["TOXICITY"], verbosity=True)
classifier_instance2 = hate_bert_classificator("toxigen_hatebert", verbosity = True)

# Load environment variables
# print(os.getcwd())
# load_dotenv()

BOT_TOKEN = '6668823611:AAFe1suVmPEVHBwsGTTsrGp13oOv3s9MQVw'
api_id = '27486167'
api_hash = 'b6f8fbecc6568cf05834f9419582e8ca'

#Check the token
print("Token:", BOT_TOKEN)

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize the Telethon client with MemorySession
client = TelegramClient(sessions.MemorySession(), api_id, api_hash)

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
        if msg.message:
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
               
#analyze w7perspective
@bot.message_handler(commands=['analyze w/perspective'])
def analyze_channel_perspective(message):
    try:
            channel_name = message.text.split(' ')[1]
            if channel_name:
                messages = fetch_last_50_messages(channel_name)
                if messages:
                    total_toxicity_score = 0
                    for msg in messages:
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
@bot.message_handler(commands=['analyze w/bert'])
def analyze_channel_bert(message): 
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            messages = fetch_last_50_messages(channel_name)
            if messages:
                total_toxicity_score=0
                for msg in messages:
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
    

#analyze w/langchain
@bot.message_handler(commands=['analyze w/gpt'])
def analyze_channel_gpt(message):
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


#analyze w/knowledge of crowds
@bot.message_handler(commands=['analyze w/knowledge of crowds'])
def knowledge_of_crowds(message):
    return 0

#summarize 
@bot.message_handler(commands=['summarize'])
def analyze_channel_gpt(message):
    try:
        channel_name = message.text.split(' ')[1]
        if channel_name:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            messages = loop.run_until_complete(fetch_last_50_messages(channel_name))
            processed_messages = process_messages(messages)
            if len(processed_messages) > 0:
                bot.reply_to(message, "Channel analysis in progress...")
                data = processed_messages[:50]
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """You are moderator of toxic content in chat groups. Your task is to summarize the content of the following messages in a paragraph, no more than 200 words. The summary has to filter the toxic content and be as objective as possible. The classification of toxicity is the following:

                    1- **Non-Toxic and Non-Polarizing**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    2- **Slightly Toxic or Polarizing**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    3- **Moderately Toxic or Polarizing**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    4- **Highly Toxic or Polarizing**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    5- **Extremely Toxic and Polarizing**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                    So, if the message is classified as 1, it is not toxic and you can include it in the summary. If it is classified as 2, you can include it but you have to be careful and start filtering information. If it is classified as 3, you should filter it and tell the information in an objective way. If it is classified as 4 or 5, you should do the same as 3, but also include a warning for the user and a recommendation regarding how to read that kind of information.
                    """),

                    ("user", " Reply with the summary of the following messages: {processed_messages}")
                ])
                chain = prompt_template | llm | output_parser
                # Batch input for classification
                batch_input = [{"message": msg["message"]} for msg in data]
                output = chain.batch(batch_input)
                bot.reply_to(message, f'{output}')

            else:
                bot.reply_to(message, f'Failed! Try with another channel')
        
        else:
            bot.reply_to(message, "Please provide a channel name.")
    except IndexError:
        bot.reply_to(message, "Usage: /summarize @channelname")



@bot.message_handler(commands=['help'])
def helper(message):
    bot.reply_to(message, "You are a curious one! 🤓 This is our list of commands:\n - /start: This command will say hi to you and start the bot!\n -/analyze: When you want to analyze the toxicity of a channel, you should write /analyze @\n -/learn: Did you get interested in how is this bot made? Write this command and obtain information about toxicity and political polarization!")

# @bot.message_handler(commands=['learn'])
# def learner(message):
#     bot.reply_to(message, '''Coming soon...
# ''')

bot.infinity_polling()


