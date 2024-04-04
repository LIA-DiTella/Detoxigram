import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from telebot import types


class Explainer:
    def __init__(self, bot, loop, formatter, mistral, bert, output_parser, last_channel_analyzed):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.mistral = mistral
        self.bert = bert
        self.output_parser = output_parser
        self.last_channel_analyzed = last_channel_analyzed
        self.channel_name = None
        self.llm = mistral.chat

    def explain(self, message, channel_name, filtered_messages, toxicity):
        markup = types.InlineKeyboardMarkup(row_width=1)
        go_back = types.InlineKeyboardButton('Restart! 🔄', callback_data='restart')
        new_analyze = types.InlineKeyboardButton('New analyze 🔍', callback_data='analyze')

        if 0 <= toxicity < 1:
            toxicity = "Non-toxic"
        elif toxicity >= 1 and toxicity < 2:
            toxicity = "Slightly toxic"
        elif 2 <= toxicity < 3:
            toxicity = "Moderately toxic"
        elif 3 <= toxicity < 3.5:
            toxicity = "Highly toxic"
        else:
            toxicity = "Extremely toxic"

        escala = '''

Toxicity Scale:

0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting kindness and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.

1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.

2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, it seeks to hurt, ridicule, or belittle others, showing a rejection towards the diversity of opinions and people.

3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, exclude, or incite hatred, showing an explicit intention to cause harm.

4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    '''
        if channel_name:            
            if len(filtered_messages) > 0:
                self.bot.reply_to(message, "Give me a moment... 🤔")
                filtered_messages = filtered_messages[:15]

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system","""
                    Your task is to explain why a channel has been classified as {toxicity}. According to the following scale: {escala}.

                    The user will provide you a few example messages extracted from the group and the classification you must endorse and explain.
                    
                    ###
                    SAMPLE OUTPUT:
            
                        '''The messages cover various topics, including political figures, legal cases, media bias, and criminal investigations. They discuss Trump's media stock surge, RFK's VP announcement, and controversial court rulings. There are mentions of raids on Sean Combs' homes, a dog show host's arrest, and Idaho's ban on diversity statements. The content reflects strong opinions on political and legal matters.

                        The channel exhibits a 🟡 Slightly Toxic level due to the biased and emotionally charged comments present in the messages. While the discussions involve political and legal events, there is a notable presence of aggressive language and negative portrayals of individuals and groups. The toxicity stems from the strong opinions expressed, potentially influencing a confrontational atmosphere.'''
                     
                    
                                        
                    """),

                    ("user", """

                    These are some of the channel messages: {filtered_messages}

                    Explain why the group of messages belongs to the classification {toxicity}. First mention the main topics discussed in the channel, you can use one or two examples to support your explanation of the classification. 

                    """), 
                ])
                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{
                        'filtered_messages': filtered_messages,
                        'escala': escala,
                        'toxicity': toxicity
                    }])
                print(output)
                markup.add(new_analyze, go_back)
                self.bot.reply_to(message, f'{output[0]}', reply_markup=markup)

            else:
                self.bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            self.bot.reply_to(message, "Please provide a channel name!")

    def detoxify_single_message(self, message):
        self.bot.reply_to(message, "Let's see... 👀")
        markup = types.InlineKeyboardMarkup(row_width=1)
        go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
        toxicity_result = self.mistral.predictToxicity(message.text)
        _, toxicity_score = toxicity_result
        toxicity_score = str(toxicity_score)
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """As an AI language model, your task is to detoxify and provide non-toxic alternatives for the following messages if they are found to be toxic based on the provided toxicity scale.

Toxicity Scale:
0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

**Task:**
Review the provided messages and determine their toxicity levels. If the messages are Slightly, Moderately, Highly, or Extremely toxic, suggest rephrased, non-toxic versions that convey the intended messages in a respectful and positive manner.

**Examples of detoxification:**

1. **Non-toxic:**
   - Original Message: "I appreciate your perspective and would like to discuss this further."
   - Output: This message is 🟢 Non-toxic. It promotes a respectful and open dialogue.
     
2. **Slightly Toxic:**
   - Original Message: "That's a naive way of looking at things, don't you think?"
   - Output: This message is 🟡 Slightly Toxic due to its patronizing tone. A more respectful phrasing could be: "I think there might be a different way to view this situation. Can we explore that together?"

3. **Moderately Toxic:**
   - Original Message: "People who believe that are living in a fantasy world."
   - Output: This message is 🟡 Moderately Toxic because it dismisses others' beliefs. A less toxic version could be: "I find it hard to agree with that perspective, but I'm open to understanding why people might feel that way."

4. **Highly Toxic:**
   - Original Message: "This is the dumbest idea I've ever heard."
   - Output: The message is 🔴 Highly Toxic due to its derogatory language. A constructive alternative might be: "I have some concerns about this idea and would like to discuss them further."

5. **Extremely Toxic:**
   - Original Message: "Anyone who supports this policy must be a complete idiot. We should kill them all, they don't deserve to exist."
   - Output: This message is 🔴 Extremely Toxic and offensive. A non-toxic rephrasing could be: "I'm surprised that there's support for this and would like to understand the reasoning behind it."

Now, please detoxify the following message:
    """),
    ("user", message.text),
    ("system", f"Toxicity level: {toxicity_score}.") 
])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{}])
        print(output)
        markup.add(go_back)
        self.bot.reply_to(message, f'{output[0]}', reply_markup=markup)

        
    