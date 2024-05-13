import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from telebot import types
import json
import os

class explainer:
    def __init__(self, bot, loop, formatter, mistral, bert, output_parser, user_management):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.mistral = mistral
        self.bert = bert
        self.output_parser = output_parser
        self.llm = mistral.chat
        self.user_management = user_management
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))

    
    def write_cache(self, user_id, channel_name, average_toxicity_score):
        cache_file_path = os.path.join(self.cache_dir, "explainer_cache.json")
        try:
            with open(cache_file_path, 'r') as cache_file:
                cache = json.load(cache_file)
        except FileNotFoundError:
            cache = {}

        cache[channel_name] = {
            'user_id': user_id,
            'channel_name': channel_name,
            'average_toxicity_score': average_toxicity_score
        }

        with open(cache_file_path, 'w') as cache_file:
            json.dump(cache, cache_file)
            print("Cache updated successfully!")

    def explain(self, message):
        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        toxicity = state.last_analyzed_toxicity

        markup = types.InlineKeyboardMarkup(row_width=1)
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        new_analyze = types.InlineKeyboardButton('Analyze another channel 🔍', callback_data='analyze')
        toxicity_ = types.InlineKeyboardButton('Toxicity dimensions 📊', callback_data='learn_more')
        if 0 <= toxicity < 1:
            toxicity = "🟢 Non-toxic"
        elif toxicity >= 1 and toxicity < 1.75:
            toxicity = "🟡 Slightly toxic"
        elif 1.75 <= toxicity < 2.5:
            toxicity = "🟡 Moderately toxic"
        elif 2.5 <= toxicity < 3.5:
            toxicity = "🔴 Highly toxic"
        else:
            toxicity = "🔴 Extremely toxic"

        escala = '''

                    Toxicity Scale:

                    0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting kindness and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.

                    1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.

                    2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, it seeks to hurt, ridicule, or belittle others, showing a rejection towards the diversity of opinions and people.

                    3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, exclude, or incite hatred, showing an explicit intention to cause harm.

                    4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    
                    '''
        if state.last_channel_analyzed:            
            if len(state.last_chunk_of_messages) > 0:
                self.bot.reply_to(message, "Give me a moment... 🤔")
                filtered_messages = state.last_chunk_of_messages[:15]

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system","""
                    <s>[INST] Your task is to explain why a channel has been classified as {toxicity}. According to the following scale: {escala}.

                    The user will provide you a few example messages extracted from the group and the classification you must endorse and explain.
                    
                    ###
                    EXAMPLES:
                    1. 🟢 Non-toxic example:
                    '''🟢 This channel maintains a Non-toxic environment by promoting constructive dialogue and community engagement. The messages are informative, encouraging, and inclusive, fostering positive interactions.
 
                    📝 The channel features a range of topics including including political figures, community events, international politics and news updates. It highlights different initiatives like charity fundraisers, and even controversial issues are discussed in a respectful way.

                    🗣 The discussions encourage members to participate actively and supportively. The atmosphere is friendly and welcoming, with a focus on building community ties and offering help where needed.'''
                    
                    2. 🟡 Slightly toxic example:
                    '''🟡 The channel exhibits a Slightly Toxic level due to the biased and emotionally charged comments present in the messages. 

                    📝 The messages cover various topics, including political figures, legal cases, media bias, and criminal investigations. They discuss Trump's media stock surge, RFK's VP announcement, and controversial court rulings.
                     
                    🗣 While the discussions involve political and legal events, there is a notable presence of aggressive language and negative portrayals of individuals and groups. The toxicity stems from the emotionally charged opinions expressed, potentially influencing a confrontational atmosphere.'''

                    3. 🟡 Moderately toxic example:
                    '''🟡 The channel is Moderately Toxic due to frequent use of harsh language and occasional derogatory remarks towards specific groups or individuals. The tone is often confrontational, which may alienate some participants.
                    
                    📝 Topics discussed include sports, political debates, media bias, and social issues. Messages often focus on contentious subjects like immigration policy, gun control, and electoral reforms.

                    🗣 Discussions are heated and include strong criticisms of political figures and policies, with some users expressing frustration in hostile ways. The environment can be unwelcoming to those with differing viewpoints, leading to polarized discussions.'''

                    4. 🔴 Highly toxic example:
                    '''🔴 The channel displays a Highly Toxic level with frequent use of offensive language and clear contempt for individuals or groups based on their identity or beliefs. The conversations are marked by negativity and hostility.
                     
                    📝 The discussions in this channel revolve around highly polarizing and sensitive topics such as religious conflicts, racial tensions, and extreme political views. It includes derogatory terms and insults targeted at specific groups.

                    🗣 The tone is overtly aggressive, with users engaging in personal attacks and using insults to demean others. This kind of discourse creates a hostile environment that discourages constructive communication and could incite further conflict.'''

                    5. 🔴 Extremely toxic example:
                    '''🔴 The Extremely Toxic level of the channel is evident through the aggressive disrespect and threats made in the messages. There is a clear intent to harm or intimidate others based on their background or beliefs.

                    📝 This channel contains discussions that often escalate into threats and calls for violence against specific groups or individuals. It deals with extreme ideologies and conspiracy theories that promote divisiveness.

                    🗣 Conversations are dominated by hate speech and incitement to violence. Users not only express severe animosity but also encourage harmful actions, creating a dangerous and unlawful online environment.'''
                     
                    ###
                    FORMAT EXAMPLE
                    🟢 Non-toxic / 🟡 Slightly or Moderately toxic / 🔴 Highly or Extemely toxic + [Classification reason]
                     
                    📝 [Main topics discussed]
                     
                    🗣 [Consequences for the user]
                     
                    [INST]"""),

                    ("user", """

                    <s>[INST]These are some of the channel messages: {filtered_messages}

                    1- Mention the classification {toxicity} and explain the reason for that classification. 2- 📝 Mention the main topics discussed in the channel. 3- 🗣 Finally explain the consequences for the user. Use 2 sentences for each paragraph. Remember to follow the format examples provided in the system prompt. Do your best, this is very important for my career. Be straightforward and concise. No yapping.[INST] 

                    """), 
                ])
                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{
                        'filtered_messages': filtered_messages,
                        'escala': escala,
                        'toxicity': toxicity
                    }])
                print(output)
                markup.add(new_analyze, toxicity_, go_back)
                self.bot.reply_to(message, f'{output[0]}', reply_markup=markup, parse_mode='Markdown')
                self.write_cache(user_id, state.last_channel_analyzed, toxicity)
            else:
                self.bot.reply_to(message, "Failed! Try with another channel!")
        else:
            self.bot.reply_to(message, "Please provide a channel name!")
        state.is_explaining = False

    def detoxify_single_message(self, message):
        self.bot.reply_to(message, "Let's see... 👀")
        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        markup = types.InlineKeyboardMarkup(row_width=1)
        detoxify_new = types.InlineKeyboardButton('Detoxify another message 📩', callback_data='detoxify')
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        toxicity_result = self.mistral.predictToxicity(message.text)
        _, toxicity_score = toxicity_result
        toxicity_score = str(toxicity_score)
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """<s>[INST] You are a moderator of online content, your task is to detoxify and provide non-toxic alternatives for messages, if they are found to be toxic. If the message only contains instuls, say: "This message has no informative content and is simply an insult, therefore, there's no relevant information here to detoxify." Keep your rephrasing as close to the original message as possible.


            Toxicity Scale:
            0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
            1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
            2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
            3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
            4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

            **Task:**
            Revise the provided message. If the message is 🟡 Slightly Toxic, 🟡 Moderately Toxic,  🔴 Highly Toxic, or 🔴 Extremely Toxic, suggest a rephrased non-toxic version that conveys the meaning of the original message in a more respectful and positive manner. Maintain the intention of the original message DO NOT add phrases like "I'd be interested in discussing this further" if the user didn't say that. 

            **Examples of detoxification:**

            1. **Non-toxic:**
            - Original Message: Now, please detoxify the following message which has a toxicity level of {toxicity}: [[["I appreciate your perspective and would like to discuss this further."]]]
            - Output: This message is 🟢 Non-toxic. It promotes a respectful and open dialogue.
                
            2. **Slightly Toxic:**
            - Original Message: Now, please detoxify the following message which has a toxicity level of {toxicity}:[[["That's a naive way of looking at things, don't you think?"]]]
            - Output: This message is 🟡 Slightly Toxic due to its patronizing tone. 
     
            A more respectful phrasing could be: "Could there be a more comprehensive way of looking at it?"

            3. **Moderately Toxic:**
            - Original Message: Now, please detoxify the following message which has a toxicity level of {toxicity}:[[["People who believe that are living in a fantasy world."]]]
            - Output: This message is 🟡 Moderately Toxic because it dismisses others' beliefs. 
            
            A less toxic version could be: "I find it hard to agree with that perspective, I think it's unrealistic."

            4. **Highly Toxic:**
            - Original Message: Now, please detoxify the following message which has a toxicity level of {toxicity}:[[["This is the dumbest idea I've ever heard."]]]
            - Output: The message is 🔴 Highly Toxic due to its derogatory language. 
            
            A constructive alternative might be: "I don't think that idea is the best approach at all."

            5. **Extremely Toxic:**
            - Original Message: Now, please detoxify the following message which has a toxicity level of {toxicity}:[[["Anyone who supports this policy must be a complete idiot. We should kill them all, they don't deserve to exist."]]]
            - Output: This message is 🔴 Extremely Toxic and offensive. 
            
            A non-toxic rephrasing could be: "I'm surprised that there's support for this policy. I have a completely different point of view"[INST]

    """),
    ("user", "<s>[INST] Now, please detoxify the following message which has a toxicity level of {toxicity}: [[[ " + message.text + "]]][INST]")
])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{toxicity_score}])
        print(output)
        markup.add(detoxify_new, go_back)
        self.bot.reply_to(message, f'{output[0]}', reply_markup=markup, parse_mode='Markdown')
        state.is_detoxifying = False
        
    
