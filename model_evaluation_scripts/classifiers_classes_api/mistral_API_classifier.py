from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import time
import os
import sys
import contextlib
from .generic_classifier import Classifier
from dotenv import load_dotenv
load_dotenv()


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI



class mistral_large_classifier(Classifier):
    def __init__(self, templatetype, verbosity = False):
        self.mistral_api_key= os.getenv('MISTRAL_API_KEY')
        self.chat = ChatMistralAI(model= "mistral-large-latest", mistral_api_key= self.mistral_api_key, temperature= 0)
        self.verbosity = verbosity
        self.templatetype = templatetype
        

    def predictToxicity(self, input_message):
        llm = self.chat
        output_parser = StrOutputParser()
        prompt = self.createPrompt((self.templatetype))


        chain = prompt | llm | output_parser
        
        toxicity_score = int(chain.batch([{"message": input_message}])[0])

        isToxic = False
        if (toxicity_score >= 2): isToxic = True

        return  isToxic, toxicity_score
        
    def createPrompt(self, template_type):
            """
            Creates a prompt template based on the provided template type and data.

            Args:
                template_type: The type of prompt template to create.
                data: Optional data to be used in the prompt template.

            Returns:
                A ChatPromptTemplate object representing the prompt.

            Raises:
                ValueError: If an invalid template type is provided.
            """
            if template_type == 'prompt_template_few_shot':
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """<s>[INST] You are moderator of toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and agressiveness. Reply only with a number from 0 to 4, no additional characters. Use the following scale:

                    0- **Non-Toxic: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    1- **Slightly Toxic**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    2- **Moderately Toxic**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    3- **Highly Toxic**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    4- **Extremely Toxic**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                Again, answer with only one number, no other characters.

                ###
                EXAMPLES
                user: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.\n\nAnna Sevidova was injured when her home was shelled in Ukraine. Now she\u2019s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.\n\nAs Ukrainian refugees flood into Moldova, one of Europe\u2019s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova\u2019s population.\n\nThe New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more\n\n@nytimes
                assistant: 0

                user: Classify the following message: With #LloydAustin out of commission, Joe Biden for several days had no idea who his Secretary of Defense was. But then Americans right now don’t know who their President is. The country is being run by an unelected junta operating through a complicit semi-retarded figurehead 
                assistant: 3
             
                user: Classify the following message: These people are so much stoopiderer than I ever thought possible!
                assistant: 2

                user: Classify the following message:  The JEWS ARE THE ONES RUINING THIS COUNTRY they are stealing our money, from the top down, and WE ARE LETTING IT HAPPEN. This is a war and we will fight!
                assistant: 4

                ## FORMAT INSTRUCTIONS
                    Return only 1 number. Use no additional words or punctuation. 
                      
                ###
                Inquiry:
                Classify the following message: {message}[/INST]"""),

                    

                        ("user", "Classify the following message: {message}")            ])
                
            elif template_type == 'prompt_template_COT':
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """You are moderator of polarizing and toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and aggressiveness. Think of the problem step by step. First reason on the content of the message and its relation with the provided categories. After that, Reply with a number from 0 to 4 according to the selected category. Use the following scale:

                    0- **Non-Toxic and Non-Polarizing**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    1- **Slightly Toxic or Polarizing**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    2- **Moderately Toxic or Polarizing**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    3- **Highly Toxic or Polarizing**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    4- **Extremely Toxic and Polarizing**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                    ## EXAMPLES

                    {{user: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.

                    Anna Sevidova was injured when her home was shelled in Ukraine. Now she’s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.

                    As Ukrainian refugees flood into Moldova, one of Europe’s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova’s population.

                    The New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more

                    @nytimes
                    assistant: - REASONING - This message discusses the situation of Ukrainian refugees in Moldova in a factual and neutral tone. It mentions the strain on Moldova due to the influx of refugees but does so without any negative undertones or divisive intent. The message aims to inform and provide insight into the challenges faced by both refugees and the host country. Based on the given scale, this message is **Non-Toxic and Non-Polarizing** because it promotes understanding and awareness without any disrespect or harmful intent.
                    - CLASSIFICATION - 0}}


                    {{user: Classify the following message: These people are so much stoopiderer than I ever thought possible.!
                    assistant: - REASONING - The message uses a derogatory term "stoopiderer" to describe a group of people, which indicates a negative and biased viewpoint. The language is disrespectful and dismissive, showing a clear lack of respect for the individuals being referred to. However, it does not contain explicit hate speech, threats, or severe personal attacks. Thus, it falls under **Moderately Toxic or Polarizing** because it includes targeted insults and language that dismisses certain groups, but it doesn't reach the level of hate speech or inciting violence.
                    - CLASSIFICATION - 2}}


                    {{user: Classify the following message: They're not happy in Gaza .. \nThey're not happy in Egypt .. \nThey're not happy in  Libya .. \nThey're not happy in Morocco .. \nThey're not happy in Iran .. \nThey're not happy in Iraq .. \nThey're not happy in Yemen ... \nThey're not happy in Afghanistan ... \nThey're not happy in Pakistan .. \nThey're not happy in Syria .. \nThey're not happy in Lebanon ...\n SO.. WHERE ARE THEY HAPPY?\n They're happy in Australia .. \nThey're happy in Canada .. \nThey're happy in England .. \nThey're happy in Ireland...\nThey're happy in France .. \nThey're happy in Italy .. \nThey're happy in Germany .. \nThey're happy in Sweden .. \nThey're happy in the USA .. \nThey're happy in Norway .. \nThey're happy in Holland .. \nThey're happy in Denmark ..\n Basically, they're happy in every country that is not Muslim and unhappy in every country that is!\n AND WHO DO THEY BLAME?\n Not Islam. \nNot their leadership. \nNot themselves\n THEY BLAME THE COUNTRIES THEY ARE HAPPY IN !\n AND THEN- They want to change those countries to be like, THE COUNTRY THEY CAME FROM WHERE THEY WERE UNHAPPY!\n Excuse me, but I can't help wondering... \nHow frigging dumb can you get? \nEveryone seems to be wondering why Muslim Terrorists are so quick to commit suicide.\n Lets have a look at the evidence: \n- No Christmas \n- No television \n- No nude women \n- No football \n- No pork chops \n- No hot dogs \n- No burgers \n- No beer \n- No bacon \n- Rags for clothes \n- Towels for hats \n- Constant wailing from some bloke in a tower \n- More than one wife \n- More than one mother-in-law \n- You can't shave \n- Your wife can't shave \n- You can't wash off the smell of donkeys \n- You cook over burning camel shit \n- Your wife is picked by someone else for you \n- and your wife smells worse than your donkey \n- Then they tell them that \"when they die, it all gets better\"???\n Well No Shit Sherlock!.... \nIt's not like it could get much worse!
                    assistant: - REASONING - This message leverages stereotypes and derogatory assumptions about Muslim-majority countries, using a narrative that is highly offensive and polarizing. It mocks cultural and religious practices and promotes bias and xenophobia. By trivializing serious issues such as suicide and terrorism and promoting division, the message surpasses mere toxicity, aiming to foster hostility and misunderstanding between different groups. Given its explicit intent to incite division and its use of offensive language, the content is classified as Extremely Toxic and Polarizing.
                    - CLASSIFICATION -  4}}
                        """),

                    ("user", "Classify the following message: {message}")
                ])
            elif template_type == 'prompt_template_CARP':
                prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are moderator of polarizing and toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and aggressiveness. Think of the problem step by step. First reason on the content of the message and its relation with the provided categories. After that, Reply with a number from 0 to 4 according to the selected category. Use the following scale:

                0- **Non-Toxic and Non-Polarizing**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                1- **Slightly Toxic or Polarizing**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                2- **Moderately Toxic or Polarizing**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                3- **Highly Toxic or Polarizing**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                4- **Extremely Toxic and Polarizing**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                ## EXAMPLES

                {{user: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.

                Anna Sevidova was injured when her home was shelled in Ukraine. Now she’s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.

                As Ukrainian refugees flood into Moldova, one of Europe’s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova’s population.

                The New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more

                @nytimes
                assistant: - REASONING - This message discusses the situation of Ukrainian refugees in Moldova in a factual and neutral tone. It mentions the strain on Moldova due to the influx of refugees but does so without any negative undertones or divisive intent. The message aims to inform and provide insight into the challenges faced by both refugees and the host country. Based on the given scale, this message is **Non-Toxic and Non-Polarizing** because it promotes understanding and awareness without any disrespect or harmful intent.
                - CLASSIFICATION - 0}}


                {{user: Classify the following message: These people are so much stoopiderer than I ever thought possible.!
                assistant: - REASONING - The message uses a derogatory term "stoopiderer" to describe a group of people, which indicates a negative and biased viewpoint. The language is disrespectful and dismissive, showing a clear lack of respect for the individuals being referred to. However, it does not contain explicit hate speech, threats, or severe personal attacks. Thus, it falls under **Moderately Toxic or Polarizing** because it includes targeted insults and language that dismisses certain groups, but it doesn't reach the level of hate speech or inciting violence.
                - CLASSIFICATION - 2}}


                {{user: Classify the following message: They're not happy in Gaza .. \nThey're not happy in Egypt .. \nThey're not happy in  Libya .. \nThey're not happy in Morocco .. \nThey're not happy in Iran .. \nThey're not happy in Iraq .. \nThey're not happy in Yemen ... \nThey're not happy in Afghanistan ... \nThey're not happy in Pakistan .. \nThey're not happy in Syria .. \nThey're not happy in Lebanon ...\n SO.. WHERE ARE THEY HAPPY?\n They're happy in Australia .. \nThey're happy in Canada .. \nThey're happy in England .. \nThey're happy in Ireland...\nThey're happy in France .. \nThey're happy in Italy .. \nThey're happy in Germany .. \nThey're happy in Sweden .. \nThey're happy in the USA .. \nThey're happy in Norway .. \nThey're happy in Holland .. \nThey're happy in Denmark ..\n Basically, they're happy in every country that is not Muslim and unhappy in every country that is!\n AND WHO DO THEY BLAME?\n Not Islam. \nNot their leadership. \nNot themselves\n THEY BLAME THE COUNTRIES THEY ARE HAPPY IN !\n AND THEN- They want to change those countries to be like, THE COUNTRY THEY CAME FROM WHERE THEY WERE UNHAPPY!\n Excuse me, but I can't help wondering... \nHow frigging dumb can you get? \nEveryone seems to be wondering why Muslim Terrorists are so quick to commit suicide.\n Lets have a look at the evidence: \n- No Christmas \n- No television \n- No nude women \n- No football \n- No pork chops \n- No hot dogs \n- No burgers \n- No beer \n- No bacon \n- Rags for clothes \n- Towels for hats \n- Constant wailing from some bloke in a tower \n- More than one wife \n- More than one mother-in-law \n- You can't shave \n- Your wife can't shave \n- You can't wash off the smell of donkeys \n- You cook over burning camel shit \n- Your wife is picked by someone else for you \n- and your wife smells worse than your donkey \n- Then they tell them that \"when they die, it all gets better\"???\n Well No Shit Sherlock!.... \nIt's not like it could get much worse!
                assistant: - REASONING - This message leverages stereotypes and derogatory assumptions about Muslim-majority countries, using a narrative that is highly offensive and polarizing. It mocks cultural and religious practices and promotes bias and xenophobia. By trivializing serious issues such as suicide and terrorism and promoting division, the message surpasses mere toxicity, aiming to foster hostility and misunderstanding between different groups. Given its explicit intent to incite division and its use of offensive language, the content is classified as Extremely Toxic and Polarizing.
                - CLASSIFICATION -  4}}
                    """),

                    ("user", "Classify the following message: {message}")                ])
            else:
                raise ValueError("Invalid template type provided.")

            return prompt_template       
    
    def summarize(self, message):
        global last_channel_analyzed
        if last_channel_analyzed:
            channel_name = last_channel_analyzed
        else:
            channel_name = message.text
            last_channel_analyzed = channel_name

        if channel_name:
            self.bot.reply_to(message, f"Got it! I'll summarize {channel_name}... Please wait a moment 🙏")
            messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
            processed_messages = self.formatter.process_messages(messages)
            if len(processed_messages) > 0:
                self.bot.reply_to(message, "Give me one sec... 🤔")
                data = processed_messages[:20]
                transformed_data = self.formatter.transform_data_to_expected_format(data)
                print(transformed_data)

                message_gpt = 0
                message_bert = 0
                for msg in data:
                    toxicity_result_gpt = self.gpt.predictToxicity(msg['message'])
                    _, numeric_toxicity_gpt = toxicity_result_gpt
                    message_gpt += numeric_toxicity_gpt

                    toxicity_result_bert = self.bert.predictToxicity(msg['message'])
                    _, numeric_toxicity_bert = toxicity_result_bert
                    message_bert += numeric_toxicity_bert

                average_gpt = message_gpt / len(data)
                average_bert = message_bert / len(data)
                toxicity = (average_gpt + average_bert) / 2

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    Your tasks as a moderator are:

                    1. **Summarize Messages:** Provide a concise summary of the following messages, up to 55 words. Keep the summary objective and exclude any toxic content.
                    2. **Evaluate Toxicity:** Assess the channel's overall toxicity level based on the summarized content. Use the provided toxicity scale for guidance.

                    Toxicity Scale:

                    0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
                    1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
                    2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
                    3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
                    4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

                    **Note:** For Non-toxic content, include it directly in the summary. For Slightly to Moderately Toxic content, incorporate it carefully, filtering out any bias or offensive elements. For Highly to Extremely Toxic content, exclude the offensive details and focus on objective information. Additionally, provide a warning about the content's potential impact. After summarizing, conclude with your evaluation of the channel's overall toxicity. Consider the prevalence of toxic messages and their severity to explain whether the channel is generally toxic or not and why.

                    Your complete response, including the summary and toxicity evaluation, should be concise and informative, helping readers understand the channel's nature without exceeding the word limit for the summary.

                    ### EXAMPLES
                    1. **Non-toxic:**
                    - Conversation:
                        ```
                        User1: I think we should focus on renewable energy to combat climate change.
                        User2: I agree, it's a sustainable solution.
                        User3: Absolutely, and it can create new jobs in the green sector.
                        ```
                    - Output: Summary: Discussion on the benefits of renewable energy for climate change and job creation. Toxicity Level: 🟢 Non-toxic. The conversation is respectful and constructive.

                    2. **Slightly Toxic:**
                    - Conversation:
                        ```
                        User1: The government's economic policies are not helping the middle class.
                        User2: I disagree, I think they're beneficial in the long run.
                        User3: Maybe, but they seem to favor the wealthy right now.
                        User4: That's a typical leftist argument, always blaming the rich.
                        ```
                    - Output: Summary: Debate over the government's economic policies and their impact on the middle class. Toxicity Level: 🟡 Slightly Toxic. The conversation includes some bias and a subtle lack of appreciation for differing viewpoints.

                    3. **Moderately Toxic:**
                    - Conversation:
                        ```
                        User1: We need more investment in public healthcare.
                        User2: Agreed, everyone deserves access to healthcare.
                        User3: That's just socialist nonsense. Private healthcare is more efficient.
                        User4: You're so ignorant. Public healthcare is a basic right.
                        ```
                    - Output: Summary: Discussion on public vs. private healthcare investment. Toxicity Level: 🟡 Moderately Toxic. The conversation shows disrespect and aggression between differing opinions.

                    4. **Highly Toxic:**
                    - Conversation:
                        ```
                        User1: Immigration policies should be more inclusive.
                        User2: Definitely, diversity strengthens our society.
                        User3: Inclusive? More like inviting criminals and freeloaders.
                        User4: Watch your language! That's xenophobic and offensive.
                        ```
                    - Output: Summary: Conversation on immigration policies met with xenophobic remarks. Toxicity Level: 🔴 Highly Toxic. The conversation includes offensive language and clear contempt for inclusive views.

                    5. **Extremely Toxic:**
                    - Conversation:
                        ```
                        User1: It's important to address income inequality in our country.
                        User2: Yes, the wealth gap is a growing issue.
                        User3: Oh please, stop with the class warfare. You're just envious of the successful.
                        User4: People like you are the problem, promoting greed and division!
                        ```
                    - Output: Summary: Discussion on income inequality met with accusations of envy and greed. Toxicity Level: 🔴 Extremely Toxic. The conversation contains explicit aggression and personal attacks.

                                        """),

                    ("user", "Based on the first 100 messages provided and the toxicity level indicated, reply with a summary and your evaluation of the channel's toxicity."), 
                ]+ transformed_data + [toxicity])

                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{}])
                print(output)
                self.bot.reply_to(message, f'{output[0]}')
            else:
                self.bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            self.bot.reply_to(message, "Please provide a channel name!")
    
# mistral_classifier.predictToxicity()
