from datasets import load_dataset
import json
import os
import sys
import contextlib
import numpy as np
from llama_cpp import Llama
import guidance
import subprocess
from .generic_classifier import Classifier



class llama_cpp_classifier(Classifier):
    def __init__(self, model_path, optimized = True, verbosity = False):
        self.model_path = model_path
        self.verbosity = verbosity
        if (not optimized): self.model = Llama(model_path = self.model_path, verbose=verbosity, n_ctx=1024, model_type="llama", n_threads = 16, n_gpu_layers=-1, seed = 1) #se puede mover la ventana de contexto
        else:
            
            process = subprocess.Popen(self.createCacheCommand(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    def test_model(self, message):
        llm = self.model
        output = self.generate_text_from_prompt(llm, message, echo = True, max_tokens = 1000, stop = ["Q", "/n/n", "[INST]"])["choices"][0]["text"]
        print(output)

        

    def predictToxicity(self, input_message):
        llm = self.model
        print("Clasificando el mensaje: \n")
        print(input_message)
        prompt = self.createPrompt(input_message)
        output = self.generate_text_from_prompt(llm, prompt, echo = False, max_tokens = 5000)["choices"][0]["text"]

        try:
            toxicity_score = int(output)
        except:
            print("El modelo está devolviendo niveles de toxicidad no numeros. Revisar el prompt o el modelo")
            print(output)
            return  False, -1
        isToxic = False
        if (toxicity_score >= 2): isToxic = True

        return  isToxic, toxicity_score

    def generate_text_from_prompt(self,llm, user_prompt, max_tokens = 300, temperature = 0.0, top_p = 0.1, echo = True, stop = ["Q"]):
         model_output = llm(
         user_prompt,
         max_tokens=max_tokens,
         temperature=temperature,
         top_p=top_p,
         echo=echo,
         stop=stop,
                )
         return model_output
        
    def createPrompt(self, message):
        prompt_template = f"""<s>[INST] You are moderator of toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and agressiveness. Reply only with a number from 0 to 4, no additional characters. Use the following scale:

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
                ###
                Inquiry:
                Classify the following message: {message}[/INST]"""
        return prompt_template

        
    def createPrompt_COT(self, message):

        prompt_template= f"""[INST]
          You are moderator of polarizing and toxic content in chat groups. Your task is to classify the user's inquiry after <<<>>> in a scale from 0 to 4 according to their level of toxicity and aggressiveness. Think of the problem step by step. First write a REASONING on the content of the message and its relation with the provided categories. After that, Reply with a number from 0 to 4 according to the selected category. Use the following scale:

                    0- **Non-Toxic and Non-Polarizing**: Language is respectful, inclusive, and considerate. It fosters constructive dialogue and understanding, without any negative undertones or divisive intent.

                    1- **Slightly Toxic or Polarizing**: Language may contain mild negative or biased statements, but they are not intended to harm or deeply divide. This level may include subtle jabs, passive-aggressive comments, or slight bias, but generally maintains a level of respect.

                    2- **Moderately Toxic or Polarizing**: Language at this level is noticeably negative, biased, or divisive. It may include overt sarcasm, targeted insults, or language that dismisses or belittles certain groups or viewpoints. Respect for differing opinions starts to wane.

                    3- **Highly Toxic or Polarizing**: Language is clearly harmful, disrespectful, and divisive. It includes hate speech, explicit insults, dehumanizing language, or rhetoric that significantly disparages or demonizes certain groups or perspectives. The intent to harm or alienate is evident.

                    4- **Extremely Toxic and Polarizing**: This is the most severe level, where language is overtly hostile, incites hatred or violence, and includes severe personal attacks, threats, or egregious hate speech. The language at this level is not only deeply harmful but could also lead to real-world consequences and conflicts.

                    ####
                    Here you have some examples:

                    Inquiry: <<Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.

                    Anna Sevidova was injured when her home was shelled in Ukraine. Now she’s recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.

                    As Ukrainian refugees flood into Moldova, one of Europe’s poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova’s population.

                    The New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more

                    @nytimes>>

                    assistant: 
                    - REASONING - This message discusses the situation of Ukrainian refugees in Moldova in a factual and neutral tone. It mentions the strain on Moldova due to the influx of refugees but does so without any negative undertones or divisive intent. The message aims to inform and provide insight into the challenges faced by both refugees and the host country. Based on the given scale, this message is **Non-Toxic and Non-Polarizing** because it promotes understanding and awareness without any disrespect or harmful intent.
                    - CLASSIFICATION - 0

                    


                    Inquiry: <<Classify the following message: These people are so much stoopiderer than I ever thought possible.!>>
                    assistant: 
                    - REASONING - The message uses a derogatory term "stoopiderer" to describe a group of people, which indicates a negative and biased viewpoint. The language is disrespectful and dismissive, showing a clear lack of respect for the individuals being referred to. However, it does not contain explicit hate speech, threats, or severe personal attacks. Thus, it falls under **Moderately Toxic or Polarizing** because it includes targeted insults and language that dismisses certain groups, but it doesn't reach the level of hate speech or inciting violence.
                    - CLASSIFICATION - 2

                    


                    Inquiry: <<Classify the following message: They're not happy in Gaza .. \nThey're not happy in Egypt .. \nThey're not happy in  Libya .. \nThey're not happy in Morocco .. \nThey're not happy in Iran .. \nThey're not happy in Iraq .. \nThey're not happy in Yemen ... \nThey're not happy in Afghanistan ... \nThey're not happy in Pakistan .. \nThey're not happy in Syria .. \nThey're not happy in Lebanon ...\n SO.. WHERE ARE THEY HAPPY?\n They're happy in Australia .. \nThey're happy in Canada .. \nThey're happy in England .. \nThey're happy in Ireland...\nThey're happy in France .. \nThey're happy in Italy .. \nThey're happy in Germany .. \nThey're happy in Sweden .. \nThey're happy in the USA .. \nThey're happy in Norway .. \nThey're happy in Holland .. \nThey're happy in Denmark ..\n Basically, they're happy in every country that is not Muslim and unhappy in every country that is!\n AND WHO DO THEY BLAME?\n Not Islam. \nNot their leadership. \nNot themselves\n THEY BLAME THE COUNTRIES THEY ARE HAPPY IN !\n AND THEN- They want to change those countries to be like, THE COUNTRY THEY CAME FROM WHERE THEY WERE UNHAPPY!\n Excuse me, but I can't help wondering... \nHow frigging dumb can you get? \nEveryone seems to be wondering why Muslim Terrorists are so quick to commit suicide.\n Lets have a look at the evidence: \n- No Christmas \n- No television \n- No nude women \n- No football \n- No pork chops \n- No hot dogs \n- No burgers \n- No beer \n- No bacon \n- Rags for clothes \n- Towels for hats \n- Constant wailing from some bloke in a tower \n- More than one wife \n- More than one mother-in-law \n- You can't shave \n- Your wife can't shave \n- You can't wash off the smell of donkeys \n- You cook over burning camel shit \n- Your wife is picked by someone else for you \n- and your wife smells worse than your donkey \n- Then they tell them that \"when they die, it all gets better\"???\n Well No Shit Sherlock!.... \nIt's not like it could get much worse!>>
                    assistant: 
                    - REASONING - This message leverages stereotypes and derogatory assumptions about Muslim-majority countries, using a narrative that is highly offensive and polarizing. It mocks cultural and religious practices and promotes bias and xenophobia. By trivializing serious issues such as suicide and terrorism and promoting division, the message surpasses mere toxicity, aiming to foster hostility and misunderstanding between different groups. Given its explicit intent to incite division and its use of offensive language, the content is classified as Extremely Toxic and Polarizing.
                    - CLASSIFICATION -  4
                    ####
                    [/INST] Sure! I can be  a moderator that outputs both my reasoning and my classification in that specific format. What message would you like me to classify?
                    [INST]Please classify the following message: {message} [/INST]
                    """

        return prompt_template
