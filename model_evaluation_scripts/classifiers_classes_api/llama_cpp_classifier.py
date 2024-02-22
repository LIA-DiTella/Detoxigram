from datasets import load_dataset
import json
import os
import sys
import contextlib
import numpy as np
from llama_cpp import Llama

from .generic_classifier import Classifier



class llama_cpp_classifier(Classifier):
    def __init__(self, model_path, verbosity = False):
        self.model_path = model_path
        self.verbosity = verbosity
        

    def predictToxicity(self, input_message):
        llm = Llama(model_path = self.model_path, verbose=False, n_ctx=1024) #se puede mover la ventana de contexto
        input_message = input("Message to classify")
        prompt = self.createPrompt(input_message)
        output = self.generate_text_from_prompt(llm, prompt, echo = False, max_tokens = 5000)["choices"][0]["text"]
        try:
            toxicity_score = int(output)
        except:
            print("El modelo está devolviendo niveles de toxicidad no numeros. Revisar el prompt o el modelo")
        isToxic = False
        if (toxicity_score >= 2): isToxic = True

        return  isToxic, toxicity_score

    def generate_text_from_prompt(self,llm, user_prompt, max_tokens = 100, temperature = 0.0, top_p = 0.1, echo = True, stop = ["Q"]):
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
        prompt_template = f"""[INST] You are moderator of toxic content in chat groups. Your task is to Classify the following messages in a scale from 0 to 4 according to their level of toxicity and agressiveness. Reply only with a number from 0 to 4, no additional characters. Use the following scale:

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


                user: Classify the following message:  The JEWS ARE THE ONES RUINING THIS COUNTRY they are stealing our money, from the top down, and WE ARE LETTING IT HAPPEN. This is a war and we will fight!
                assistant: 4 
,

                user: Classify the following message: {message}[/INST]"""

        return prompt_template