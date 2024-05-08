from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import time
import os
import sys
import re
import contextlib
import bisect
from .generic_classifier import Classifier
from .hate_bert_classifier import hate_bert_classifier
import threading


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI



class mistral_classifier(Classifier):

	def __init__(self, mistral_api_key, templatetype, toxicity_distribution_path, calculate_toxicity_distribution = False, verbosity = False):
		self.mistral_api_key= mistral_api_key
		self.chat = ChatMistralAI(model= "open-mixtral-8x7b", mistral_api_key= self.mistral_api_key)
		self.verbosity = verbosity
		self.templatetype = templatetype
		self.toxicity_distribution_path = toxicity_distribution_path
		self.toxicity_distribution = self.load_toxicity_distribution(calculate_toxicity_distribution)
		
	def load_toxicity_distribution(self, calculate_toxicity_distribution):
		if (calculate_toxicity_distribution):
			toxicity_distribution = self.predict_toxicity_distribution()
			with open(self.toxicity_distribution_path, 'w') as f:
				json.dump(toxicity_distribution, f)
		else:
			with open(self.toxicity_distribution_path, 'r') as f:
				toxicity_distribution = json.load(f)
		return toxicity_distribution
	
	# Para que sea más directo con como funciona el pipeline, en esta función debo llamar a BERT.
 	# Horrible pero lo más rápido en esta instancia
	def predict_toxicity_distribution(self):
		relative_path = os.path.join( '..', 'dataset/new_dataset/')
		files = os.listdir(relative_path)
		hate_bert = hate_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert/")
		detected_toxicity = []

		for f in files:
			if "testing_datasets" == f: continue
			print(f"Procesando el archivo: {f}")
			most_toxic_messages = hate_bert.get_most_toxic_messages_file(f) #mas toxicos segun hatebert
			if (len(most_toxic_messages) < 10):
				print("Warning: El grupo anterior analizado tienen menos de 10 mensajes válidos. Será ignorado al calular la distribución")
				continue
			average_toxicity = self.predict_average_toxicity_score(most_toxic_messages) #promedio segun mixtral	
			print(average_toxicity)
			detected_toxicity.append(average_toxicity)
		print(detected_toxicity)
		detected_toxicity.sort()
		return detected_toxicity
	
	def get_group_toxicity_distribution(self, message_list):
		res = []
		average_toxicity_scores = self.predict_average_toxicity_score(message_list)
		
		index = bisect.bisect_left(self.toxicity_distribution, average_toxicity_scores)
		res = index / len(self.toxicity_distribution) #normalizo para que quede entre 0 y 1 		
		return res

	def predictToxicity(self, input_message):
		'''
		Requires: input_message is a string
		Ensures: returns a tuple with a boolean and an integer. If it fails to predict the toxicity, it returns None in both values
		'''
		llm = self.chat
		output_parser = StrOutputParser()
		prompt = self.createPrompt((self.templatetype))

		chain = prompt | llm | output_parser
		
		output = chain.batch([{"message": input_message}])[0]
		toxicity_score = None 
		match = re.search(r'\d+', output)
		if match:
			toxicity_score = int(match.group(0))
		
		isToxic = False
		if toxicity_score is not None and toxicity_score >= 2:
			isToxic = True

		#print(input_message, "toxicity_score:", toxicity_score, "\n\n\n\n")
		return isToxic, toxicity_score,


	def concurrent_predict_toxicity(self, message, i, res, lock):
		#solo para ser usada por la fución posterior
		toxicity = self.predictToxicity(message)[1]
		with lock:
			res[i] = toxicity

	def predict_average_toxicity_score(self, messages):
	
		res = [None] * len(messages)
		threads = [None] * len(messages)
		lock = threading.Lock()

		for i in range(0, len(messages)):
			threads[i] = threading.Thread(target=self.concurrent_predict_toxicity, args=( messages[i], i, res, lock))
			threads[i].start()
			if (i == 5): time.sleep(0.5)
		
		for i in range(0, len(messages)): threads[i].join()

		average_toxicity = sum(res) / len(res)
		return average_toxicity





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

				Answer with only one number, NO explanations.

				###
				EXAMPLES
				Inquiry: Classify the following message: Ukrainians pouring into tiny Moldova find a welcoming but strained country.\n\nAnna Sevidova was injured when her home was shelled in Ukraine. Now she's recovering in a hotel turned refugee home in Moldova, facing an uncertain future for her family.\n\nAs Ukrainian refugees flood into Moldova, one of Europe's poorest countries says it is reaching a breaking point. With a population of just 2.6 million, Moldova has taken in 107,000 people, making it the highest per-capita recipient of those fleeing the war with Russia. They now account for over 4 percent of Moldova's population.\n\nThe New York Times traveled to the country in March and met Ukrainians who have recently sought refuge there. Read more\n\n@nytimes
				0

				Inquiry: Classify the following message: "With #LloydAustin out of commission, Joe Biden for several days had no idea who his Secretary of Defense was. But then Americans right now don’t know who their President is. The country is being run by an unelected junta operating through a complicit semi-retarded figurehead" 
				3
			 
				Inquiry: Classify the following message: "These people are so much stoopiderer than I ever thought possible!"
				2

				Inquiry: Classify the following message:  "The JEWS ARE THE ONES RUINING THIS COUNTRY they are stealing our money, from the top down, and WE ARE LETTING IT HAPPEN. This is a war and we will fight!"
				4

				## FORMAT INSTRUCTIONS
				Return only 1 character. Use no additional words or punctuation. For example, if you believe the message is non-toxic, reply with "0". If you believe the message is extremely toxic, reply with "4".[/INST]"""),

					

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
