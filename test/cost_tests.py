import os
import sys

from dotenv import main
import time
import contextlib
from datasets import load_dataset
from transformers import AutoTokenizer

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
sys.path.append('..')
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier



def create_classification_prompt():
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
	return prompt_template
def get_scale():
	escala = '''

                    Toxicity Scale:

                    0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting kindness and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.

                    1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.

                    2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, it seeks to hurt, ridicule, or belittle others, showing a rejection towards the diversity of opinions and people.

                    3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, exclude, or incite hatred, showing an explicit intention to cause harm.

                    4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    
                    '''
	return escala

def length_of_tokenized_text(text, tokenizer):
	return len(tokenizer(text)["input_ids"])

#la explanation en relida no calcula un average pero quedo el nombre
def average_sent_tokens_mistral_explanation(telegram_messages, tokenizer, prompt):
	total_sent_tokens = 0
	telegram_messages = [m["message"] for m in telegram_messages["train"]]
	prompt_as_string =prompt.format(filtered_messages = telegram_messages, toxicity = "🟡 Moderately toxic", escala = get_scale())
	total_sent_tokens += length_of_tokenized_text(prompt_as_string, tokenizer)
	return total_sent_tokens

def average_sent_tokens_gpt_explanation(telegram_messages, tokenizer, prompt):
	total_sent_tokens = 0
	telegram_messages = [m["message"] for m in telegram_messages["train"]]
	prompt_as_string =prompt.format(filtered_messages = telegram_messages, toxicity = "🟡 Moderately toxic", escala = get_scale())
	total_sent_tokens += len(tokenizer.encode(prompt_as_string))
	return total_sent_tokens

#la explanation en relida no calcula un average pero quedo el nombre
def average_rcv_tokens_mistral_explanation(telegram_messages, tokenizer, chain):
	telegram_messages = [m["message"] for m in telegram_messages["train"]]

	output = chain.batch([{'filtered_messages': telegram_messages, 'escala': get_scale(), 'toxicity': "🟡 Moderately toxic"}])

	#retorno también el output de mistral para no recalcularlo más adelante
	return output, length_of_tokenized_text(output, tokenizer)

def average_rcv_tokens_gpt_explanation(answer, tokenizer):
	total_sent_tokens = 0
	total_sent_tokens += len(tokenizer.encode(answer))
	return total_sent_tokens

def average_sent_tokens_mistral_classification(telegram_messages, tokenizer, prompt):
	total_sent_tokens = 0
	for message in telegram_messages["train"]:
			prompt_as_string =prompt.format(message = message["message"])
			total_sent_tokens += length_of_tokenized_text(prompt_as_string, tokenizer)
	return total_sent_tokens

def average_sent_tokens_gpt(telegram_messages, tokenizer, prompt):
	total_sent_tokens = 0
	for message in telegram_messages["train"]:
			prompt_as_string =prompt.format(message = message["message"])
			total_sent_tokens += len(tokenizer.encode(prompt_as_string))
	
	return total_sent_tokens

def caculate_costs(tokens_per_file_input, tokens_per_file_output, cost_per_million_tokens_input, cost_per_million_tokens_output):
	#input_costs
	input_costs = 0
	for s in tokens_per_file_input:
		input_costs += cost_per_million_tokens_input * s
	input_cost_in_dollars = input_costs/1000000

	#output_costs. En teoría se genera un solo token

	output_costs = 0
	for s in tokens_per_file_output:
		output_costs += cost_per_million_tokens_output * s

	output_cost_in_dollars = output_costs/1000000


	total_cost = input_cost_in_dollars + output_cost_in_dollars
	average_cost_per_file = total_cost/len(tokens_per_file_input)

	return average_cost_per_file


def test_classification_costs(mistral, tokenizer, files, message_amount):
	#hacemos los calculos
	prompt = mistral.createPrompt((mistral.templatetype))

	mistral_tokens_per_file = []
	gpt_3_turbo_tokens_per_file = []
	gpt_4_turbo_tokens_per_file = []

	for file in files:
		if "testing_datasets" == file: continue
	
		#print(f"Procesando el archivo: {file}")
		with contextlib.redirect_stdout(None): #me molesta este output
			telegram_data = load_dataset( "json", data_files=relative_path + file)
			#me quedo con 10 mensajes cualquiera
			telegram_data["train"] = telegram_data["train"].select(range(message_amount))
		mistral_tokens_per_file.append(average_sent_tokens_mistral_classification(telegram_data, tokenizer, prompt))
		gpt_3_turbo_tokens_per_file.append(average_sent_tokens_gpt(telegram_data,tiktoken.encoding_for_model("gpt-3.5-turbo"), prompt ))
		gpt_4_turbo_tokens_per_file.append(average_sent_tokens_gpt(telegram_data,tiktoken.encoding_for_model("gpt-4-turbo"), prompt ))

		#gpt_3_turbo_tokens_per_file.append(len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(prompt)))
		#gpt_4_turbo_tokens_per_file.append(tiktoken.encoding_for_model("gpt-4-turbo").encode(prompt))


	classification_costs_mistral = caculate_costs(mistral_tokens_per_file, tokens_per_file_output=[1], cost_per_million_tokens_input = 0.7, cost_per_million_tokens_output = 0.7)
	classification_costs_gpt3_turbo = caculate_costs(gpt_3_turbo_tokens_per_file, tokens_per_file_output=[1], cost_per_million_tokens_input = 0.5, cost_per_million_tokens_output = 1.5)
	classification_costs_gpt4_turbo = caculate_costs(gpt_4_turbo_tokens_per_file, tokens_per_file_output=[1], cost_per_million_tokens_input = 10, cost_per_million_tokens_output = 30)

	print(f"Costo promedio de clasificar {message_amount} mensajes con mistral: ", classification_costs_mistral)
	print(f"Costo promedio de clasificar {message_amount} con gpt 3.5: ", classification_costs_gpt3_turbo)
	print(f"Costo promedio de clasificar {message_amount} con gpt 4 turbo: ", classification_costs_gpt4_turbo)
	#classification_costs_gpt4_turbo = caculate_costs_mistral(tokens_per_file)

def test_explanation_costs(mistral, tokenizer, files):
	#hacemos los calculos
	prompt = create_classification_prompt()
	
	mistral_sent_tokens_per_file = []
	gpt_3_turbo_sent_tokens_per_file = []
	gpt_4_turbo_sent_tokens_per_file = []

	mistral_rcv_tokens_per_file = []
	gpt_3_turbo_rcv_tokens_per_file = []
	gpt_4_turbo_rcv_tokens_per_file = []

	for file in files:
		if "testing_datasets" == file: continue
	
		#print(f"Procesando el archivo: {file}")
		with contextlib.redirect_stdout(None): #me molesta este output
			telegram_data = load_dataset( "json", data_files=relative_path + file)
			#me quedo con 10 mensajes cualquiera
			telegram_data["train"] = telegram_data["train"].select(range(10))

		mistral_sent_tokens_per_file.append(average_sent_tokens_mistral_explanation(telegram_data, tokenizer, prompt))
		explanation, expĺanation_tokens = average_rcv_tokens_mistral_explanation(telegram_data, tokenizer, chain = prompt | mistral.chat | StrOutputParser())
		mistral_rcv_tokens_per_file.append(expĺanation_tokens)

		gpt_3_turbo_sent_tokens_per_file.append(average_sent_tokens_gpt_explanation(telegram_data, tiktoken.encoding_for_model("gpt-3.5-turbo"), prompt ))
		gpt_4_turbo_sent_tokens_per_file.append(average_sent_tokens_gpt_explanation(telegram_data,tiktoken.encoding_for_model("gpt-4-turbo"), prompt ))

		gpt_3_turbo_rcv_tokens_per_file.append(average_rcv_tokens_gpt_explanation(explanation[0], tiktoken.encoding_for_model("gpt-3.5-turbo")))
		gpt_4_turbo_rcv_tokens_per_file.append(average_rcv_tokens_gpt_explanation(explanation[0], tiktoken.encoding_for_model("gpt-4-turbo")))


	classification_costs_mistral = caculate_costs(mistral_sent_tokens_per_file, mistral_rcv_tokens_per_file, cost_per_million_tokens_input = 0.7, cost_per_million_tokens_output = 0.7)
	classification_costs_gpt3_turbo = caculate_costs(gpt_3_turbo_sent_tokens_per_file, gpt_3_turbo_rcv_tokens_per_file, cost_per_million_tokens_input = 0.5, cost_per_million_tokens_output = 1.5)
	classification_costs_gpt4_turbo = caculate_costs(gpt_4_turbo_sent_tokens_per_file, gpt_4_turbo_rcv_tokens_per_file, cost_per_million_tokens_input = 10, cost_per_million_tokens_output = 30)

	print(f"Costo promedio de explicar con mistral: ", classification_costs_mistral)
	print(f"Costo promedio de explicar con gpt 3.5: ", classification_costs_gpt3_turbo)
	print(f"Costo promedio de clasificar explicar con gpt 4 turbo: ", classification_costs_gpt4_turbo)



if __name__ == "__main__":

	#cargo mistral
	main.load_dotenv()
	MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']
	print("Inicializando Mistral")
	mistral = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', toxicity_distribution_path = "../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/mistral_distribution.json", calculate_toxicity_distribution = False )
	
	#cargo el tokenizer, para medir los tokens
	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

	#cargo los datasets
	relative_path = os.path.join( '..', 'dataset/new_dataset/')
	files = os.listdir(relative_path)

	test_explanation_costs(mistral, tokenizer, files)
	test_classification_costs(mistral, tokenizer, files, message_amount=10)
	test_classification_costs(mistral, tokenizer, files, message_amount=50)