
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import sys
import contextlib

class hate_bert_classificator():
	def __init__(self, model_path, verbosity = False):
		self.model = BertForSequenceClassification.from_pretrained(model_path)
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.verbosity = verbosity

	def predictToxicity(self, input_message):
		tokens = self.tokenizer(input_message, return_tensors='pt') #tensores en formato pytorch
		if (len(tokens["input_ids"][0]) >= 512):
			print("Bert soporta mensajes de hasta 512 tokens. Este es de " + str(len(tokens["input_ids"][0])))
			sys.exit(1)
		else:
			outputs = self.model(**tokens)
			logits = outputs.logits
			probabilities = F.softmax(logits, dim=1)
			
		predicted_class = torch.argmax(probabilities, dim=1).item() # 1 es toxico
	
		toxicity_score = probabilities[0, 1].item() # nivel de toxicidad
	
		isToxic = False
		if (predicted_class == 1): isToxic = True
	
		return  isToxic, (toxicity_score/2)*10 #normalizo todo a un score sobre 5

	def predictToxicityFile(self, file_path): #precondicion, file_path es el nombre de un archivo en la carpeta datasets
			script_dir = os.path.dirname(os.path.realpath(__file__))
			relative_path = os.path.join('..', '..', 'dataset/', file_path)
			absolute_path = os.path.join(script_dir, relative_path)

			with contextlib.redirect_stdout(None): #me molesta este output
				telegram_data = load_dataset( "json", data_files=absolute_path)

			predicted_toxic_messages = 0
			predicted_toxicity_scores = 0

			for m in telegram_data["train"]:
				message = m["message"][0:512] #no me quiero pasar de los tokens de Bert
				isToxic, toxicity_score = self.predictToxicity(message)
				if (isToxic): predicted_toxic_messages += 1
				predicted_toxicity_scores += toxicity_score

				if (self.verbosity and isToxic):
					print("El siguient mensaje fue clasificado como toxico: \n")
					print(message + "\n")
					print("Su nivel de toxicidad fue de: ", toxicity_score)
					print("##################################### \n")

			dataset_size = len(telegram_data["train"])
			print(f"Se detectaron {predicted_toxic_messages} mensajes toxicos, lo que corresponde a un {predicted_toxic_messages / dataset_size}   del total")
			print(f"La toxicidad promedio fue del {predicted_toxicity_scores /dataset_size }")


#bert_classifier = hate_bert_classificator("toxigen_hatebert", verbosity = True)
#bert_classifier.predictToxicityFile('Benjaminnorton_processed.json')