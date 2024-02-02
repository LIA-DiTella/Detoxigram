from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import sys
import contextlib

class Classifier:
	def __init__(self, model_path, verbosity = False):
		self.model = BertForSequenceClassification.from_pretrained(model_path)
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.verbosity = verbosity
		
	def predictToxicity(self, input_message):
		#must be implemented in child classes
		pass

	def predictToxicityFile(self, file_path): #precondicion, file_path es el nombre de un archivo en la carpeta datasets
			script_dir = os.path.dirname(os.path.realpath(__file__))
			relative_path = os.path.join('..', '..', 'dataset/', file_path)
			absolute_path = os.path.join(script_dir, relative_path)
			with contextlib.redirect_stdout(None): #me molesta este output
				telegram_data = load_dataset( "json", data_files=absolute_path)
			predicted_toxic_messages = 0
			predicted_toxicity_scores = 0
			for m in telegram_data["train"]:
				message = m["message"] if len(m["message"]) > 0 else None
				if message is None: break
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
