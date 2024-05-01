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
	def filter_toxic_messages(self, messages):
		#child classes can optimize if they suport batching
		toxic_messages = []
		for m in messages:
			isToxic, _ = self.predictToxicity(m)
			if isToxic: toxic_messages.append(m)
		return toxic_messages
	
	def get_most_toxic_messages(self, messages):
		toxicity_levels = []
		for message in messages:
			toxicity_levels.append((message, self.predictToxicity(message)[1]))
		
		#sort list of tuples by second element of tuple
		sorted_messages = sorted(toxicity_levels, key = lambda d : d[1])[-10:]
		 #me quedo solo con los mensajes
		message_list = list(map(lambda x: x[0], sorted_messages))
		return message_list
	
	def get_most_toxic_messages_file(self, file):
		script_dir = os.path.dirname(os.path.realpath(__file__))
		relative_path = os.path.join('..', '..', 'dataset/', file)
		absolute_path = os.path.join(script_dir, relative_path)

		with contextlib.redirect_stdout(None): #me molesta este output
			telegram_data = load_dataset( "json", data_files=absolute_path)

		messages = []
		for m in telegram_data["train"]:
			message = m["message"] if len(m["message"]) > 0 else None
			if message is None: break
			else: messages.append(message)
		return self.get_most_toxic_messages(messages)
	
	def predict_average_toxicity_score(self, messages):
		toxicity = 0
		for txm in messages: toxicity = toxicity + self.predictToxicity(txm)[1]
		return toxicity/len(messages)
	

	def predictToxicityFile(self, file_path): #precondicion, file_path es el nombre de un archivo en la carpeta datasets
			script_dir = os.path.dirname(os.path.realpath(__file__))
			relative_path = os.path.join('..', '..', 'dataset/testing_datasets/', file_path)
			absolute_path = os.path.join(script_dir, relative_path)
			with contextlib.redirect_stdout(None): #me molesta este output
				telegram_data = load_dataset( "csv", data_files=absolute_path)
			predicted_toxic_messages = 0
			predicted_toxicity_scores = 0
			for m in telegram_data["train"]:
				message = m["text"] if len(m["text"]) > 0 else None
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

	def compareToxicity(self, file_path, text_label, toxicity_label, toxicity_threshold = 3, binary = True):
		script_dir = os.path.dirname(os.path.realpath(__file__))
		relative_path = os.path.join('..', '..', 'dataset/testing_datasets/', file_path)
		absolute_path = os.path.join(script_dir, relative_path)
		with contextlib.redirect_stdout(None): #me molesta este output
			dataset = load_dataset( "csv", data_files=absolute_path, split="train")
		predicitions = []
		labels = []
		for m in dataset:
			message = m[text_label]
			predicted_toxicity = self.predictToxicity(message)[1]/4
			real_toxicity = m[toxicity_label]
			if binary:
				predicted_toxicity = 1 if predicted_toxicity >= 0.5 else 0
				real_toxicity = 1 if real_toxicity  >= toxicity_threshold else 0
			predicitions.append(predicted_toxicity)
			labels.append(real_toxicity)
		
		return predicitions, labels
	
		
