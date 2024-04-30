from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import sys
import contextlib
import bisect
from .generic_classifier import Classifier
from huggingface_hub import snapshot_download

class multi_bert_classifier(Classifier):
	def __init__(self, model_path, toxicity_distribution_path,  verbosity = False, calculate_toxicity_distribution = False):
		#intentar descargar multibert 
		try:
			self.model = BertForSequenceClassification.from_pretrained(model_path)
		except:
			snapshot_download(repo_id="SantiagoCorley/multibert", local_dir = model_path, local_dir_use_symlinks = False)
			self.model = BertForSequenceClassification.from_pretrained(model_path)
		
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.labels = {"sarcastic" : 0, "antagonize" : 1, "condescending" : 2, "dismissive" : 3, "generalisation" : 4, "healthy" : 5, "hostile" : 6}
		self.verbosity = verbosity
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

	def predictToxicity(self, input_message):
		input_message = input_message[0:512]  #no me quiero pasar de los 512 tokens de bert
		tokens = self.tokenizer(input_message, return_tensors='pt') #tensores en formato pytorch
		if (len(tokens["input_ids"][0]) >= 512):
			print("Bert soporta mensajes de hasta 512 tokens. Este es de " + str(len(tokens["input_ids"][0])))
			sys.exit(1)
		else:
			outputs = self.model(**tokens)
			logits = outputs.logits
			probabilities = F.sigmoid(logits)[0]

		unhealthy_prediction = probabilities[self.labels["healthy"]].item()	
		isToxic = False
		if (unhealthy_prediction <= 0.5): isToxic = True

		return  isToxic, ((1 - unhealthy_prediction) *4) #normalizo todo a un score sobre 4
	
	def predict_toxicity_scores(self, input_message):
		input_message = input_message[0:512]
		tokens = self.tokenizer(input_message, return_tensors='pt')
		outputs = self.model(**tokens)
		logits = outputs.logits
		probabilities = F.sigmoid(logits)[0]
		res = {}
		for label, position  in self.labels.items(): res[label] = probabilities[position].item()
		return input_message, res 
	
	def get_group_toxicity_distribution(self, message_list):
		res = {}
		average_toxicity_scores = self.predict_average_toxicity_scores(message_list)
		#average_toxicity_scores = {"healthy": 0.66, "healthy": 0.69, "healthy": 0.991, "healthy":0.993, "sarcastic": 0.1}
		for label, score in average_toxicity_scores.items():
			index = bisect.bisect_left(self.toxicity_distribution[label], score)
			res[label] = index / len(self.toxicity_distribution[label])
		#res["healthy"] = 1 - res["healthy"] #como siempre, corrijo para que sea una clase negativa
		return res

	def predict_average_toxicity_scores(self, messages):
		valores_promedio = {}
		for key, _ in self.labels.items(): valores_promedio[key] = 0
		for txm in messages:
			scores = self.predict_toxicity_scores(txm)
			for label, score in scores[1].items(): valores_promedio[label] = valores_promedio[label] + score

		for key, _ in self.labels.items(): valores_promedio[key] = valores_promedio[key] / len(messages)
		return valores_promedio 
	
	def get_most_toxic_messages(self, messages):
		toxicity_levels = []
		for message in messages:
			toxicity_levels.append(self.predict_toxicity_scores(message))

		sorted_messages = sorted(toxicity_levels, key = lambda d : d[1]["healthy"])[0:10]
		 #me quedo solo con los comentarios
		message_list = list(map(lambda x: x[0], sorted_messages))
		return message_list
	
	def predict_toxicity_distribution(self):
		relative_path = os.path.join( '..', 'dataset/')
		files = os.listdir(relative_path)
		detected_label_distribution = {key: [] for key in self.labels}
		for f in files:
			if "testing_datasets" == f: continue
			print(f"Procesando el archivo: {f}")
			predicted_labels = self.predictToxicityFile(f)
			for label in detected_label_distribution:
				detected_label_distribution[label].append(predicted_labels[label])
		for label in detected_label_distribution: detected_label_distribution[label].sort()
		return detected_label_distribution

	
	def predictToxicityFile(self, file_path): #precondicion, file_path es el nombre de un archivo en la carpeta datasets
			script_dir = os.path.dirname(os.path.realpath(__file__))
			relative_path = os.path.join('..', '..', 'dataset/', file_path)
			absolute_path = os.path.join(script_dir, relative_path)

			with contextlib.redirect_stdout(None): #me molesta este output
				telegram_data = load_dataset( "json", data_files=absolute_path)
			predicted_toxic_messages = 0
			predicted_toxicity_scores = 0
			# print(f"Analizando {len(telegram_data["train"])} mensajes \n")

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
			#print("Los mensajes mas toxicos son: ")
			toxic_messages = self.get_most_toxic_messages([x["message"] for x in telegram_data["train"]])
			print(toxic_messages)
			print("Los valores promedio de los diferentes scores son: \n")
			average_scores = self.predict_average_toxicity_scores(toxic_messages)
			#print(average_scores)
			return average_scores
