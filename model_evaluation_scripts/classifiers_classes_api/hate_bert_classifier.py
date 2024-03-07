from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import sys
import contextlib
from .generic_classifier import Classifier

class hate_bert_classifier(Classifier):
	def __init__(self, model_path, verbosity = False):
		self.model = BertForSequenceClassification.from_pretrained(model_path)
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.verbosity = verbosity

	def predictToxicity(self, input_message):
		input_message = input_message[0:512]  #no me qiero pasar de los 512 tokens de bert
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
	
		return  isToxic, (toxicity_score *4) #normalizo todo a un score sobre 4
	
	def filter_toxic_messages(self, messages):
		#TODO batching para aumentar eficiencia
		#TODO emprolijar
		input_ids = self.tokenizer(messages, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True, padding=True)
		print(input_ids)
		outputs = self.model(**input_ids)
		logits = outputs.logits
		probas = F.softmax(logits, dim=1)
		res = []
		for i in range(0, len(probas)):
			if torch.argmax(probas[i]).item() == 1: res.append(self.tokenizer.decode(input_ids["input_ids"][i][1:-1], skip_special_tokens = True))
			
		return res







#bert_classifier = hate_bert_classificator("toxigen_hatebert", verbosity = True)
#bert_classifier.predictToxicityFile('Benjaminnorton_processed.json')