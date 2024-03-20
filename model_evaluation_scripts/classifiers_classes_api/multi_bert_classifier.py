from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os
import sys
import contextlib
from .generic_classifier import Classifier

class multi_bert_classifier(Classifier):
	def __init__(self, model_path, verbosity = False):
		self.model = BertForSequenceClassification.from_pretrained(model_path)
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
		self.labels = {"sarcastic" : 0, "antagonize" : 1, "condescending" : 2, "dismissive" : 3, "generalisation" : 4, "healthy" : 5, "hostile" : 6}
		self.verbosity = verbosity

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
		return res 










#bert_classifier = hate_bert_classificator("toxigen_hatebert", verbosity = True)
#bert_classifier.predictToxicityFile('Benjaminnorton_processed.json')