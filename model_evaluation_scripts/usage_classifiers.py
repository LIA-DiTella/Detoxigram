
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F
import json
import os

from hate_bert_api.hate_bert_classifier import hate_bert_classificator
from hate_bert_api.perspective_classifier import perspective_classificator

def main():
	script_dir = os.path.dirname(os.path.realpath(__file__))
	relative_path = os.path.join('..', 'dataset/')
	files = os.listdir(relative_path)

	#bert_classifier = hate_bert_classificator("hate_bert_api/toxigen_hatebert", verbosity = False)
	classifier = perspective_classificator("AIzaSyBLcQ87gA8wc_960mNzT6uCiDkUWRoz6mE", verbosity = True)
	
	for f in files:
		print(f"Procesando el archivo: {f}")
		classifier.predictToxicityFile(f)

if __name__ == "__main__":
	main()