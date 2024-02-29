from datasets import load_dataset
import json
import os
import sys
import contextlib

from googleapiclient import discovery
from time import sleep
from .generic_classifier import Classifier


class perspective_classifier(Classifier):

	def __init__(self, API_KEY, attributes = ["TOXICITY"], verbosity = False):
		self.key = API_KEY
		self.client = discovery.build(
	  	"commentanalyzer",
	  	"v1alpha1",
	  	developerKey=API_KEY,
	  	discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
	  	static_discovery=False,
	  )
		self.multiple_attributes = len(attributes) > 1
		self.attributes = attributes
		self.verbosity = verbosity

	def predictToxicity(self, input_message):

		sleep(1) 	#no quiero pasarme de los requests necesarios si le pegamos mucho
		attributes_dicc = {}
		for attribute in self.attributes:
			attributes_dicc[attribute] = {}

		analyze_request = {
  			'comment': { 'text': input_message },
  			'requestedAttributes': attributes_dicc,
  			"languages": ["en"]
		}

		try:
			response = self.client.comments().analyze(body=analyze_request).execute()
			scores = response["attributeScores"]
		except Exception as e:
			print("Hubo un problema con el siguiente request: \n")
			print(e)
			print(input_message)
			sleep(5)
			##wrap around si tenemos un problema, devolver que el mismo no es clasificado como toxico.
			toxicity_score = 2.5 #Numero que menos cambia las cosas
			return False, 0

		values = []
		for a in self.attributes:
			values.append([a, response["attributeScores"][a]["summaryScore"]["value"]])
	
		toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"] #precondicion, siempre pedimos toxocidad
		
		isToxic = False
		if (toxicity_score >= 0.5): isToxic = True
	
		if (self.multiple_attributes):
			return False, values
		else: 
			return  isToxic, (toxicity_score *4) #normalizo todo a un score sobre 5

	

	#TODO esto no funciona correctamente cuando se le piden muchos atributos
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
			print(f"Se detectaron {predicted_toxic_messages} mensajes toxicos, lo que corresponde a un {predicted_toxic_messages / dataset_size}  del total")
			print(f"La toxicidad promedio fue del {predicted_toxicity_scores /dataset_size }")


#bert_classifier = hate_bert_classificator("toxigen_hatebert", verbosity = True)
#bert_classifier.predictToxicityFile('Benjaminnorton_processed.json')