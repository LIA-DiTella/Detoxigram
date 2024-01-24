
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn.functional as F

from googleapiclient import discovery
from time import sleep
import json
import os


def predictToxicityBert(input_text, model, tokenizer):
	tokens = tokenizer(input_text, return_tensors='pt') #tensores en formato pytorch
	if (len(tokens["input_ids"][0]) >= 100):
		print("mensajes muy largos")
		return False, 0
		#probabilities = [0,2] 
	else:
		outputs = model(**tokens)
		logits = outputs.logits
		probabilities = F.softmax(logits, dim=1)
		
	predicted_class = torch.argmax(probabilities, dim=1).item() # 1 es toxico

	toxicity_score = probabilities[0, 1].item() # nivel de toxicidad

	isToxic = False
	if (predicted_class == 1): isToxic = True

	return  isToxic, (toxicity_score/2)*10 #normalizo todo a un score sobre 5

def predictToxicityPerspective(input_text, client):
	analyze_request = {
  'comment': { 'text': input_text },
  'requestedAttributes': {'TOXICITY': {}}
}
	try:
		response = client.comments().analyze(body=analyze_request).execute()
		toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
	except Exception as e:
		print("Hubo un problema con el siguiente request: \n")
		print(e)
		print(input_text)
		sleep(5)
		##wrap around al problema, devolver que el mismo no es clasificado como toxico.

		toxicity_score = 2.5 #Numero que menos cambia las cosas
	isToxic = False
	if (toxicity_score >= 0.5): isToxic = True #Ver que onda este umbral

	return  isToxic, (toxicity_score/2)*10 #normalizo todo a un score sobre 5

def concatenate_chunks(chunks):
	res = ""
	for c in chunks:
		res = res + c #Usamos separador?
	return res


def main():
	#cargado del modelo BERT
	#base_model = 'HateBERT_offenseval/' este no se de donde lo saque
	model_name = "toxigen_hatebert"
	model = BertForSequenceClassification.from_pretrained(model_name) 
	
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #El tokenizer que viene con el modelo no parece andar bien, pero este deberia bastar
	
	#Perspective API
	API_KEY = 'AIzaSyBLcQ87gA8wc_960mNzT6uCiDkUWRoz6mE' #es mi API privada. En algun momento deberiamos tener alguna del proyecto
	client = discovery.build(
	  "commentanalyzer",
	  "v1alpha1",
	  developerKey=API_KEY,
	  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
	  static_discovery=False,
	  )


	script_dir = os.path.dirname(os.path.realpath(__file__))
	relative_path = os.path.join('..', 'dataset/', 'Benjaminnorton_processed.json')
	absolute_path = os.path.join(script_dir, relative_path)
	
	telegram_data = load_dataset( "json", data_files=absolute_path)
	
	
	messages_per_classification = 4
	perspective_classified_toxic_messages = 0
	bert_classified_toxic_messages = 0
	
	perspective_classified_toxic_chunks = 0
	bert_classified_toxic_chunks = 0
	
	perspective_classified_toxic_messages = 0
	bert_classified_toxic_messages = 0
	
	for t in range(0, len(telegram_data["train"]) - messages_per_classification, messages_per_classification):
		sleep(3)
	
		messages = telegram_data["train"][t: t + messages_per_classification]
		message_list = messages["message"]
		concatenated_messages = concatenate_chunks(message_list)
	
		bert_toxicity_score, predicted_toxicity_bert = predictToxicityBert(concatenated_messages, model, tokenizer)
		perspective_toxicity_score, predicted_toxicity_perspective = predictToxicityPerspective(concatenated_messages, client)
	
		if (predicted_toxicity_perspective): perspective_classified_toxic_chunks +=1
		if (predicted_toxicity_bert): bert_classified_toxic_chunks+=1
	
		for i in range(0, len(message_list)):
			bert_toxicity_score, predicted_toxicity_bert = predictToxicityBert(message_list[i], model, tokenizer)
			perspective_toxicity_score, predicted_toxicity_perspective = predictToxicityPerspective(message_list[i], client)
			print("Mensaje a predecir", message_list[i] )
			print("Prediccion de perspective para el mensaje:" + str(i), perspective_toxicity_score)
			print("Prediccion de Bert para iesimo mensaje:" +str(i), bert_toxicity_score)
	
			if (predicted_toxicity_perspective): perspective_classified_toxic_messages += 1 
			if (predicted_toxicity_bert): bert_classified_toxic_messages += 1
		
		print("Prediccion de perspective para los chunks: ", perspective_toxicity_score)
		print("Prediccion de Bert para los chunks: ", bert_toxicity_score)
		print("################################################################# \n \n")
	
	
	test_size = len(telegram_data["train"])
	print(f"Porcentaje de chunks toxicos Bert: {bert_classified_toxic_chunks / (test_size / messages_per_classification )} Perspective: {perspective_classified_toxic_chunks / (test_size / messages_per_classification )}")
	print(f"Porcentaje de mensajes toxicos segun Bert: {bert_classified_toxic_messages / test_size}  Segun Perspective: {perspective_classified_toxic_messages / test_size}")



if __name__ == "__main__":
	main()