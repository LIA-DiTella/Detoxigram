
from datasets import load_dataset

import os

from classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from classifiers_classes_api.perspective_classifier import perspective_classifier
from classifiers_classes_api.gpt_classifier import gpt_classifier
from classifiers_classes_api.llama_cpp_classifier import llama_cpp_classifier

def main():
	script_dir = os.path.dirname(os.path.realpath(__file__))
	relative_path = os.path.join('..', 'dataset/')
	files = os.listdir(relative_path)

	g_classifier = llama_cpp_classifier("classifiers_classes_api/minstral/mistral-7b-v0.1.Q5_K_S.gguf")

	g_classifier.predictToxicity("i love vegans")
	
	# for f in files:
	# 	print(f"Procesando el archivo: {f}")

if __name__ == "__main__":
	main()