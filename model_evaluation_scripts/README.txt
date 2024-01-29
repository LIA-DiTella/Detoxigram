Instrucciones sobre las API:


Para usar la API de perspective deben configurar su equipo y modificar la API key que esta en el codigo. 
Instrucciones:
https://developers.google.com/codelabs/setup-perspective-api#0

Si quieren usar cualquiera de los clasificadores, deben importar las clases de los clasificadores que están en la carpeta classifiers_classes_api
Ambos clasificadores, una vez inicializados, tienen funciones para predecir toxicidad. Las mismas son:

	predictToxicityFile(file_path)

función que dado el file de un archivo en el formato de Luz clasifica sus mensajes en toxicos o no

	predictToxicity(message)

función que dado un string retorna si el mismo es toxico o no (una tupla con un booleano + el valor numerico de toxicidad)



Al instanciar las clases de los clasificadores se puede especificar si se quieren usar en modo "verborragico" que manda por consola los outputs de cada mensaje
ejemplo:
	bert_classifier = hate_bert_classificator("classifiers_classes_api/toxigen_hatebert", verbosity = False)

Por ultimo, se le puede pasar a el clasificador de perspective una lista con los atributos a analizar. Por ejemplo:

	perspective_classifier = perspective_classificator("AIzaSyBLcQ87gA8wc_960mNzT6uCiDkUWRoz6mE", verbosity = False, attributes = ["TOXICITY", "FLIRTATION"])

