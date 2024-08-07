
MESSAGES = {
    'GREETING_ES': 'Hola {name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus conversaciones, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
    'NO_GREETING_ES': 'Mmm... Que tal si empezamos con un saludo? (Ej. "Hola!")',
    'GREETING_EN': 'Hello {name}, I am Detoxigram! 👋\nMy role is to help you identify toxicity in your conversations...',
    'NO_GREETING_EN': 'Mmm... Try sending a greeting to start the conversation (E.g. "Hello!")',
    'WAITING_FOR_MSG_ES': 'Por favor, envíame el mensaje que deseas detoxificar.',
    'WAITING_FOR_FILE_ES': "Por favor, envíame el archivo .txt de la conversación que deseas analizar.",
    'WAITING_FOR_MSG_EN': 'Please, send me the message to detoxify',
    'WAITING_FOR_FILE_EN': "Please, send me the .txt of the conversation to analize",
    'POST_ANALISIS_EN' : "Please select what you'd like me to do next",
    'POST_ANALISIS_ES' : "Porfavor selecciona lo siguiente que quieres que haga"
    # Agrega más mensajes según lo necesites
}

BOTONES = {
    'GREETING_ES': [['Detoxificar un mensaje 📧','id:000'], ['Analizar una conversación 💬', 'id:001']],
    'GREETING_EN': [['Detoxify a message 📧','id:000'], ['Analize a conversation 💬', 'id:001']],
    'POST_ANALISIS_EN': [['Explain the classification', 'id:002'],['Toxicity distribution', 'id:003']],
    'POST_ANALISIS_ES': [['Explicar la clasificación', 'id:002'],['Distribución de toxicidad', 'id:003']]
}