MESSAGES = {
    'GREETING_ES': 'Hola {name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus conversaciones, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
    'NO_GREETING_ES': 'Mmm... Qué tal si empezamos con un saludo? (Ej. "Hola!")',
    'GREETING_EN': 'Hello {name}, I am Detoxigram! 👋\nMy role is to help you identify toxicity in your conversations...',
    'NO_GREETING_EN': 'Mmm... Try sending a greeting to start the conversation (E.g. "Hello!")',
    'BOT_LOADED_EN': 'Bot token loaded, Detoxigram is live 🚀',
    'BOT_LOADED_ES': 'Token del bot cargado, Detoxigram está en vivo 🚀',
    'ENTERING_TESTING_MODE_EN': 'Entering testing mode. I will now output some internal information, and I will now work on downloaded channels.',
    'ENTERING_TESTING_MODE_ES': 'Entrando en modo de prueba. Ahora mostraré alguna información interna y trabajaré en canales descargados.',
    'LEAVING_TESTING_MODE_EN': 'Leaving testing mode.',
    'LEAVING_TESTING_MODE_ES': 'Saliendo del modo de prueba.',
    'GOODBYE_EN': 'Goodbye! 👋 If you need anything else, just say hi!',
    'GOODBYE_ES': '¡Adiós! 👋 Si necesitas algo más, solo saluda.',
    'NOT_SURE_EN': 'Mmm... I\'m not sure what that means, {username}. Would any of these options be helpful? 😁',
    'NOT_SURE_ES': 'Mmm... No estoy seguro de qué significa eso, {username}. ¿Alguna de estas opciones sería útil? 😁',
    'ANALYZE_START_EN': 'Great!\n\nJust so you know, when we evaluate the toxicity, we\'ll only consider the last 50 messages of the channel ⚠️\n\nNow, please provide the @ChannelName you would like to analyze 🤓',
    'ANALYZE_START_ES': '¡Genial!\n\nSolo para que lo sepas, cuando evaluamos la toxicidad, solo consideraremos los últimos 50 mensajes del canal ⚠️\n\nAhora, por favor proporciona el @NombreDelCanal que te gustaría analizar 🤓',
    'STILL_WORKING_EN': 'I\'m sorry, I\'m still working on your last request! 🕣',
    'STILL_WORKING_ES': 'Lo siento, ¡aún estoy trabajando en tu última solicitud! 🕣',
    'EXPLAIN_START_EN': 'After evaluating the content of {last_channel_analyzed}, we saw that this channel is {last_toxicity}. Now I will explain to you why, it will take a few seconds 🕣',
    'EXPLAIN_START_ES': 'Después de evaluar el contenido de {last_channel_analyzed}, vimos que este canal es {last_toxicity}. Ahora te explicaré por qué, tomará unos segundos 🕣',
    'NO_CHANNEL_TO_EXPLAIN_EN': 'I\'m sorry, I don\'t have any channel to explain. Please analyze a channel first!',
    'NO_CHANNEL_TO_EXPLAIN_ES': 'Lo siento, no tengo ningún canal para explicar. ¡Por favor, analiza un canal primero!',
    'WAITING_FOR_MSG_EN': 'Great! Now, please send the message you would like to detoxify 🤓',
    'WAITING_FOR_MSG_ES': '¡Genial! Ahora, por favor envía el mensaje que te gustaría desintoxicar 🤓',
    'HELP_TEXT_EN': '''Welcome to Detoxigram! 🌟 Here's how you can use our bot to make your Telegram experience safer:

1. **Analyze a Channel:** To start analyzing a channel for toxic content, simply tap the 'Analyze a Channel 🔍' button. Then, enter the @ChannelName or send the invite to the channel. I'll check the last 50 messages and let you know how toxic the conversations are.

2. **Explain Toxicity:** Curious about why a channel was rated a certain way? Tap the 'Explain why 📝' button after analyzing a channel. I'll provide you with a summary of the channel's messages, highlighting specific examples of toxicity. This helps you understand the context and specifics of the content I've analyzed.

3. **Detoxify a Message:** Want to clean up a specific message? Use the 'Detoxify a message📩' option to send me a message you think is problematic. I'll offer a less toxic version, providing a cleaner, more respectful alternative.

Need more help or have any questions? Don't hesitate to reach out. You can contact us directly at malbaposse@mail.utdt.edu. We're here to help make your digital spaces safer! 🛡️''',
    'HELP_TEXT_ES': '''¡Bienvenido a Detoxigram! 🌟 Aquí tienes cómo puedes usar nuestro bot para hacer que tu experiencia en Telegram sea más segura:

1. **Analizar un canal:** Para empezar a analizar un canal en busca de contenido tóxico, simplemente toca el botón 'Analizar un canal 🔍'. Luego, ingresa el @NombreDelCanal o envía la invitación al canal. Revisaré los últimos 50 mensajes y te haré saber qué tan tóxicas son las conversaciones.

2. **Explicar la toxicidad:** ¿Tienes curiosidad sobre por qué un canal fue calificado de cierta manera? Toca el botón 'Explicar por qué 📝' después de analizar un canal. Te proporcionaré un resumen de los mensajes del canal, destacando ejemplos específicos de toxicidad. Esto te ayudará a comprender el contexto y los detalles específicos del contenido que he analizado.

3. **Desintoxicar un mensaje:** ¿Quieres limpiar un mensaje específico? Usa la opción 'Desintoxicar un mensaje📩' para enviarme un mensaje que crees que es problemático. Te ofreceré una versión menos tóxica, proporcionando una alternativa más limpia y respetuosa.

¿Necesitas más ayuda o tienes alguna pregunta? No dudes en contactarnos directamente en malbaposse@mail.utdt.edu. ¡Estamos aquí para ayudarte a que tus espacios digitales sean más seguros! 🛡️''',
    'TOXICITY_ANALYSIS_START_EN': 'We\'ve just classified the channel you sent. I will send you a more detailed analysis of the channel shortly 📊',
    'TOXICITY_ANALYSIS_START_ES': 'Acabamos de clasificar el canal que enviaste. Te enviaré un análisis más detallado del canal en breve 📊',
    'CACHE_UPDATED_EN': 'Cache updated successfully!',
    'CACHE_UPDATED_ES': '¡Cache actualizado con éxito!',
    'DETOXIFY_START_EN': 'Great! ⚠️ Now, please write a message you would like to detoxify 🤓',
    'DETOXIFY_START_ES': '¡Genial! ⚠️ Ahora, por favor escribe un mensaje que te gustaría desintoxicar 🤓',
    'HERE_ARE_MORE_OPTIONS_EN': 'Here are some more options! 🤓',
    'HERE_ARE_MORE_OPTIONS_ES': '¡Aquí hay algunas opciones más! 🤓',
    'WHAT_NOW_EN': 'Alright! What would you like to do now? 🤔',
    'WHAT_NOW_ES': '¡Muy bien! ¿Qué te gustaría hacer ahora? 🤔',
    'NOT_VALID_CHANNEL_ES': '¡Ups! Ese no es un nombre de canal válido. ¡Inténtalo de nuevo! 🫣',
    'NOT_VALID_CHANNEL_EN': 'Oops! That is not a valid channel name. Try again! 🫣',
    'ANALYZE_RESPONSE_ES': '¡Entendido! Analizaré {channel_name}... Por favor espera un momento 🙏',
    'ANALYZE_RESPONSE_EN': 'Got it! I will analyze {channel_name}... Please wait a moment 🙏',
    'NO_MESSAGES_FOUND_ES': '¡No se encontraron mensajes en el canal especificado! ¿Por qué no empezamos de nuevo?',
    'NO_MESSAGES_FOUND_EN': 'No messages found in the specified channel! Why don\'t we start again?',
    'TOOK_LONGER_ES': 'Eso tomó más tiempo del esperado... Dame un segundo para verificar la toxicidad en el canal 🕣',
    'TOOK_LONGER_EN': 'That took longer than expected... Now give me a second to check for toxicity in the channel 🕣',
    'FAILED_TOXICITY_ANALYSIS_ES': '¡Fallido! ¡Inténtalo con otro canal!',
    'FAILED_TOXICITY_ANALYSIS_EN': 'Failed! Try with another channel!',
    'SEND_RESPONSE_MESSAGE_ES': '{channel_name} es: {emoji} {answer} \n\n¿Quieres aprender más sobre nuestro análisis? ¡Haz clic en los botones de abajo! 👀',
    'SEND_RESPONSE_MESSAGE_EN': '{channel_name} is: {emoji} {answer} \n\nDo you want to learn more about our analysis? Click on the buttons below! 👀',
    'DETOXIFY_MESSAGE_ES': 'Vamos a ver... 👀',
    'DETOXIFY_MESSAGE_EN': 'Let\'s see... 👀',
    'FAILED_DETOXIFY_ES': 'Este mensaje no tiene contenido informativo y simplemente es un insulto, por lo tanto, no hay información relevante para desintoxicar.',
    'FAILED_DETOXIFY_EN': 'This message has no informative content and is simply an insult, therefore, there\'s no relevant information here to detoxify.',
    'TOXICITY_SCALE_ES': '''**Escala de toxicidad:**

0. **No tóxico:** Mensajes que promueven un ambiente positivo y respetuoso. Son inclusivos y constructivos, sin contenido ofensivo.
1. **Ligeramente tóxico:** Mensajes que son mayormente respetuosos, pero pueden incluir críticas pasivo-agresivas o un sesgo leve.
2. **Moderadamente tóxico:** Mensajes con un tono agresivo o que contienen lenguaje despectivo hacia grupos específicos.
3. **Altamente tóxico:** Mensajes que muestran un claro desprecio por individuos o grupos, utilizando insultos o lenguaje ofensivo.
4. **Extremadamente tóxico:** Mensajes que son agresivamente irrespetuosos, con amenazas o llamados a la acción violenta.''',
    'TOXICITY_SCALE_EN': '''**Toxicity Scale:**

0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.''',
    'GOODBYE_EN': 'You have become a star Detoxigramer! Give us your (anonymous) feedback here: {url}',
    'GOODBYE_ES': '¡Sos un Detoxigramer estrella! Nos podés dejar feedback anónimo acá: {url}'
}
BUTTONS = {
    'GREETING_EN': [['Detoxify Msg 📧', 'id:000'], ['Analyze Conv 💬', 'id:001']],
    'GREETING_ES': [['Detoxificar Msg 📧', 'id:000'], ['Analizar Conv 💬', 'id:001']],
    'POST_ANALISIS_EN': [['Explain', 'id:002'], ['Toxicity Dist', 'id:003']],
    'POST_ANALISIS_ES': [['Explicar', 'id:002'], ['Distrib Tox', 'id:003']],
    'FINAL_MSG_EN': [['End', 'id:004']],
    'FINAL_MSG_ES': [['Terminar', 'id:004']]
}
MESSAGES_WPP = {
    'GREETING_ES': 'Hola {name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus conversaciones, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
    'NO_GREETING_ES': 'Mmm... Qué tal si empezamos con un saludo? (Ej. "Hola!")',
    'GREETING_EN': 'Hello {name}, I am Detoxigram! 👋\nMy role is to help you identify toxicity in your conversations...',
    'NO_GREETING_EN': 'Mmm... Try sending a greeting to start the conversation (E.g. "Hello!")',
    'GOODBYE_EN': 'Goodbye! 👋 If you need anything else, just say hi!',
    'GOODBYE_ES': '¡Adiós! 👋 Si necesitas algo más, solo saluda.',
    'WAITING_FOR_MSG_EN': 'Great! Now, please send the message you would like to detoxify 🤓',
    'WAITING_FOR_MSG_ES': '¡Genial! Ahora, por favor envía el mensaje que te gustaría desintoxicar 🤓',
    'DETOXIFY_MESSAGE_ES': 'Vamos a ver... 👀',
    'DETOXIFY_MESSAGE_EN': 'Let\'s see... 👀',
    'POST_DETOX_EN':'Do you want to continue detoxifying?',
    'POST_DETOX_ES': '¿Quieres seguir detoxificando?',
    'START_AGAIN_EN':'Great! What would you like to do now?',
    'START_AGAIN_ES':'¡Genial! ¿Qué te gustaría hacer ahora?',
}
BUTTONS_WPP = {
    'GREETING_EN': [['Detoxify Msg 📧', 'id:000'], ['Analyze Conv 💬', 'id:001']],
    'GREETING_ES': [['Detoxificar Msg 📧', 'id:000'], ['Analizar Conv 💬', 'id:001']],
    'POST_ANALISIS_EN': [['Explain', 'id:002'], ['Toxicity Dist', 'id:003']],
    'POST_ANALISIS_ES': [['Explicar', 'id:002'], ['Distrib Tox', 'id:003']],
    'FINAL_MSG_EN': [['End', 'id:004']],
    'FINAL_MSG_ES': [['Terminar', 'id:004']],
    'SI_NO_ES': [['Si', 'id:005'], ['No', 'id:006']],
    'SI_NO_EN': [['Yes', 'id:005'], ['No', 'id:006']],
    
}



