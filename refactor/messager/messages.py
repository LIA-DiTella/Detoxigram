MESSAGES = {
    'GREETING_SP': 'Hola {name}, soy Detoxigram! 👋\nMi rol es ayudarte a identificar la toxicidad en tus conversaciones, para que puedas tomar decisiones informadas sobre el contenido que consumís y compartís 🤖\n¿Qué te gustaría hacer?',
    'NO_GREETING_SP': 'Mmm... Que tal si empezamos con un saludo? (Ej. "Hola!")',
    'GREETING_EN': 'Hello {name}, I am Detoxigram! 👋\nMy role is to help you identify toxicity in your conversations...',
    'NO_GREETING_EN': 'Mmm... Try sending a greeting to start the conversation (E.g. "Hello!")',
    'BOT_LOADED': 'Bot token loaded, Detoxigram is live 🚀',
    'ENTERING_TESTING_MODE': 'Entering testing mode. I will now output some internal information, and I will now work on downloaded channels.',
    'LEAVING_TESTING_MODE': 'Leaving testing mode.',
    'GOODBYE': 'Goodbye! 👋 If you need anything else, just say hi!',
    'NOT_SURE': 'Mmm... I\'m not sure what that means, {username}. Would any of these options be helpful? 😁',
    'ANALYZE_START': 'Great!\n\nJust so you know, when we evaluate the toxicity, we\'ll only consider the last 50 messages of the channel ⚠️\n\nNow, please provide the @ChannelName you would like to analyze 🤓',
    'STILL_WORKING': 'I\'m sorry, I\'m still working on your last request! 🕣',
    'EXPLAIN_START': 'After evaluating the content of {last_channel_analyzed}, we saw that this channel is {last_toxicity}. Now I will explain to you why, it will take a few seconds 🕣',
    'NO_CHANNEL_TO_EXPLAIN': 'I\'m sorry, I don\'t have any channel to explain. Please analyze a channel first!',
    'HELP_TEXT': '''Welcome to Detoxigram! 🌟 Here's how you can use our bot to make your Telegram experience safer:

1. **Analyze a Channel:** To start analyzing a channel for toxic content, simply tap the 'Analyze a Channel 🔍' button. Then, enter the @ChannelName or send the invite to the channel. I'll check the last 50 messages and let you know how toxic the conversations are.

2. **Explain Toxicity:** Curious about why a channel was rated a certain way? Tap the 'Explain why 📝' button after analyzing a channel. I'll provide you with a summary of the channel's messages, highlighting specific examples of toxicity. This helps you understand the context and specifics of the content I've analyzed.

3. **Detoxify a Message:** Want to clean up a specific message? Use the 'Detoxify a message📩' option to send me a message you think is problematic. I'll offer a less toxic version, providing a cleaner, more respectful alternative.

Need more help or have any questions? Don't hesitate to reach out. You can contact us directly at malbaposse@mail.utdt.edu. We're here to help make your digital spaces safer! 🛡️''',
    'TOXICITY_ANALYSIS_START': 'We\'ve just classified the channel you sent. I will send you a more detailed analysis of the channel shortly 📊',
    'CACHE_UPDATED': 'Cache updated successfully!',
    'DETOXIFY_START': 'Great! ⚠️ Now, please write a message you would like to detoxify 🤓',
    'HERE_ARE_MORE_OPTIONS': 'Here are some more options! 🤓',
    'WHAT_NOW': 'Alright! What would you like to do now? 🤔',
    'NOT_VALID_CHANNEL_SP': 'Ups! Ese no es un nombre de canal válido. ¡Inténtalo de nuevo! 🫣',
    'NOT_VALID_CHANNEL_EN': 'Oops! That is not a valid channel name. Try again! 🫣',
    'ANALYZE_RESPONSE_SP': '¡Entendido! Analizaré {channel_name}... Por favor espera un momento 🙏',
    'ANALYZE_RESPONSE_EN': 'Got it! I will analyze {channel_name}... Please wait a moment 🙏',
    'NO_MESSAGES_FOUND_SP': '¡No se encontraron mensajes en el canal especificado! ¿Por qué no empezamos de nuevo?',
    'NO_MESSAGES_FOUND_EN': 'No messages found in the specified channel! Why don\'t we start again?',
    'TOOK_LONGER_SP': 'Eso tomó más tiempo del esperado... Dame un segundo para verificar la toxicidad en el canal 🕣',
    'TOOK_LONGER_EN': 'That took longer than expected... Now give me a second to check for toxicity in the channel 🕣',
    'FAILED_TOXICITY_ANALYSIS_SP': '¡Fallido! ¡Inténtalo con otro canal!',
    'FAILED_TOXICITY_ANALYSIS_EN': 'Failed! Try with another channel!',
    'SEND_RESPONSE_MESSAGE_SP': '{channel_name} es: {emoji} {answer} \n\n¿Quieres aprender más sobre nuestro análisis? ¡Haz clic en los botones de abajo! 👀',
    'SEND_RESPONSE_MESSAGE_EN': '{channel_name} is: {emoji} {answer} \n\nDo you want to learn more about our analysis? Click on the buttons below! 👀',
    'DETOXIFY_MESSAGE_SP': 'Vamos a ver... 👀',
    'DETOXIFY_MESSAGE_EN': 'Let\'s see... 👀',
    'FAILED_DETOXIFY_SP': 'Este mensaje no tiene contenido informativo y simplemente es un insulto, por lo tanto, no hay información relevante para desintoxicar.',
    'FAILED_DETOXIFY_EN': 'This message has no informative content and is simply an insult, therefore, there\'s no relevant information here to detoxify.',
    'TOXICITY_SCALE_SP': '''**Escala de toxicidad:**

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
    'GOODBYE_EN':'You have become a star Detoxigramer! Give us your (anonymus) feedback here: {url}',
    'GOODBYE_SP': 'Sos un Detoxigramer estrella! Nos podes dejar feedback anónimo acá: {url}'

}

BOTONES ={

}