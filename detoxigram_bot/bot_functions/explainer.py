import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from telebot import types
import json
import os

class explainer:
    def __init__(self, bot, loop, formatter, mistral, bert, output_parser, user_management):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.mistral = mistral
        self.bert = bert
        self.output_parser = output_parser
        self.llm = mistral.chat
        self.user_management = user_management
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))

    
    def write_cache(self, user_id, channel_name, average_toxicity_score):
        cache_file_path = os.path.join(self.cache_dir, "explainer_cache.json")
        try:
            with open(cache_file_path, 'r') as cache_file:
                cache = json.load(cache_file)
        except FileNotFoundError:
            cache = {}

        cache[channel_name] = {
            'user_id': user_id,
            'channel_name': channel_name,
            'average_toxicity_score': average_toxicity_score
        }

        with open(cache_file_path, 'w') as cache_file:
            json.dump(cache, cache_file)
            print("Cache updated successfully!")

    def explain(self, message):
        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        if not state.is_explaining:
            return
        toxicity = state.last_analyzed_toxicity

        markup = types.InlineKeyboardMarkup(row_width=1)
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        new_analyze = types.InlineKeyboardButton('Analyze another channel 🔍', callback_data='analyze')
        toxicity_ = types.InlineKeyboardButton('Toxicity dimensions 📊', callback_data='learn_more')
        if 0 <= toxicity < 1:
            toxicity = "🟢 Tranqui panki, no es toxi"
        elif toxicity >= 1 and toxicity < 1.75:
            toxicity = "🟡 Un toque toxi"
        elif 1.75 <= toxicity < 2.5:
            toxicity = "🟠 Toxi"
        elif 2.5 <= toxicity < 3.5:
            toxicity = "🔴 Zarpado en toxi"
        else:
            toxicity = "🔴 Recontra pasado de toxi, se fue de tema mal"

        escala = '''
                    ESCALA DE TOXICIDAD:

                    0. **Tranqui panki, no es toxi:** Mensaje que fomenta un entorno positivo, respetuoso e integrador, promoviendo la amabilidad y el entendimiento mutuo. Se valoran las opiniones y perspectivas de todos, contribuyendo a un diálogo constructivo sin descalificaciones personales ni contenidos ofensivos. Puede contener información objetiva o neutral.

                    1. **Un toque toxi:** Mensaje que, aunque mayoritariamente respetuoso, puede incluir críticas o desacuerdos expresados de forma pasivo-agresiva o ligeramente tendenciosa. No ataca directamente a personas o grupos, y la intención de causar daño no está clara, pero sugiere una falta de aprecio por el punto de vista de los demás.

                    2. **Toxi:** Mensaje con tono agresivo o irrespetuoso, que puede incluir sarcasmo, ironía o lenguaje despectivo hacia determinados colectivos por su género, etnia, orientación sexual, ideología o religión. Aunque no ataca violentamente, busca herir, ridiculizar o menospreciar a los demás, mostrando un rechazo hacia la diversidad de opiniones y personas.

                    3. **Zarpado en toxi:** Mensaje que demuestra un claro rechazo y desprecio hacia personas o grupos, utilizando insultos, referencias racistas, sexistas, misóginas, degradantes o deshumanizadoras. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Este lenguaje ofensivo busca intimidar, excluir o incitar al odio, mostrando una intención explícita de causar daño.

                    4. **Recontra pasado de toxi, se fue de tema mal:** Mensaje que no sólo es explícitamente agresivo e irrespetuoso, sino que además contiene amenazas o llamadas a la acción violenta. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Promueve la hostilidad, la incitación al odio y sugiere consecuencias perjudiciales en el mundo real contra individuos o grupos, violando principios éticos y morales y poniendo en peligro la seguridad y el bienestar de las personas.                   
                    '''
        if state.last_channel_analyzed:            
            if len(state.last_chunk_of_messages) > 0:
                self.bot.reply_to(message, "Give me a moment... 🤔")
                filtered_messages = state.last_chunk_of_messages[:15]

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system","""
                    <s>[INST] Su tarea consiste en explicar por qué un canal ha sido clasificado como {toxicity}, de acuerdo con la siguiente escala: {escala}.

                    El usuario te proporcionará unos mensajes de ejemplo extraídos del grupo y la clasificación que debes apoyar y explicar.
                    
                    ###
                    EJEMPLOS:
                    1. 🟢 Ejemplo Tranqui panki, no es toxi:
                    '''🟢 Este canal mantiene un ambiente Tranqui panki, no es toxi promoviendo el diálogo constructivo y la participación de la comunidad. Los mensajes son informativos, alentadores e inclusivos, fomentando interacciones positivas.
 
                    📝 El canal presenta una serie de temas que incluyen figuras políticas, eventos comunitarios, política internacional y noticias de última hora. Destaca distintas iniciativas, como recaudaciones de fondos para obras benéficas, e incluso se debaten temas controvertidos de forma respetuosa.

                    🗣 Los debates animan a los miembros a participar de forma activa y solidaria. El ambiente es cordial y acogedor, y se centra en crear lazos comunitarios y ofrecer ayuda cuando es necesario.''
                    
                    2. 🟡 Ejemplo Un toque toxi:
                    '''🟡 El canal exhibe un nivel Un toque toxi debido a los comentarios tendenciosos y con carga emocional presentes en los mensajes. 

                    📝 Los mensajes cubren varios temas, incluyendo figuras políticas, casos legales, sesgo de los medios e investigaciones criminales. Hablan de la subida de Trump en los medios, del anuncio de RFK como vicepresidente y de sentencias judiciales controvertidas.
                    
                    🗣 Aunque las discusiones versan sobre acontecimientos políticos y jurídicos, hay una notable presencia de lenguaje agresivo y descripciones negativas de personas y grupos. La toxicidad se deriva de las opiniones cargadas de emotividad que se expresan, lo que puede influir en un ambiente de confrontación.'''

                    3. 🟠 Ejemplo toxi:
                    '''🟠 El canal es toxi debido al uso frecuente de un lenguaje soez y a ocasionales comentarios despectivos hacia grupos o individuos específicos. El tono suele ser de confrontación, lo que puede alejar a algunos participantes.
                    
                    📝 Entre los temas tratados figuran los deportes, los debates políticos, la parcialidad de los medios de comunicación y las cuestiones sociales. Los mensajes suelen centrarse en temas polémicos como la política de inmigración, el control de armas y las reformas electorales.

                    🗣 Los debates son acalorados e incluyen fuertes críticas a figuras políticas y políticas, y algunos usuarios expresan su frustración de forma hostil. El ambiente puede ser poco acogedor para quienes tienen puntos de vista diferentes, lo que lleva a discusiones polarizadas.'''
                    4. 🔴 Ejemplo Zarpado en toxi:
                    '''🔴 El canal muestra un nivel Zarpado en toxi con uso frecuente de lenguaje ofensivo y claro desprecio hacia personas o grupos en función de su identidad o creencias. Las conversaciones están marcadas por la negatividad y la hostilidad.
                     
                    📝 Las discusiones en este canal giran en torno a temas muy polarizantes y delicados, como conflictos religiosos, tensiones raciales y opiniones políticas extremas. Incluye términos despectivos e insultos dirigidos a grupos específicos.

                    🗣 El tono es abiertamente agresivo, con usuarios que realizan ataques personales y utilizan insultos para degradar a los demás. Este tipo de discurso crea un ambiente hostil que desalienta la comunicación constructiva y podría incitar a más conflictos.'''
                    5. 🔴 Ejemplo Recontra pasado de toxi, se fue de tema mal:
                    '''🔴 El nivel Recontra pasado de toxi, se fue de tema mal es evidente a través de las agresivas faltas de respeto y amenazas vertidas en los mensajes. Hay una clara intención de dañar o intimidar a otros por su procedencia o creencias.
                     
                    📝 Este canal contiene debates que a menudo desembocan en amenazas y llamamientos a la violencia contra grupos o individuos concretos. Trata de ideologías extremas y teorías de la conspiración que fomentan la división.

                    🗣 Las conversaciones están dominadas por el discurso del odio y la incitación a la violencia. Los usuarios no solo expresan una grave animadversión, sino que también fomentan acciones dañinas, creando un entorno potencialmente peligroso e ilegal.'''                     
                    ###
                    EJEMPLO DE FORMATO
                    {toxicity}: [razón para la clasificación]
                     
                    📝 [Principales temas tratados]
                     
                    🗣 [Consecuencias para el usuario]
                     
                    [INST]"""),

                    ("user", """

                    <s>[INST]Estos son algunos de los mensajes del canal: {filtered_messages}

                    1- Menciona la clasificación {toxicity} y explica el porqué de dicha clasificación. 2- 📝 Menciona los principales temas tratados en el canal. 3- 🗣 Por último, explica las consecuencias para el usuario. Utiliza solo 2 frases para cada párrafo. Recuerda seguir los ejemplos de formato proporcionados en el prompt del sistema. Esfuérzate, esto es muy importante para mi carrera. No yapping, nada de cháchara. Sé directo y conciso. 

                    """), 
                ])
                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{
                        'filtered_messages': filtered_messages,
                        'escala': escala,
                        'toxicity': toxicity
                    }])
                print(output)
                markup.add(new_analyze, toxicity_, go_back)
                self.bot.reply_to(message, f'{output[0]}', reply_markup=markup, parse_mode='Markdown')
                self.write_cache(user_id, state.last_channel_analyzed, toxicity)
            else:
                self.bot.reply_to(message, "Failed! Try with another channel!")
        else:
            self.bot.reply_to(message, "Please provide a channel name!")
        state.is_explaining = False

    def detoxify_single_message(self, message):
        user_id = message.chat.id
        state = self.user_management.get_user_state(user_id)
        if not state.is_detoxifying:
            return
        self.bot.reply_to(message, "Let's see... 👀")
        markup = types.InlineKeyboardMarkup(row_width=1)
        detoxify_new = types.InlineKeyboardButton('Detoxify another message 📩', callback_data='detoxify')
        go_back = types.InlineKeyboardButton('Restart 🔄', callback_data='restart')
        toxicity_result = self.mistral.predictToxicity(message.text)
        _, toxicity = toxicity_result
        if 0 <= toxicity < 1:
            toxicity = "🟢 Tranqui panki, no es toxi"
        elif toxicity >= 1 and toxicity < 1.75:
            toxicity = "🟡 Un toque toxi"
        elif 1.75 <= toxicity < 2.5:
            toxicity = "🟠 Toxi"
        elif 2.5 <= toxicity < 3.5:
            toxicity = "🔴 Zarpado en toxi"
        else:
            toxicity = "🔴 Recontra pasado de toxi, se fue de tema mal"
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """<s>[INST] Sos un moderador de contenidos online en español rioplatense. Tu tarea es detoxificar y proporcionar alternativas no-tóxicas para los mensajes, si es que el mensaje tiene contenido informativo y fue clasificados como Si el mensaje es "🟡 Un toque toxi", "🟠 Toxi", "🔴 Zarpado en toxi" o "🔴 Recontra pasado de toxi, se fue de tema mal". Si el mensaje sólo contiene insultos, di: "Este mensaje no tiene contenido informativo y es simplemente un insulto, por lo tanto, no hay información relevante aquí para detoxificar". Mantené tu reformulación lo más cerca posible del mensaje original, sin cambiar su intención.


           ## ESCALA DE TOXICIDAD:

                    0. **Tranqui panki, no es toxi:** Mensaje que fomenta un entorno positivo, respetuoso e integrador, promoviendo la amabilidad y el entendimiento mutuo. Se valoran las opiniones y perspectivas de todos, contribuyendo a un diálogo constructivo sin descalificaciones personales ni contenidos ofensivos. Puede contener información objetiva o neutral.

                    1. **Un toque toxi:** Mensaje que, aunque mayoritariamente respetuoso, puede incluir críticas o desacuerdos expresados de forma pasivo-agresiva o ligeramente tendenciosa. No ataca directamente a personas o grupos, y la intención de causar daño no está clara, pero sugiere una falta de aprecio por el punto de vista de los demás.

                    2. **Toxi:** Mensaje con tono agresivo o irrespetuoso, que puede incluir sarcasmo, ironía o lenguaje despectivo hacia determinados colectivos por su género, etnia, orientación sexual, ideología o religión. Aunque no ataca violentamente, busca herir, ridiculizar o menospreciar a los demás, mostrando un rechazo hacia la diversidad de opiniones y personas.

                    3. **Zarpado en toxi:** Mensaje que demuestra un claro rechazo y desprecio hacia personas o grupos, utilizando insultos, referencias racistas, sexistas, misóginas, degradantes o deshumanizadoras. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Este lenguaje ofensivo busca intimidar, excluir o incitar al odio, mostrando una intención explícita de causar daño.

                    4. **Recontra pasado de toxi, se fue de tema mal:** Mensaje que no sólo es explícitamente agresivo e irrespetuoso, sino que además contiene amenazas o llamadas a la acción violenta. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Promueve la hostilidad, la incitación al odio y sugiere consecuencias perjudiciales en el mundo real contra individuos o grupos, violando principios éticos y morales y poniendo en peligro la seguridad y el bienestar de las personas.                   
                    '''

             # Tarea:
            Revisa el mensaje recibido. Si el mensaje es "🟡 Un toque toxi", "🟠 Toxi", "🔴 Zarpado en toxi" o "🔴 Recontra pasado de toxi, se fue de tema mal", sugerí una reformulación no-tóxica que transmita el significado del mensaje original de una manera más respetuosa y positiva. Manténé la intención del mensaje original NO añadas frases como "Me interesaría discutir esto más a fondo" si el usuario no dijo eso. Respondé siempre en 2 párrafos.

            ####
            Ejemplos de detoxificación:

            1. **Tranqui panki, no es toxi:**
                - Input: '''Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de 🟢 Tranqui panki, no es toxi: [[["Aprecio su perspectiva y me gustaría discutir esto más a fondo"]]]]'''
                - Output: '''Este mensaje es 🟢 Tranqui panki, no es toxi. Promueve un diálogo respetuoso y abierto.'''
                
            2. **Un toque toxi:**
                - Input: '''Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de 🟡 Un toque toxi: [[["Esa es una forma ingenua de ver las cosas, ¿no crees?"]]]'''
                - Output: '''Este mensaje es 🟡 Un toque toxi, debido a su tono condescendiente. 
        
                Una opción más constructiva podría ser: "¿Podría haber una forma más completa de verlo?"'''

            3. **Moderadamente Tóxico:**
                - Input: '''Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de 🟠 Toxi: [[["La gente que cree eso vive en un mundo de fantasía"]]]]'''
                - Output: '''Este mensaje es 🟠 Toxi, porque desestima las creencias de los demás.
                
                Una opción más constructiva podría ser: "Me cuesta estar de acuerdo con esa perspectiva, creo que no es realista."'''

             4. **Altamente Tóxico:**
                - Input: '''Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de 🔴 Zarpado en toxi debido a su lenguaje despectivo: [[["Esta es la idea más pelotuda que escuché"]]]]'''
                - Output: '''El mensaje es 🔴 Zarpado en toxi, debido a su lenguaje despectivo.
        
                Una opción más constructiva podría ser: "No creo que esa idea sea el mejor enfoque en absoluto"'''

            5. **Extremadamente Tóxico:**
                - Input: '''Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de 🔴 Recontra pasado de toxi, se fue de tema mal: [[["Cualquiera que apoye esta política debe ser un tarado de la concha de su madre. Deberíamos cagarlos a tiros para que se dejen de joder"]]]'''
                - Output: '''Este mensaje es 🔴 Recontra pasado de toxi, se fue de tema mal.
        
                Una opción más constructiva podría ser: "Me sorprende que haya apoyo a esta política. Yo tengo un punto de vista completamente diferente y me enoja mucho que esta diferencia nos impida avanzar"'''[INST]
                
    """),
    ("user", "<s>[INST] Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de {toxicity}: [[[ " + message.text + "]]][INST]")
])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{'toxicity': toxicity}])
        print(output)
        markup.add(detoxify_new, go_back)
        self.bot.reply_to(message, f'{output[0]}', reply_markup=markup, parse_mode='Markdown')
        state.is_detoxifying = False
        
    
