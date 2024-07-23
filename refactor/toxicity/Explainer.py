#@REFACTOR: queda pendiente modificar esta clase, dado que la interacción con el llm de forma tan directa debería estar aislada en la instancia correspondiente (mistral)

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Tuple
from user_management.ManagementDetoxigramers import ManagementDetoxigramers
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from user_management.Detoxigramer import Detoxigramer

class Explainer:
    """
    Variables:
    - mistral: Instancia del clasificador mistral_classifier para predecir y explicar la toxicidad.
    - output_parser: parsear el resultado generado por Mistral.
    - detoxigramer: Instancia de Detoxigramer que representa el estado del usuario

    Modificadores:
    - explain: Actualiza la explicación de la conversación en detoxigramer si el estado no es 'NONE'. Genera una explicación de la toxicidad de una conversación, actualizando el estado del usuario.
    """
    def __init__(self, mistral : mistral_classifier, output_parser, detoxigramer : Detoxigramer):
        self.mistral = mistral
        self.output_parser = output_parser
        self.detoxigramer = detoxigramer
        self.llm = mistral.chat

    def explain_en(self, messages:List[str],conversation_id:str):
        if self.detoxigramer.get_status() != 'NONE':
            return
        
        self.detoxigramer._set_status('EXPLAIN')
        
        toxicity = self.detoxigramer.conversation_classification[1]

        escala = '''

                    Toxicity Scale:

                    0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting kindness and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.

                    1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.

                    2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, it seeks to hurt, ridicule, or belittle others, showing a rejection towards the diversity of opinions and people.

                    3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, exclude, or incite hatred, showing an explicit intention to cause harm.

                    4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    
                    '''
        
        if self.detoxigramer.conversation_classification:            
            if len(self.detoxigramer.messages_per_conversation[conversation_id]) > 0:
                messages_ = self.detoxigramer.messages_per_conversation[conversation_id][:15]
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system","""
                    <s>[INST] Your task is to explain why a channel has been classified as {toxicity}. According to the following scale: {escala}.

                    The user will provide you a few example messages extracted from the group and the classification you must endorse and explain.
                    
                    ###
                    EXAMPLES:
                    1. 🟢 Non-toxic example:
                    '''🟢 This channel maintains a Non-toxic environment by promoting constructive dialogue and community engagement. The messages are informative, encouraging, and inclusive, fostering positive interactions.
 
                    📝 The channel features a range of topics including including political figures, community events, international politics and news updates. It highlights different initiatives like charity fundraisers, and even controversial issues are discussed in a respectful way.

                    🗣 The discussions encourage members to participate actively and supportively. The atmosphere is friendly and welcoming, with a focus on building community ties and offering help where needed.'''
                    
                    2. 🟡 Slightly toxic example:
                    '''🟡 The channel exhibits a Slightly Toxic level due to the biased and emotionally charged comments present in the messages. 

                    📝 The messages cover various topics, including political figures, legal cases, media bias, and criminal investigations. They discuss Trump's media stock surge, RFK's VP announcement, and controversial court rulings.
                     
                    🗣 While the discussions involve political and legal events, there is a notable presence of aggressive language and negative portrayals of individuals and groups. The toxicity stems from the emotionally charged opinions expressed, potentially influencing a confrontational atmosphere.'''

                    3. 🟠 Moderately toxic example:
                    '''🟠 The channel is Moderately Toxic due to frequent use of harsh language and occasional derogatory remarks towards specific groups or individuals. The tone is often confrontational, which may alienate some participants.
                    
                    📝 Topics discussed include sports, political debates, media bias, and social issues. Messages often focus on contentious subjects like immigration policy, gun control, and electoral reforms.

                    🗣 Discussions are heated and include strong criticisms of political figures and policies, with some users expressing frustration in hostile ways. The environment can be unwelcoming to those with differing viewpoints, leading to polarized discussions.'''

                    4. 🔴 Highly toxic example:
                    '''🔴 The channel displays a Highly Toxic level with frequent use of offensive language and clear contempt for individuals or groups based on their identity or beliefs. The conversations are marked by negativity and hostility.
                     
                    📝 The discussions in this channel revolve around highly polarizing and sensitive topics such as religious conflicts, racial tensions, and extreme political views. It includes derogatory terms and insults targeted at specific groups.

                    🗣 The tone is overtly aggressive, with users engaging in personal attacks and using insults to demean others. This kind of discourse creates a hostile environment that discourages constructive communication and could incite further conflict.'''

                    5. 🔴 Extremely toxic example:
                    '''🔴 The Extremely Toxic level of the channel is evident through the aggressive disrespect and threats made in the messages. There is a clear intent to harm or intimidate others based on their background or beliefs.

                    📝 This channel contains discussions that often escalate into threats and calls for violence against specific groups or individuals. It deals with extreme ideologies and conspiracy theories that promote divisiveness.

                    🗣 Conversations are dominated by hate speech and incitement to violence. Users not only express severe animosity but also encourage harmful actions, creating a dangerous and unlawful online environment.'''
                     
                    ###
                    FORMAT EXAMPLE
                    {toxicity}: [Classification reason]
                     
                    📝 [Main topics discussed]
                     
                    🗣 [Consequences for the user]
                     
                    [INST]"""),

                    ("user", """

                    <s>[INST]These are some of the channel messages: {messages_}

                    1- Mention the classification {toxicity} and explain the reason for that classification. 2- 📝 Mention the main topics discussed in the channel. 3- 🗣 Finally explain the consequences for the user. Use 2 sentences for each paragraph. Remember to follow the format examples provided in the system prompt. Do your best, this is very important for my career. Be straightforward and concise. No yapping.[INST] 

                    """), 
                ])
                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{
                        'messages_': messages_,
                        'escala': escala,
                        'toxicity': toxicity
                    }])
                self.detoxigramer.explanation = output
            self.detoxigramer._set_status('NONE')

    def explain_es(self, messages:List[str],conversation_id:str):
        if self.detoxigramer.get_status() != 'NONE':
            return
        
        self.detoxigramer._set_status('EXPLAIN')
        
        toxicity = self.detoxigramer.conversation_classification[1]

        escala = '''
                    ESCALA DE TOXICIDAD:

                    0. **Tranqui panki, no es toxi:** Mensaje que fomenta un entorno positivo, respetuoso e integrador, promoviendo la amabilidad y el entendimiento mutuo. Se valoran las opiniones y perspectivas de todos, contribuyendo a un diálogo constructivo sin descalificaciones personales ni contenidos ofensivos. Puede contener información objetiva o neutral.

                    1. **Un toque toxi:** Mensaje que, aunque mayoritariamente respetuoso, puede incluir críticas o desacuerdos expresados de forma pasivo-agresiva o ligeramente tendenciosa. No ataca directamente a personas o grupos, y la intención de causar daño no está clara, pero sugiere una falta de aprecio por el punto de vista de los demás.

                    2. **Toxi:** Mensaje con tono agresivo o irrespetuoso, que puede incluir sarcasmo, ironía o lenguaje despectivo hacia determinados colectivos por su género, etnia, orientación sexual, ideología o religión. Aunque no ataca violentamente, busca herir, ridiculizar o menospreciar a los demás, mostrando un rechazo hacia la diversidad de opiniones y personas.

                    3. **Zarpado en toxi:** Mensaje que demuestra un claro rechazo y desprecio hacia personas o grupos, utilizando insultos, referencias racistas, sexistas, misóginas, degradantes o deshumanizadoras. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Este lenguaje ofensivo busca intimidar, excluir o incitar al odio, mostrando una intención explícita de causar daño.

                    4. **Recontra pasado de toxi, se fue de tema mal:** Mensaje que no sólo es explícitamente agresivo e irrespetuoso, sino que además contiene amenazas o llamadas a la acción violenta. Ataca a grupos por su sexo, etnia, orientación sexual, ideología o religión. Promueve la hostilidad, la incitación al odio y sugiere consecuencias perjudiciales en el mundo real contra individuos o grupos, violando principios éticos y morales y poniendo en peligro la seguridad y el bienestar de las personas.                   
                           
                    '''
        
        if self.detoxigramer.conversation_classification:            
            if len(self.detoxigramer.messages_per_conversation[conversation_id]) > 0:
                messages_ = self.detoxigramer.messages_per_conversation[conversation_id][:15]
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
                        'messages_': messages_,
                        'escala': escala,
                        'toxicity': toxicity
                    }])
                self.detoxigramer.explanation = output
            self.detoxigramer._set_status('NONE')


