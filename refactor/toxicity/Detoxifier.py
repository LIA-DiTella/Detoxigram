from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Tuple, Literal
from user_management.ManagementDetoxigramers import ManagementDetoxigramers
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from user_management.Detoxigramer import Detoxigramer
from Analyzer import Analyzer

class Detoxifier:

    def __init__(self, mistral : mistral_classifier, output_parser, detoxigramer : Detoxigramer, analyzer : Analyzer):
            self.mistral = mistral
            self.output_parser = output_parser
            self.detoxigramer = detoxigramer
            self.analyzer = analyzer
            self.llm = mistral.chat

    def detoxify_single_message_en(self, message:str):
        if self.detoxigramer.get_status() != 'NONE':
            return
        self.detoxigramer._set_status('DETOX')
        toxicity : str = self._set_toxicity(self.analyzer.message_classifier(message), 'EN')
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """<s>[INST] You are a moderator of online content, your task is to detoxify and provide non-toxic alternatives for messages, if they are found to be toxic. If the message only contains instuls, say: "This message has no informative content and is simply an insult, therefore, there's no relevant information here to detoxify." Keep your rephrasing as close to the original message as possible.


            ## Toxicity Scale:
            0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
            1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
            2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
            3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
            4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

            # Task:
            Revise the provided message. If the message is 🟡 Slightly Toxic, 🟡 Moderately Toxic,  🔴 Highly Toxic, or 🔴 Extremely Toxic, suggest a rephrased non-toxic version that conveys the meaning of the original message in a more respectful and positive manner. Maintain the intention of the original message DO NOT add phrases like "I'd be interested in discussing this further" if the user didn't say that. Always answer in 2 paragraphs.

            ####
            Examples of detoxification:

            1. **Non-toxic:**
                - User Message: '''Now, please detoxify the following message which has a toxicity level of 🟢 Non-toxic: [[["I appreciate your perspective and would like to discuss this further."]]]'''
                - Output: '''This message is 🟢 Non-toxic. It promotes a respectful and open dialogue.'''
                
            2. **Slightly Toxic:**
                - User Message: '''Now, please detoxify the following message which has a toxicity level of 🟡 Slightly Toxic: [[["That's a naive way of looking at things, don't you think?"]]]'''
                - Output: '''This message is 🟡 Slightly Toxic due to its patronizing tone. 
        
                A more respectful phrasing could be: "Could there be a more comprehensive way of looking at it?"'''

            3. **Moderately Toxic:**
                - User Message: '''Now, please detoxify the following message which has a toxicity level of 🟡 Moderately toxic: [[["People who believe that are living in a fantasy world."]]]'''
                - Output: '''This message is 🟠 Moderately Toxic because it dismisses others' beliefs.
                
                A less toxic version could be: "I find it hard to agree with that perspective, I think it's unrealistic."'''

            4. **Highly Toxic:**
                - User Message: '''Now, please detoxify the following message which has a toxicity level of 🔴 Highly Toxic: [[["This is the dumbest idea I've ever heard."]]]'''
                - Output: '''The message is 🔴 Highly Toxic due to its derogatory language.
        
                A constructive alternative might be: "I don't think that idea is the best approach at all."'''

            5. **Extremely Toxic:**
                - User Message: '''Now, please detoxify the following message which has a toxicity level of 🔴 Extremely Toxic: [[["Anyone who supports this policy must be a complete idiot. We should kill them all, they don't deserve to exist."]]]'''
                - Output: '''This message is 🔴 Extremely Toxic and offensive.
        
                A non-toxic rephrasing could be: "I'm surprised that there's support for this policy. I have a completely different point of view"'''[INST]
                
    """),
    ("user", "<s>[INST] Now, please detoxify the following message which has a toxicity level of {toxicity}: [[[ " + message.text + "]]][INST]")
])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{'toxicity': toxicity}])
        return output 

    def detoxify_single_message_es(self, message:str):
        if self.detoxigramer.get_status() != 'NONE':
            return
        self.detoxigramer._set_status('DETOX')
        toxicity : str = self._set_toxicity(self.analyzer.message_classifier(message), 'ES')
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
    ("user", "<s>[INST] Ahora, por favor, detoxificá el siguiente mensaje que tiene un nivel de toxicidad de {toxicity}: [[[ " + message.text + "]]][INST]")])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{'toxicity': toxicity}])
        return output 

    def _set_toxicity(self, classification : int, language:Literal['EN','ES']):
        if language == 'EN':
            if classification < 1:
                    return "🟢 Non-toxic"
            elif 1 <= classification < 1.75:
                    return "🟡 Slightly toxic"
            elif 1.75 <= classification < 2.5:
                    return "🟠 Moderately toxic"
            elif 2.5 <= classification < 3.5:
                    return "🔴 Highly toxic"
            else:
                    return "🔴 Extremely toxic"
        elif language == 'ES':
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
   