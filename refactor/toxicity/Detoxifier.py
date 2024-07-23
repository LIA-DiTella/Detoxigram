from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Tuple
from user_management.ManagementDetoxigramers import ManagementDetoxigramers
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from user_management.Detoxigramer import Detoxigramer
from Analyzer import Analyzer
class Detoxifier:
    """
    Variables:
    - mistral: Instancia del clasificador mistral_classifier para predecir y explicar la toxicidad.
    - output_parser: Función o clase utilizada para parsear el resultado generado por el modelo Mistral.
    - detoxigramer: Instancia de Detoxigramer que representa el estado del usuario.
    - analyzer: Instancia de Analyzer para clasificar mensajes individuales.
    - llm: Interfaz del modelo de lenguaje utilizado para generar las explicaciones y detoxificaciones.

    Funciones:
    - detoxify_single_message: Detoxifica un mensaje individual si el estado del usuario es 'NONE'.
    - detoxify_single_message: Actualiza el estado del usuario y genera una versión detoxificada del mensaje.
    - set_toxicity: Ajusta la representación de la toxicidad en función de un valor numérico.
    """

    def __init__(self, mistral : mistral_classifier, output_parser, detoxigramer : Detoxigramer, analyzer : Analyzer):
            self.mistral = mistral
            self.output_parser = output_parser
            self.detoxigramer = detoxigramer
            self.analyzer = analyzer
            self.llm = mistral.chat

    def detoxify_single_message(self, message:str):
        if self.detoxigramer.get_status() != 'NONE':
            return
        self.detoxigramer._set_status('DETOX')
        toxicity : Tuple[str,int] = self.analyzer.message_classifier(message)
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

    def set_toxicity(self, toxicity : int):
        if 0 <= toxicity < 1:
            toxicity = "🟢 Non-toxic"
        elif toxicity >= 1 and toxicity < 1.75:
            toxicity = "🟡 Slightly toxic"
        elif 1.75 <= toxicity < 2.5:
            toxicity = "🟠 Moderately toxic"
        elif 2.5 <= toxicity < 3.5:
            toxicity = "🔴 Highly toxic"
        else:
            toxicity = "🔴 Extremely toxic"

      
 