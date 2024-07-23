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

    def explain(self, messages:List[str],conversation_id:str):
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
