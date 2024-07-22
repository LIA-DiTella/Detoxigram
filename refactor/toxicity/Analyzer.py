from typing import List, Any, Tuple
from user_management.ManagementDetoxigramers import ManagementDetoxigramers
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier
from user_management.Detoxigramer import Detoxigramer

class Analyzer:
    """
    Variables:
    - hatebert: Instancia del clasificador hate_bert_classifier para filtrar mensajes.
    - mistral: Instancia del clasificador mistral_classifier para predecir la toxicidad.
    - management_detoxigramers: Instancia de ManagementDetoxigramers para gestionar los estados de los usuarios.
    - detoxigramer: Instancia de Detoxigramer que representa el estado del usuario.

    Funciones:
    - message_classifier: Clasifica la toxicidad de un mensaje individual.
    - conversation_classifier: Clasifica la toxicidad promedio de una conversación completa.
    - conversation_classifier: Actualiza el estado del canal y los mensajes en detoxigramer si el estado es 'NONE'.
    """
    def __init__(self, hatebert : hate_bert_classifier, mistral : mistral_classifier, management_detoxigramers : ManagementDetoxigramers, detoxigramer : Detoxigramer):
    
        self.hatebert = hatebert
        self.mistral = mistral
        self.management_detoxigramers = management_detoxigramers

    def message_classifier(self, message:str) -> int:
        toxicity : Tuple[bool, int] = self.mistral.predictToxicity(message)
        return toxicity[1]
    
    def conversation_classifier(self, conversation_id:str, messages:List[str]):
        if self.detoxigramer.status == 'NONE':
            self.detoxigramer.status = 'ANALYZE'
            most_toxic_messages:List[str] = self.hatebert.get_most_toxic_messages_none_batch(messages)
            toxicity : Tuple[bool, int] = self.mistral.predict_average_toxicity_score(most_toxic_messages)
            self.detoxigramer._update_channel(conversation_id, toxicity[1], most_toxic_messages)
            return self.detoxigramer.conversation_classification[1]
        else:
            return None

    
    