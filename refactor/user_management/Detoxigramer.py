import os
import sys
import hashlib
sys.path.append('..')
from typing import Literal, Tuple, Dict, List, Optional

class WhatsApp_Detoxigramer:
    def __init__(self):
        '''
        Variables:
        - id: identificador de cada usuario.
        - status: dado un momento del programa, determina qué feature está usando el usuario.
        - conversation_classification: en caso de estar utilizando Telegram, contiene el nombre del Channel analizado y su clasificación.
        - messages_per_conversation: contiene los mensajes de los canales analizados
        - testing: activar o desactivar el modo de testing. No lo debería poder activar cualquier usuario.
        - platform: determina en qué plataforma está operando el usuario.

        Invariante de Representación:
        - self.id: str y self.id ≠ "" (cadena no vacía). Todo ID es único.
        - self.status: Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION'] y debe ser consistente con la función actualmente en ejecución.
        - self.conversation_classification: Optional[Tuple[str, int]] y, si no es None, cumple que self.conversation_classification[0] ≠ "" y self.conversation_classification[1] ≥ 0.
        - self.messages_per_conversation: Optional[Dict[str, List[str]]] y, si no es None, todas las claves son cadenas no vacías y todos los valores son listas de cadenas.
        - self.testing: bool.
        - self.platform: Tuple[bool, bool], donde el primer elemento indica disponibilidad de Telegram y el segundo disponibilidad de WhatsApp.

        Observadores:
        - get_id: devuelve el id del usuario
        - get_status: devuelve el status del usuario
        - get_conversation_classification: devuelve el nombre del último Channel analizado y su clasificación
        - get_messages_channel: devuelve los mensajes de un Channel en particular.

        Modificadores:
        - _update_channel: modifica la clasificación del canal y los mensajes.
        - _set_platform: establece la plataforma del usuario.
        - _set_status: establece el estado del usuario.
        
        '''
        self.id: str
        self.status : Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION', 'NONE']
        self.conversation_classification : Optional[Tuple[str, str]]
        self.messages_per_conversation : Optional[Dict[str, List[str]]]
        self.explanation : Optional[str]
        self.testing : bool
    
    def set_id(self, number:int):
        self.id : str = hashlib.shake_256(str(number).encode())
        self.id : str = id.hexdigest(15)
         
    def _last_toxicity(self, classification : int, language:Literal['EN','ES']):
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
        
    def _update_conversation(self, conversation_id:str, classification:int, messages:List[str], language:Literal['EN','ES']):
        self.conversation_classification = [conversation_id, self._last_toxicity(classification, language)]
        self.messages_per_conversation = {conversation_id : messages }
    
    def _set_platform(self, platform : str):
        if platform == 'TELEGRAM' and self.platform[0] != 1:
            self.platform[0] = 1
        elif platform == 'WHATSAPP' and self.platform[1] != 1:
            self.platform[1] = 1
        else:
            self.platform = (0,0)

    def _set_status(self, status : Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION']):
          self.status = status

    def get_id(self) -> int:
          return self.id
    
    def get_status(self) -> Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION']:
          return self.status
    
    def get_conversation_classification(self) -> Tuple[str,str]:
          return self.conversation_classification
    
    def get_messages_conversation(self, conversation_id : str) -> List[str]:
        if conversation_id in self.messages_per_conversation:
              return self.messages_per_conversation[conversation_id] 

class Telegram_Detoxigramer:
    def __init__(self):
        '''
        Variables:
        - id: identificador de cada usuario.
        - status: dado un momento del programa, determina qué feature está usando el usuario.
        - conversation_classification: en caso de estar utilizando Telegram, contiene el nombre del Channel analizado y su clasificación.
        - messages_per_conversation: contiene los mensajes de los canales analizados
        - testing: activar o desactivar el modo de testing. No lo debería poder activar cualquier usuario.
        - platform: determina en qué plataforma está operando el usuario.

        Invariante de Representación:
        - self.id: str y self.id ≠ "" (cadena no vacía). Todo ID es único.
        - self.status: Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION'] y debe ser consistente con la función actualmente en ejecución.
        - self.conversation_classification: Optional[Tuple[str, int]] y, si no es None, cumple que self.conversation_classification[0] ≠ "" y self.conversation_classification[1] ≥ 0.
        - self.messages_per_conversation: Optional[Dict[str, List[str]]] y, si no es None, todas las claves son cadenas no vacías y todos los valores son listas de cadenas.
        - self.testing: bool.
        - self.platform: Tuple[bool, bool], donde el primer elemento indica disponibilidad de Telegram y el segundo disponibilidad de WhatsApp.

        Observadores:
        - get_id: devuelve el id del usuario
        - get_status: devuelve el status del usuario
        - get_conversation_classification: devuelve el nombre del último Channel analizado y su clasificación
        - get_messages_channel: devuelve los mensajes de un Channel en particular.

        Modificadores:
        - _update_channel: modifica la clasificación del canal y los mensajes.
        - _set_platform: establece la plataforma del usuario.
        - _set_status: establece el estado del usuario.
        
        '''

        self.id: str
        self.status : Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION', 'NONE']
        self.conversation_classification : Optional[Tuple[str, str]]
        self.messages_per_conversation : Optional[Dict[str, List[str]]]
        self.explanation : Optional[str]
        self.testing : bool
        
    def set_id(self, number:int):
        self.id : str = hashlib.shake_256(str(number).encode())
        self.id : str = id.hexdigest(15)
         
    def _last_toxicity(self, classification : int, language:Literal['EN','ES']):
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
        
    def _update_conversation(self, conversation_id:str, classification:int, messages:List[str], language:Literal['EN','ES']):
        self.conversation_classification = [conversation_id, self._last_toxicity(classification, language)]
        self.messages_per_conversation = {conversation_id : messages }
    
    def _set_platform(self, platform : str):
        if platform == 'TELEGRAM' and self.platform[0] != 1:
            self.platform[0] = 1
        elif platform == 'WHATSAPP' and self.platform[1] != 1:
            self.platform[1] = 1
        else:
            self.platform = (0,0)

    def _set_status(self, status : Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION']):
          self.status = status

    def get_id(self) -> int:
          return self.id
    
    def get_status(self) -> Literal['DETOX', 'ANALYZE', 'EXPLAIN', 'DISTRIBUTION']:
          return self.status
    
    def get_conversation_classification(self) -> Tuple[str,str]:
          return self.conversation_classification
    
    def get_messages_conversation(self, conversation_id : str) -> List[str]:
        if conversation_id in self.messages_per_conversation:
              return self.messages_per_conversation[conversation_id] 
