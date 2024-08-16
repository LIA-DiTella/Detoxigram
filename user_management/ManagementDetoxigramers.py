# @REFACTOR: Queda pendiente generar Base de Datos -> Conectar a Django

import os
import sys
from typing import Dict, List, Optional, Literal, Tuple
from .Detoxigramer import WhatsApp_Detoxigramer, Telegram_Detoxigramer
sys.path.append('..')


class ManagementDetoxigramers_Whatsapp:
    '''
            Invariante de Representación:
            - self.detoxigramers: Dict[str, Detoxigramer]
                - Las claves son cadenas no vacías que representan los IDs de los usuarios.
                - Los valores son instancias de Detoxigramer.
                
            Observadores:
            - get_detoxigramer: Devuelve el estado del Detoxigramer correspondiente al user_id.
            - get_all_detoxigramers: Devuelve una lista de todas las instancias de Detoxigramer.

            Modificadores:
            - set_detoxigramer: Establece el Detoxigramer para un user_id específico.
            - reset_detoxigramer: Resetea el estado de un Detoxigramer dado un user_id.
            - remove_detoxigramer: Elimina el Detoxigramer correspondiente a un user_id.
            
    '''
    def __init__(self):
        self.detoxigramers: Dict[str, WhatsApp_Detoxigramer] = {}

    def get_detoxigramer(self, user_id: str) -> WhatsApp_Detoxigramer:
        if user_id not in self.detoxigramers:
            self.detoxigramers[user_id] = WhatsApp_Detoxigramer()
            self.detoxigramers[user_id].id = user_id  
        return self.detoxigramers[user_id]
    
    def set_detoxigramer(self, user_id: str, detoxigramer: WhatsApp_Detoxigramer):
        self.detoxigramers[user_id] = detoxigramer

    def reset_detoxigramer(self, user_id: str):
        if user_id in self.detoxigramers:
            self.detoxigramers[user_id] = WhatsApp_Detoxigramer()
            self.detoxigramers[user_id].id = user_id
    
    def get_all_detoxigramers(self) -> List[WhatsApp_Detoxigramer]:
        return list(self.detoxigramers.values())

    def remove_detoxigramer(self, user_id: str):
        if user_id in self.detoxigramers:
            del self.detoxigramers[user_id]


class ManagementDetoxigramers_Telegram:
    '''
            Invariante de Representación:
            - self.detoxigramers: Dict[str, Detoxigramer]
                - Las claves son cadenas no vacías que representan los IDs de los usuarios.
                - Los valores son instancias de Detoxigramer.
                
            Observadores:
            - get_detoxigramer: Devuelve el estado del Detoxigramer correspondiente al user_id.
            - get_all_detoxigramers: Devuelve una lista de todas las instancias de Detoxigramer.

            Modificadores:
            - set_detoxigramer: Establece el Detoxigramer para un user_id específico.
            - reset_detoxigramer: Resetea el estado de un Detoxigramer dado un user_id.
            - remove_detoxigramer: Elimina el Detoxigramer correspondiente a un user_id.
            
    '''
    def __init__(self):
        self.detoxigramers: Dict[str, Telegram_Detoxigramer] = {}

    def get_detoxigramer(self, user_id: str) -> Telegram_Detoxigramer:
        if user_id not in self.detoxigramers:
            self.detoxigramers[user_id] = Telegram_Detoxigramer()
            self.detoxigramers[user_id].id = user_id  
        return self.detoxigramers[user_id]
    
    def set_detoxigramer(self, user_id: str, detoxigramer: Telegram_Detoxigramer):
        self.detoxigramers[user_id] = detoxigramer

    def reset_detoxigramer(self, user_id: str):
        if user_id in self.detoxigramers:
            self.detoxigramers[user_id] = Telegram_Detoxigramer()
            self.detoxigramers[user_id].id = user_id
    
    def get_all_detoxigramers(self) -> List[Telegram_Detoxigramer]:
        return list(self.detoxigramers.values())

    def remove_detoxigramer(self, user_id: str):
        if user_id in self.detoxigramers:
            del self.detoxigramers[user_id]
