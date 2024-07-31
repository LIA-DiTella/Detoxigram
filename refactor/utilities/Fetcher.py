from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from typing import Dict, List, Optional, Literal, Tuple, Any
import re
import zipfile
import os
from urllib.parse import urlparse
from collections import defaultdict
from huggingface_hub import hf_hub_download
from transformers import pipeline


class Telegram_Fetcher:
    def __init__(self, client: TelegramClient):
        self.telegram = TelegramClient

    async def fetch(self, channel_name: str) -> List[str]:
        """
        Obtiene los mensajes de un canal de Telegram.

        Parámetros:
        - channel_name: Nombre del canal de Telegram.

        Retorna:
        - Lista de mensajes del canal.
        """
        try:
            await self.client.start()
            channel_entity = await self.client.get_entity(channel_name)
            posts = await self.client(GetHistoryRequest(
                peer=channel_entity,
                limit=50,
                offset_date=None,
                offset_id=0,
                max_id=0,
                min_id=0,
                add_offset=0,
                hash=0
            ))
            await self.client.disconnect()
            return [msg.message for msg in posts.messages if msg.message]
        except Exception as e:
            print(f'Error fetching messages: {e}')
            return []

    # def transform_data(self, data: List[Dict[str, Any]]) -> List[str]:
    #     return [item["message"] for item in data]

    # def process_messages(self, messages: List[str]) -> List[str]:
    #     return [msg for msg in messages if msg]

class WhatsApp_Fetcher:

    def __init__(self):
        return
        
    def zip_extract(self, archivo:str, output:str):
            os.makedirs(output, exist_ok=True)

            with zipfile.ZipFile(archivo, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith('.txt'):
                        zip_ref.extract(file, output)

            
    def txt_to_list(self, archivo: str) -> List[str]:
            """
            Convierte un archivo de texto en una lista de mensajes, eliminando números de teléfono y fechas,
            y reemplazando nombres de usuario.

            Parámetros:
            - archivo: Ruta del archivo de texto.

            Retorna:
            - Lista de mensajes procesados.
            """
            archivo = self.zip_extract(archivo, './output')
            messages = []
            phone_number_pattern = re.compile(r'\+?\d[\d -]{8,}\d')
            user_pattern = re.compile(r'\[.?\] (.?):')
            date_pattern = re.compile(r'^\[.*?\]')
            user_map = defaultdict(lambda: f'usuario {len(user_map) + 1}')
            
            with open(archivo, 'r', encoding='utf-8') as file:
                for line in file:
                    # Eliminar fechas
                    line = date_pattern.sub('', line)
                    # Reemplazar números de teléfono
                    line = phone_number_pattern.sub('[PHONE NUMBER]', line)
                    # Reemplazar nombres de usuario
                    match = user_pattern.search(line)
                    if match:
                        user = match.group(1)
                        if user not in user_map:
                            user_map[user] = f'usuario {len(user_map) + 1}'
                        line = user_pattern.sub(f'{user_map[user]}:', line)
                    messages.append(line.strip())
            
            return messages    

    def fetch(self, archivo:str):
        conversacion:List[str] = self.txt_to_list(archivo)
        return conversacion




wpp = WhatsApp_Fetcher()
mensajes = wpp.fetch('./test.zip')
print(mensajes)