from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from typing import Dict, List, Optional, Literal, Tuple, Any
import re
import zipfile
import os
from urllib.parse import urlparse
from collections import defaultdict
from huggingface_hub import hf_hub_download
import fasttext
from transformers import pipeline

class Utilities:
    """
    Varias operaciones de procesamiento de datos, manejo de mensajes de Telegram,
    y detección de lenguaje y saludos.

    """

    def __init__(self, client: TelegramClient):
        self.model_language = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))
        self.model_name = "luzalbaposse/HelloBERT"
        self.classifier = pipeline("text-classification", model=self.model_name)
        self.client = client

    async def fetch_telegram_messages(self, channel_name: str) -> List[str]:
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

    def transform_data(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Transforma los datos de entrada en una lista de mensajes.

        Parámetros:
        - data: Lista de diccionarios con los datos de entrada.

        Retorna:
        - Lista de mensajes transformados.
        """
        return [item["message"] for item in data]

    def process_messages(self, messages: List[str]) -> List[str]:
        """
        Procesa una lista de mensajes para extraer solo el texto del mensaje.

        Parámetros:
        - messages: Lista de mensajes de entrada.

        Retorna:
        - Lista de mensajes procesados.
        """
        return [msg for msg in messages if msg]

    def zip_extract(self, archivo:str, output:str) -> str:
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
        user_pattern = re.compile(r'\[.*?\] (.*?):')
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

    def _parse_channel_from_url(self, url: str) -> Optional[str]:
        """
        Analiza una URL de Telegram y extrae el nombre del canal.

        Parámetros:
        - url: URL del canal de Telegram.

        Retorna:
        - Nombre del canal si la URL es válida, de lo contrario None.
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc in ['t.me', 'telegram.me']:
                return parsed_url.path.lstrip('/').lstrip('+')
            else:
                return None
        except Exception as e:
            print(f'Error parsing URL: {e}')
            return None

    def obtain_channel_name(self, message: str) -> Optional[str]:
        """
        Obtiene el nombre del canal a partir de un mensaje.

        Parámetros:
        - message: Mensaje que contiene el nombre o URL del canal.

        Retorna:
        - Nombre del canal si es válido, de lo contrario None.
        """
        channel_name = message.strip()

        if channel_name.startswith('http'):
            parsed_name = self._parse_channel_from_url(channel_name)
            if parsed_name:
                return parsed_name
            else:
                print("Invalid channel URL.")
                return None
        elif channel_name.startswith('@'):
            return channel_name
        else:
            print("Invalid channel name.")
            return None

    def language_detection(self, message: str) -> Literal['EN', 'ES']:
        """
        Detecta el idioma de un mensaje.

        Parámetros:
        - message: Mensaje a analizar.

        Retorna:
        - 'EN' si el mensaje está en inglés, 'ES' si está en español.
        """
        prediction = self.model_language.predict(message)
        label = prediction[0][0]
        if label == "__label__en":
            return 'EN'
        elif label == "__label__es":
            return 'ES'
        else:
            return 'UNKNOWN'

    def greeting_detection(self, message: str) -> Literal['GREETING', 'NONE']:
        """
        Detecta si un mensaje es un saludo.

        Parámetros:
        - message: Mensaje a analizar.

        Retorna:
        - 'GREETING' si el mensaje es un saludo, 'NONE' de lo contrario.
        """
        prediction = self.classifier(message)
        if prediction[0]['score'] == 'greeting':
            return 'GREETING'
        else:
            return 'NONE'
