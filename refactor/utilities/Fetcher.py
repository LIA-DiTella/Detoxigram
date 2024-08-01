from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from typing import Dict, List, Optional, Literal, Tuple, Any
import re
import zipfile
import os
from urllib.parse import urlparse
from collections import defaultdict
import shutil
from telethon import TelegramClient, sessions
import asyncio


class Telegram_Fetcher:
    def __init__(self, client: TelegramClient): 
        self.client = client

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
    def extract_txt_from_zip(self, zip_path: str, extract_to: str) -> str:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.txt'):
                    zip_ref.extract(file, extract_to)
                    return os.path.join(extract_to, file)
        return None

    def txt_to_list(self, archivo: str) -> List[str]:
        messages = []

        message_pattern = re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[ap]\.\s*m\.\]\s*(.*?):\s*(.*)$')
        phone_pattern = re.compile(r'@\d{10,15}')

        with open(archivo, 'r', encoding='utf-8') as file:
            for line in file:
                match = message_pattern.match(line)
                if match:
                    user = match.group(1).strip()
                    message = match.group(2).strip()
                    # Eliminar números de teléfono en el formato @549...
                    message = phone_pattern.sub('@', message)
                    formatted_message = f'{user}: {message}'
                    messages.append(formatted_message)
                else:
                    print(f"No match for line: {line.strip()}")  # Línea de depuración
        
        return messages  
   

    def fetch(self, archivo: str) -> List[str]:
        extract_to = './extracted_files'
        txt_file = self.extract_txt_from_zip(archivo, extract_to)
        if txt_file:
            conversacion = self.txt_to_list(txt_file)
            # Eliminar el archivo ZIP, el archivo .txt y la carpeta de salida
            os.remove(archivo)
            os.remove(txt_file)
            shutil.rmtree(extract_to)
            return conversacion
        else:
            raise FileNotFoundError("No .txt file found in the ZIP archive")


client:TelegramClient = TelegramClient(sessions.MemorySession(), API_ID_TELEGRAM, API_HASH_TELEGRAM)  

telegram = Telegram_Fetcher(client)
messages = asyncio.run(telegram.fetch('@TrumpJr'))
print(messages)

