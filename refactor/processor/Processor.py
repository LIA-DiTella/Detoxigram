from telebot import types
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest
from typing import Dict, List, Optional, Literal, Tuple
import re
from urllib.parse import parse_qs, urlparse
from collections import defaultdict

class Processor:
    def __init__(self, client):
        self.client = client

    async def fetch_telegram_messages(self, channel_name) -> List[str]:
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
        return posts.messages

    def transform_data(self, data) -> List[str]:
        transformed_data = [("user", item["message"]) for item in data]
        return transformed_data

    def process_messages(self, messages) -> list:
        processed_messages = []
        for msg in messages:
            if msg.message:
                processed_message = msg.message
                processed_messages.append(processed_message)
        return processed_messages
    
    def txt_to_list(self, archivo: str) -> List[str]:
        messages: List[str] = []
        phone_number_pattern = re.compile(r'\+?\d[\d -]{8,}\d')
        user_pattern = re.compile(r'\[.*?\] (.*?):')
        date_pattern = re.compile(r'^\[.*?\]')
        user_map: Dict[str, str] = defaultdict(lambda: f'usuario {len(user_map) + 1}')
        
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
    
    def _parse_channel_from_url(self, url) -> str:
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc in ['t.me', 'telegram.me']:
                channel_path = parsed_url.path.lstrip('/').lstrip('+')
                return channel_path
            else:
                return None
        except Exception as e:
            print('Had the error:', e)
            return None

    def obtain_channel_name(self, message) -> str:

        channel_name = message.strip()

        if channel_name.startswith('http'):
            parsed_name = self._parse_channel_from_url(channel_name)
            if parsed_name:
                return parsed_name
            else:
                self._ask_for_new_channel_name(message)
                return None

        elif channel_name.startswith('@'):
            return channel_name

        else:
            self._ask_for_new_channel_name(message)
            return None
        
