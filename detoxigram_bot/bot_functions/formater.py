from telebot import types
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest

class formater:
    def __init__(self, client):
        self.client = client

    async def fetch(self, channel_name) -> list:
        '''
        Requires: channel_name
        Returns: list of messages
        '''
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

    def transform_data_to_expected_format(self, data):
        transformed_data = [("user", item["message"]) for item in data]
        return transformed_data

    def process_messages(self, messages) -> list:
        processed_messages = []
        for msg in messages:
            if msg.message:
                processed_message = msg.message
                processed_messages.append(processed_message)
        return processed_messages
