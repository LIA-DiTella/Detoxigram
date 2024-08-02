from pywa import WhatsApp, filters
from pywa.types import Message, CallbackButton, Button, Document
from typing import List, Tuple
import telebot
from telebot import types
from telebot.types import Message, CallbackQuery
from telethon import TelegramClient, sessions
from telethon.tl.functions.messages import GetHistoryRequest

class WhatsApp_Messager:
    def __init__(self, client:WhatsApp):
        self.client = client

    def send_message(self, text:str, num:str):
        self.client.send_text(
        to=num,  
        text=text
    )
    
    def send_message_with_buttons(self, text:str, num:str, botones:List[Tuple[str,str]]):
        aux = []
        z = 0 
        for i in botones:
            aux[z] = Button(title = i[0], callback_data=i[1])
            z+=1

        self.client.send_text(
        to=num,  
        text=text,
        buttons=aux)

    def send_message_with_images(self, text:str, num:str, img:str):
        aux = []
        self.client.send_image(
            to=num,
            image = img,
            caption = text)
        
    def send_sticker(self, num:str, sti:str):
        self.client.send_sticker(
            to=num,
            sticker = sti
        )

class Telegram_Messager:
    def __init__(self, bot: telebot):
        self.bot = bot

    def send_message(self, text:str, callback:CallbackQuery):
        self.bot.send_message(callback, text)
    
    def send_message_with_buttons(self, text:str, botones:List[Tuple[str,str]], callback:CallbackQuery, rw:int):
        aux = []
        z = 0 
        markup:types = types.InlineKeyboardMarkup(row_width=rw)
        for i in botones:
            markup.add(types.InlineKeyboardButton(i[0], callback_data=i[1]))

        self.bot.send_message(callback, text, reply_markup = markup)

    def send_image(self, img:str, callback:CallbackQuery):
        self.bot.send_photo(callback, img)

    def send_sticker(self, path:str, callback:CallbackQuery):
        sti = open(path, 'rb')
        self.bot.send_sticker(callback, sti)