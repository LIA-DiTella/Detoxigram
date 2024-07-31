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

    def __init__(self):
        self.model_language = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))
        self.model_name = "luzalbaposse/HelloBERT"
        self.classifier = pipeline("text-classification", model=self.model_name)

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
