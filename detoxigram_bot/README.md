
# Detoxigram Bot

## Summarizer Class

The `Summarizer` class is responsible for summarizing messages from Telegram channels and analyzing their toxicity using BERT and GPT classifiers.

### Initialization

The class is initialized with the bot, event loop, formatter, BERT and GPT classifiers, language model, and output parser:

```python
from summarizer import Summarizer

summarizer = Summarizer(bot, loop, formatter, bert, gpt, llm, output_parser)
```

### Methods

- `summarizor_gpt(data, channel_name)`: Generates a summary of the messages and evaluates their toxicity.

- `summarize(message)`: Handles the bot's response and triggers the summarization process.

## ChannelAnalyzer Class

The `ChannelAnalyzer` class is responsible for analyzing the toxicity of messages in a Telegram channel using the BERT classifier.

### Initialization

The class is initialized with the bot, event loop, formatter, and BERT classifier:

```python
from channel_analyzer import ChannelAnalyzer

channel_analyzer = ChannelAnalyzer(bot, loop, formatter, bert)
```

### Methods

- `analyze_channel_bert(message)`: Analyzes the toxicity of messages in the specified channel and provides feedback to the user.

## Formatter Class

The `Formatter` class is a utility class that handles the fetching, transformation, and processing of messages from Telegram channels. It encapsulates the functionality required to retrieve the last 50 messages from a specified channel, transform the data into the format we need, and process the messages to extract relevant information.

### Initialization

The class is initialized with a `TelegramClient` instance:

```python
from telethon import TelegramClient
from formatter import Formatter

client = TelegramClient('session_name', api_id, api_hash)
formatter = Formatter(client)
```

### Methods

- `fetch_last_50_messages(channel_name)`: Asynchronously fetches the last 50 messages from the specified Telegram channel.

```python
messages = await formatter.fetch_last_50_messages('@channel_name')
```

- `transform_data_to_expected_format(data)`: Transforms the data into a format suitable for further processing. In this case, it creates a list of tuples with the format `("user", message)`.

```python
transformed_data = formatter.transform_data_to_expected_format(messages)
```

- `process_messages(messages)`: Processes the messages to extract the message content and timestamp. Returns a list of dictionaries, each containing the message and its timestamp.

```python
processed_messages = formatter.process_messages(transformed_data)
```

