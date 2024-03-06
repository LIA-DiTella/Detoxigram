import time

class Summarizer:
    def __init__(self, bot, loop, formatter, gpt, bert, llm, output_parser):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.gpt = gpt
        self.bert = bert
        self.llm = llm
        self.output_parser = output_parser

    def summarizor_gpt(self, data, channel_name) -> str:
        transformed_data = self.formatter.transform_data_to_expected_format(data)
        print(transformed_data)
        if channel_name:
            start = time.time()
            messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
            end = time.time()
            print(f"Time to fetch messages: {end - start}")
            self.bot.reply_to(channel_name, "I've got the messages! Now give me one second to summarize them... 🕣")
            processed_messages = self.formatter.process_messages(messages)
            if len(processed_messages) > 0:
                data = processed_messages[:50]
                message_gpt = 0
                message_bert = 0
                for msg in data:
                    toxicity_result = self.gpt.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_gpt += numeric_toxicity
                average_gpt = message_gpt / len(data)
                print(average_gpt)
                for msg in data:
                    toxicity_result = self.bert.predictToxicity(msg['message'])
                    _, numeric_toxicity = toxicity_result
                    message_bert += numeric_toxicity
                average_bert = message_bert / len(data)
                print(average_bert)
                toxicity = (average_gpt + average_bert) / 2
            # Additional logic for creating the prompt template and generating the output
            return "Summarized output"

    def summarize(self, message):
        global last_channel_analyzed
        if last_channel_analyzed:
            self.bot.reply_to(message, f"Got it! I'll summarize {last_channel_analyzed}... Please wait a moment 🙏")
            channel_name = last_channel_analyzed
            messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
            processed_messages = self.formatter.process_messages(messages)
            if len(processed_messages) > 0:
                self.bot.reply_to(message, "Give me one sec... 🤔")
                data = processed_messages[:20]
                output = self.summarizor_gpt(data, channel_name)
                print(output)
                self.bot.reply_to(message, f'{output[0]}')
            else:
                self.bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            self.bot.reply_to(message, "Which channel would you like to summarize? 🤔")
            channel_name = message.text
            if channel_name:
                messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
                processed_messages = self.formatter.process_messages(messages)
                if len(processed_messages) > 0:
                    self.bot.reply_to(message, "Give me one sec... 🤔")
                    data = processed_messages[:20]
                    output = self.summarizor_gpt(data, channel_name)
                    print(output)
                    self.bot.reply_to(message, f'{output[0]}')
                else:
                    self.bot.reply_to(message, f'Failed! Try with another channel!')
            else:
                self.bot.reply_to(message, "Please provide a channel name!")

