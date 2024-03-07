import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    Your role as a moderator involves two key tasks:

                    Summarize the content of the following messages in a concise paragraph, limiting your summary to no more than 50 words. The summary should remain objective and exclude toxic content.
                    Evaluate the channel's toxicity level based on the content you've summarized, providing a brief explanation for your assessment according to the toxicity scale provided.
                    Toxicity Scale:

                    0. **Non-toxic:** Message that encourages a positive, respectful, and inclusive environment, promoting **kindness** and mutual understanding. The opinions and perspectives of everyone are valued, contributing to constructive dialogue without personal disqualifications or offensive content. It may contain objective or neutral information.
                    1. **Slightly Toxic:** Message that, although mostly respectful, may include criticism or disagreements expressed in a passive-aggressive or slightly biased manner. It does not directly attack individuals or groups, and the intention to cause harm is not clear, but it suggests a lack of appreciation for the viewpoint of others.
                    2. **Moderately Toxic:** Message with an aggressive or disrespectful tone, which may include sarcasm, irony, or derogatory language towards certain groups by gender, ethnicity, sexual orientation, ideology, or religion. Although it does not attack violently, **it seeks to hurt, ridicule, or belittle others**, showing a rejection towards the diversity of opinions and people.
                    3. **Highly Toxic:** Message that demonstrates a clear rejection and contempt for individuals or groups, using insults, racist, sexist, misogynist, degrading, or dehumanizing references. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. This offensive language seeks to intimidate, **exclude**, or **incite** hatred, showing an explicit intention to cause harm.
                    4. **Extremely Toxic:** Message that is not only explicitly aggressive and disrespectful but also contains threats or calls to violent action. It attacks groups by their gender, ethnicity, sexual orientation, ideology, or religion. Promotes hostility, incitement to hatred, and suggests harmful consequences in the real world against individuals or groups, violating ethical and moral principles and endangering the safety and well-being of people.
                    
                    Instructions:

                    For Non-toxic content, include it directly in the summary.
                    For Slightly to Moderately Toxic content, incorporate it carefully, filtering out any bias or offensive elements.
                    For Highly to Extremely Toxic content, exclude the offensive details and focus on objective information. Additionally, provide a warning about the content's potential impact.
                    After summarizing, conclude with your evaluation of the channel's overall toxicity. Consider the prevalence of toxic messages and their severity to explain whether the channel is generally toxic or not and why.

                    Remember: Your complete response, including the summary and toxicity evaluation, should be concise and informative, helping readers understand the channel's nature without exceeding the word limit for the summary.
                     
                    ### EXAMPLES
                     Understood, here are conversation examples for each level of toxicity:

                        1. **Non-toxic:**
                        - Conversation:
                            ```
                            User1: I think renewable energy is a great solution for combating climate change.
                            User2: I do not agree with you, but you may have a point! It may be important to transition to sustainable energy sources.

                            Output:
                            It seems like the channel's toxicity level is 🟢 Non-toxic. The conversation encourages respectful dialogue and values diverse viewpoints, contributing to a constructive exchange of ideas.      The conversation focuses on constructive discussion about renewable energy, promoting a positive and inclusive environment.

                            ```

                        2. **Slightly Toxic:**
                        - Conversation:
                            ```
                            User1: I prefer Android phones over iPhones because they offer more customization options.
                            User2: Well, that's your opinion, but iPhones have better security features.

                            Output:
                            It seems like the channel's toxicity level is 🟡 Slightly Toxic. While the conversation maintains a mostly respectful tone, there's a subtle lack of appreciation for the opposing viewpoint, potentially hindering constructive discourse.
                        While the conversation addresses differing viewpoints, User2's response comes off as slightly dismissive, implying superiority in their preference.

                                            ```

                        3. **Moderately Toxic:**
                        - Conversation:
                            ```
                            User1: I'm considering becoming a vegetarian for ethical reasons.
                            User2: Don't be ridiculous. Humans are meant to eat meat. You'll just end up malnourished.
                            Output:
                            It seems like the channel's toxicity level is 🟡 Moderately Toxic. User2's response shows a rejection towards User1's perspective and lacks appreciation for their decision, potentially discouraging open dialogue.
                            User2's response dismisses User1's decision and exhibits a disrespectful attitude towards their dietary choice.

                                            ```

                        4. **Highly Toxic:**
                        - Conversation:
                            ```
                            User1: I support equal rights for all genders.
                            User2: You're just a feminazi pushing your agenda. Men are the real victims here.
                            Output:
                            It seems like the channel's toxicity level is 🔴 Highly Toxic. User2's response demonstrates a clear rejection of User1's viewpoint, using offensive language and inciting hatred towards a specific group.
                                User2's response contains derogatory language and dismisses User1's stance on gender equality, exhibiting clear contempt and hostility.

                                            ```

                        5. **Extremely Toxic:**
                        - Conversation:
                            ```
                            User1: I believe in freedom of speech and expression.
                            User2: Shut your mouth, you fascist pig. Your words have consequences, and I hope you suffer for them.
                            Output:
                            It seems like the channel's toxicity level is 🔴 Extremely Toxic. User2's response promotes violence and poses a severe danger to User1's well-being, crossing ethical and moral boundaries.
                                User2's response contains explicit aggression and threats towards User1, violating ethical principles and endangering their safety.

                      ```
                     """),

                    ("user", " Reply with the summary of the following messages (which is the first 50 message I attach here) and the toxicity level of the channel, which is the second number I attach \n"), 
                ]+ transformed_data + [toxicity])

        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{}])
        return output

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