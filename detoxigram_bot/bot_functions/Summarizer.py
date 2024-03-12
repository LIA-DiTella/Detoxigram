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

    def summarize(self, message):
        global last_channel_analyzed
        if last_channel_analyzed:
            channel_name = last_channel_analyzed
        else:
            channel_name = message.text
            last_channel_analyzed = channel_name

        if channel_name:
            self.bot.reply_to(message, f"Got it! I'll summarize {channel_name}... Please wait a moment 🙏")
            messages = self.loop.run_until_complete(self.formatter.fetch_last_50_messages(channel_name))
            processed_messages = self.formatter.process_messages(messages)
            if len(processed_messages) > 0:
                self.bot.reply_to(message, "Give me one sec... 🤔")
                data = processed_messages[:20]
                transformed_data = self.formatter.transform_data_to_expected_format(data)
                print(transformed_data)

                message_gpt = 0
                message_bert = 0
                for msg in data:
                    toxicity_result_gpt = self.gpt.predictToxicity(msg['message'])
                    _, numeric_toxicity_gpt = toxicity_result_gpt
                    message_gpt += numeric_toxicity_gpt

                    toxicity_result_bert = self.bert.predictToxicity(msg['message'])
                    _, numeric_toxicity_bert = toxicity_result_bert
                    message_bert += numeric_toxicity_bert

                average_gpt = message_gpt / len(data)
                average_bert = message_bert / len(data)
                toxicity = (average_gpt + average_bert) / 2

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", """
                    Your tasks as a moderator are:

                    1. **Summarize Messages:** Provide a concise summary of the following messages, up to 55 words. Keep the summary objective and exclude any toxic content.
                    2. **Evaluate Toxicity:** Assess the channel's overall toxicity level based on the summarized content. Use the provided toxicity scale for guidance.

                    Toxicity Scale:

                    0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
                    1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
                    2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
                    3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
                    4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

                    **Note:** For Non-toxic content, include it directly in the summary. For Slightly to Moderately Toxic content, incorporate it carefully, filtering out any bias or offensive elements. For Highly to Extremely Toxic content, exclude the offensive details and focus on objective information. Additionally, provide a warning about the content's potential impact. After summarizing, conclude with your evaluation of the channel's overall toxicity. Consider the prevalence of toxic messages and their severity to explain whether the channel is generally toxic or not and why.

                    Your complete response, including the summary and toxicity evaluation, should be concise and informative, helping readers understand the channel's nature without exceeding the word limit for the summary.

                    ### EXAMPLES
                    1. **Non-toxic:**
                    - Conversation:
                        ```
                        User1: I think we should focus on renewable energy to combat climate change.
                        User2: I agree, it's a sustainable solution.
                        User3: Absolutely, and it can create new jobs in the green sector.
                        ```
                    - Output: Summary: Discussion on the benefits of renewable energy for climate change and job creation. Toxicity Level: 🟢 Non-toxic. The conversation is respectful and constructive.

                    2. **Slightly Toxic:**
                    - Conversation:
                        ```
                        User1: The government's economic policies are not helping the middle class.
                        User2: I disagree, I think they're beneficial in the long run.
                        User3: Maybe, but they seem to favor the wealthy right now.
                        User4: That's a typical leftist argument, always blaming the rich.
                        ```
                    - Output: Summary: Debate over the government's economic policies and their impact on the middle class. Toxicity Level: 🟡 Slightly Toxic. The conversation includes some bias and a subtle lack of appreciation for differing viewpoints.

                    3. **Moderately Toxic:**
                    - Conversation:
                        ```
                        User1: We need more investment in public healthcare.
                        User2: Agreed, everyone deserves access to healthcare.
                        User3: That's just socialist nonsense. Private healthcare is more efficient.
                        User4: You're so ignorant. Public healthcare is a basic right.
                        ```
                    - Output: Summary: Discussion on public vs. private healthcare investment. Toxicity Level: 🟡 Moderately Toxic. The conversation shows disrespect and aggression between differing opinions.

                    4. **Highly Toxic:**
                    - Conversation:
                        ```
                        User1: Immigration policies should be more inclusive.
                        User2: Definitely, diversity strengthens our society.
                        User3: Inclusive? More like inviting criminals and freeloaders.
                        User4: Watch your language! That's xenophobic and offensive.
                        ```
                    - Output: Summary: Conversation on immigration policies met with xenophobic remarks. Toxicity Level: 🔴 Highly Toxic. The conversation includes offensive language and clear contempt for inclusive views.

                    5. **Extremely Toxic:**
                    - Conversation:
                        ```
                        User1: It's important to address income inequality in our country.
                        User2: Yes, the wealth gap is a growing issue.
                        User3: Oh please, stop with the class warfare. You're just envious of the successful.
                        User4: People like you are the problem, promoting greed and division!
                        ```
                    - Output: Summary: Discussion on income inequality met with accusations of envy and greed. Toxicity Level: 🔴 Extremely Toxic. The conversation contains explicit aggression and personal attacks.

                                        """),

                    ("user", "Based on the first 100 messages provided and the toxicity level indicated, reply with a summary and your evaluation of the channel's toxicity."), 
                ]+ transformed_data + [toxicity])

                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{}])
                print(output)
                self.bot.reply_to(message, f'{output[0]}')
            else:
                self.bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            self.bot.reply_to(message, "Please provide a channel name!")
