import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from telebot import types


class Summarizer:
    def __init__(self, bot, loop, formatter, gpt, bert, llm, output_parser, last_channel_analyzed):
        self.bot = bot
        self.loop = loop
        self.formatter = formatter
        self.gpt = gpt
        self.bert = bert
        self.llm = llm
        self.output_parser = output_parser
        self.last_channel_analyzed = last_channel_analyzed

    def summarize(self, message):
        markup = types.InlineKeyboardMarkup(row_width=1)
        summarize = types.InlineKeyboardButton('Summarize 📝', callback_data='summarize')
        go_back = types.InlineKeyboardButton('Restart! 🔄', callback_data='restart')
        new_analyze = types.InlineKeyboardButton('Analyze 🔍', callback_data='analyze')
        global last_channel_analyzed

        if self.last_channel_analyzed is not None:
            channel_name = last_channel_analyzed
            print(f"I'm here, and the last channel analyzed is {last_channel_analyzed}")
        else:
            channel_name = message.text
            last_channel_analyzed = channel_name
        
        if channel_name:
            self.bot.reply_to(message, f"Got it! I'll summarize {channel_name}... Please wait a moment 🙏")
            messages = self.loop.run_until_complete(self.formatter.fetch(channel_name))
            processed_messages = self.formatter.process_messages(messages)
            if len(processed_messages) > 0:
                self.bot.reply_to(message, "Give me a moment... 🤔")
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
                toxicity = str(toxicity)

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
                    **Note:** It should sound like a message from a human moderator, not a machine.
                    Your complete response, including the summary and toxicity evaluation, should be concise and informative, helping readers understand the channel's nature without exceeding the word limit for the summary.

                    EXAMPLES:
                    1. **Non-toxic:**
                    - Conversation:
                        ```
                        User1: I think we should focus on renewable energy to combat climate change.
                        User2: I agree, it's a sustainable solution.
                        User3: Absolutely, and it can create new jobs in the green sector.
                        ```
                    - Output: 
                    The channel involve a discussion on the benefits of renewable energy for climate change and job creation. This seems to be 🟢 Non-toxic. The conversation is respectful and constructive.

                    2. **Slightly Toxic:**
                    - Conversation:
                        ```
                        User1: The government's economic policies are not helping the middle class.
                        User2: I disagree, I think they're beneficial in the long run.
                        User3: Maybe, but they seem to favor the wealthy right now.
                        User4: That's a typical leftist argument, always blaming the rich.
                        ```
                    - Output: 
                    The channel involve a debate over the government's economic policies and their impact on the middle class. This seems to be 🟡 Slightly Toxic. The conversation includes some bias and a subtle lack of appreciation for differing viewpoints.

                    - Conversation:
                     
                     ```
                     [('user', 'BREAKING: Donald Trump is officially the GOP nominee for President\n\nhttps://thepostmillennial.com/breaking-donald-trump-is-officially-the-gop-nominee-for-president?utm_campaign=64501'), ('user', 'I want to introduce you to Gabriella Delorenzo and Megan Rothmund, two incredible TPUSA students who attend SUNY Cortland.\n\n"When I was 15 I discovered Turning Point. I immediately loved what the organization stood for. I loved the morals that it had, the views that it had."\n\n"When Gabriella reached out to me about becoming vice president for it, I was so excited, and it was exciting to become a part of something so great."\n\nThese two young women tried to start a TPUSA chapter at their school, but the Student Government refused to recognize the chapter. Instead they were belittled and demeaned for 100 minutes in public. Faculty even got involved and the university president responded by saying "We silence voices all the time in this country." \n\nThey persevered and SUNY Cortland reversed their decision in a massive victory for free speech. A huge shoutout to @ADFLegal for representing our students\' First Amendment rights, playing a huge role in the reversal. \n\nThere\'s still much work to do, but Gabriella and Megan are exactly why @TPUSA fights so hard to make sure there\'s a home for conservative students on America\'s campuses.\n\nOnward! 🇺🇸🔥'), ('user', 'Mark Robinson is poised to be the next governor of North Carolina, and the left is in a panic. In order to stop a conservative black man from making NC a new Florida, the left is planning to spend $100 million and is outrageously smearing him as a Holocaust denier… 🚨⬇️\n\nhttps://podcasts.apple.com/us/podcast/the-charlie-kirk-show/id1460600818?i=1000648968709'), ('user', 'What is with the spate of United Airlines mechanical failures involving it\'s San Francisco based aircraft?\n\n- Yesterday, a United Boeing 777-300 jet from Sydney to San Francisco was aborted when 10 seconds into its flight jet fuel began spilling from the rear right landing gear.\n\n- Last Thursday, a United 777-200 took off from San Francisco to Japan when a landing gear wheel fell off during take off. The wheel crashed onto two cars in an employee parking lot.\n\n- Last Friday, a United Airbus 320 heading from SFO to Mexico City had to re-route to Los Angeles because of hydraulic issues.\n\nRetired airline pilot Douglas Rice made this observation to NBC: “All three aircraft either originate or terminate in San Francisco."\n\nWoke city + Woke airlines = fly at your own risk!'), ('user', "It's been just days since the big leadership shakeup at the RNC, and already huge changes are evident, with dozens of members of the old guard being hit with layoffs… ⬇️⬇️⬇️\n\nhttps://podcasts.apple.com/us/podcast/the-charlie-kirk-show/id1460600818?i=1000648958491"), ('user', 'Ken Buck is resigning…. Next week. This give Republicans a ONE SEAT majority. \n\nCan’t state enough how cruel and terrible much of the DC Republican class it. It’s not about country, never was. \n\nInstead of seeing out the end of their term they can’t wait to bolt to CNN and get paid bashing Republicans. Repulsive.'),
                     ```
                    - Output:
                     The channel involves a mix of political news, including updates on leadership changes at the RNC, Joe Biden's budget proposal, and a whistleblower's death. The content varies from political commentary to breaking news. The messages exhibit a range of tones, from supportive to critical. Overall, the channel appears 🟡 Slightly Toxic due to some biased and emotionally charged comments, but it does not exhibit a pervasive toxicity issue.
                    3. **Moderately Toxic:**
                    - Conversation:
                        ```
                        User1: We need more investment in public healthcare.
                        User2: Agreed, everyone deserves access to healthcare.
                        User3: That's just socialist nonsense. Private healthcare is more efficient.
                        User4: You're so ignorant. Public healthcare is a basic right.
                        ```
                    - Output: 
                     The channel involve a discussion on public vs. private healthcare investment. This seems to be 🟡 Moderately Toxic. The conversation shows disrespect and aggression between differing opinions.

                    4. **Highly Toxic:**
                    - Conversation:
                        ```
                        User1: Immigration policies should be more inclusive.
                        User2: Definitely, diversity strengthens our society.
                        User3: Inclusive? More like inviting criminals and freeloaders.
                        User4: Watch your language! That's xenophobic and offensive.
                        ```
                    - Output: 
                     The channel involve a conversation on immigration policies met with xenophobic remarks. This seems to be 🔴 Highly Toxic. The conversation includes offensive language and clear contempt for inclusive views.

                    5. **Extremely Toxic:**
                    - Conversation:
                        ```
                        User1: It's important to address income inequality in our country.
                        User2: Yes, the wealth gap is a growing issue.
                        User3: Oh please, stop with the class warfare. You're just envious of the successful.
                        User4: People like you are the problem, promoting greed and division!
                        ```
                    - Output: 
                     The channel involve a discussion on income inequality met with accusations of envy and greed. This seems to be 🔴 Extremely Toxic. The conversation contains explicit aggression and personal attacks.

                                        """),

                    ("user", "Based on the first 100 messages provided and the toxicity level indicated, reply with a summary and your evaluation of the channel's toxicity."), 
                ]+ transformed_data + [toxicity])

                chain = prompt_template | self.llm | self.output_parser
                output = chain.batch([{}])
                print(output)
                markup.add(summarize, new_analyze, go_back)
                self.bot.reply_to(message, f'{output[0]}', reply_markup=markup)

            else:
                self.bot.reply_to(message, f'Failed! Try with another channel!')
        else:
            self.bot.reply_to(message, "Please provide a channel name!")

    def detoxify_single_message(self, message):
        self.bot.reply_to(message, "Let's see... 👀")
        markup = types.InlineKeyboardMarkup(row_width=1)
        go_back = types.InlineKeyboardButton('Start again! 🔄', callback_data='restart')
        toxicity_result = self.gpt.predictToxicity(message.text)
        _, toxicity_score = toxicity_result
        toxicity_score = str(toxicity_score)
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """As an AI language model, your task is to detoxify and provide non-toxic alternatives for the following messages if they are found to be toxic based on the provided toxicity scale.

Toxicity Scale:
0. **Non-toxic:** Messages promote a positive and respectful environment. They are inclusive and constructive, with no offensive content.
1. **Slightly Toxic:** Messages are mostly respectful but may include passive-aggressive criticism or slight bias.
2. **Moderately Toxic:** Messages have an aggressive tone or contain derogatory language towards specific groups.
3. **Highly Toxic:** Messages show clear contempt for individuals or groups, using insults or offensive language.
4. **Extremely Toxic:** Messages are aggressively disrespectful, with threats or calls to violent action.

**Task:**
Review the provided messages and determine their toxicity levels. If the messages are Slightly, Moderately, Highly, or Extremely toxic, suggest rephrased, non-toxic versions that convey the intended messages in a respectful and positive manner.

**Examples of detoxification:**

1. **Non-toxic:**
   - Original Message: "I appreciate your perspective and would like to discuss this further."
   - Output: This message is 🟢 Non-toxic. It promotes a respectful and open dialogue.
     
2. **Slightly Toxic:**
   - Original Message: "That's a naive way of looking at things, don't you think?"
   - Output: This message is 🟡 Slightly Toxic due to its patronizing tone. A more respectful phrasing could be: "I think there might be a different way to view this situation. Can we explore that together?"

3. **Moderately Toxic:**
   - Original Message: "People who believe that are living in a fantasy world."
   - Output: This message is 🟡 Moderately Toxic because it dismisses others' beliefs. A less toxic version could be: "I find it hard to agree with that perspective, but I'm open to understanding why people might feel that way."

4. **Highly Toxic:**
   - Original Message: "This is the dumbest idea I've ever heard."
   - Output: The message is 🔴 Highly Toxic due to its derogatory language. A constructive alternative might be: "I have some concerns about this idea and would like to discuss them further."

5. **Extremely Toxic:**
   - Original Message: "Anyone who supports this policy must be a complete idiot. We should kill them all, they don't deserve to exist."
   - Output: This message is 🔴 Extremely Toxic and offensive. A non-toxic rephrasing could be: "I'm surprised that there's support for this and would like to understand the reasoning behind it."

Now, please detoxify the following message:
    """),
    ("user", message.text),
    ("system", f"Toxicity level: {toxicity_score}.") 
])
        chain = prompt_template | self.llm | self.output_parser
        output = chain.batch([{}])
        print(output)
        markup.add(go_back)
        self.bot.reply_to(message, f'{output[0]}', reply_markup=markup)

        
    