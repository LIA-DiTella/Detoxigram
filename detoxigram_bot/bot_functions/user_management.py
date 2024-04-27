class UserState:
    def __init__(self):
        self.is_detoxifying = False
        self.last_analyzed_toxicity = 0
        self.last_channel_analyzed = None
        self.last_chunk_of_messages = None

    @property
    def last_toxicity(self):
        if self.last_analyzed_toxicity < 1:
            return "🟢 Non-toxic"
        elif 1 <= self.last_analyzed_toxicity < 2:
            return "🟡 Slightly toxic"
        elif 2 <= self.last_analyzed_toxicity < 3:
            return "🟡 Moderately toxic"
        elif 3 <= self.last_analyzed_toxicity < 3.5:
            return "🔴 Highly toxic"
        else:
            return "🔴 Extremely toxic"
    
    def update_channel_analysis(self, channel_name, messages):
        self.last_channel_analyzed = channel_name
        self.last_chunk_of_messages = messages

    def __str__(self):
        return f"UserState(detoxifying={self.is_detoxifying}, last_analyzed_toxicity={self.last_analyzed_toxicity}, last_channel_analyzed={self.last_channel_analyzed})"

 
class UserManagement:
    def __init__(self):
        self.user_states = {}

    def get_user_state(self, user_id):
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState()
        return self.user_states[user_id]
    
    def set_user_state(self, user_id, state):
        self.user_states[user_id] = state

    def reset_user_state(self, user_id):
        if user_id in self.user_states:
            self.user_states[user_id] = UserState()
    
        