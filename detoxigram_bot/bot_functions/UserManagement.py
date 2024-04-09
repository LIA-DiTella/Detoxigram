class UserState:
    def __init__(self):
        self.last_analyzed_toxicity = None
        self.last_channel_analyzed = None
        self.last_chunk_of_messages = None

class UserManagement:
    def __init__(self):
        self.user_states = {}

    def get_user_state(self, user_id):
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState()
        return self.user_states[user_id]

    def reset_user_state(self, user_id):
        if user_id in self.user_states:
            self.user_states[user_id] = UserState()
