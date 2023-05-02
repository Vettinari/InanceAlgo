class RewardBuffer:
    def __init__(self):
        self._rewards = {}

    def add_reward(self, reward_name, reward_value):
        self._rewards[reward_name] = reward_value

    def yield_rewards(self):
        out = self._rewards
        self._rewards = []
        return out
