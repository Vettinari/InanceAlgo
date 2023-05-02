import pandas as pd


class RewardBuffer:
    def __init__(self):
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[])

    def add_reward(self, reward_name, reward_value):
        assert reward_name in self._rewards.keys(), "Reward name not in reward buffer"
        self._rewards[reward_name] = reward_value

    def yield_rewards(self):
        out = self._rewards
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}
        return out

    def reset(self):
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[])
