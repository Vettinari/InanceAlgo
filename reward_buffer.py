from typing import Optional

import pandas as pd


class RewardBuffer:
    def __init__(self):
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[], dtype=float)
        self.trade_risk = None

    def add_reward(self, reward_name, reward_value):
        assert reward_name in self._rewards.keys(), "Reward name not in reward buffer"
        self._rewards[reward_name] = reward_value

    def yield_rewards(self):
        out = self._rewards

        self.rewards_history = pd.concat([self.rewards_history,
                                          pd.DataFrame(out, index=[0])],
                                         axis=0, ignore_index=True)
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}

        return out

    def reset(self):
        self._rewards = {'action': 0, 'transaction': 0, 'drawdown': 0}
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[])

    def log_info(self, sum_only=False) -> dict:
        """
        :returns: total_rewards and rewards for each reward type
        :param sum_only: if True, only returns the sum of each reward type
        """
        sums = {f"{key}_reward_sum": val for key, val in self.rewards_history.sum(axis=0).to_dict().items()}
        sums.update({"total_rewards": sum(sums.values())})
        if not sum_only:
            means = {f"{key}_mean": val for key, val in self.rewards_history.mean(axis=0).to_dict()}
            sums.update(means)
        return sums

    def set_trade_risk(self, trade_risk):
        self.trade_risk = trade_risk
