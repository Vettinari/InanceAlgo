from pprint import pprint
from typing import Optional, Dict, Tuple

import pandas as pd


class RewardBuffer:
    def __init__(self):
        self._rewards: Dict[str, float] = {}
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[], dtype=float)
        self.trade_risk = None
        self.reset_reward_dict()

    def add_reward(self, reward_name, reward_value):
        self._rewards[reward_name] = reward_value

    def yield_rewards(self, trade_risk: float, max_gain: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        # Scale rewards
        self.process_rewards(trade_risk=trade_risk, max_gain=max_gain)
        # Add rewards to history
        self.rewards_history = pd.concat([self.rewards_history,
                                          pd.DataFrame(self._rewards, index=[0])],
                                         axis=0, ignore_index=True)
        # Prepare info for logging
        rewards_info = self.get_log_info()
        # Reset rewards
        out = self._rewards

        rewards_info = dict(zip(rewards_info.keys(), [round(val, 5) for val in rewards_info.values()]))
        out = dict(zip(out.keys(), [round(val, 5) for val in out.values()]))

        self.reset_reward_dict()
        return out, rewards_info

    def reset(self):
        self.reset_reward_dict()
        self.rewards_history = pd.DataFrame(columns=list(self._rewards.keys()),
                                            index=[])

    def get_log_info(self) -> dict:
        """
        :returns: total_rewards and rewards for each reward type
        """
        sums = {f"{key}_rewards": val for key, val in self.rewards_history.sum(axis=0).to_dict().items()}
        sums.update({"total_rewards": sum(sums.values())})
        return sums

    def process_rewards(self, trade_risk: float, max_gain: float) -> Dict[str, float]:
        if self._rewards['transaction']:
            if self._rewards['transaction'] > 0:
                self._rewards['transaction'] /= max_gain
            else:
                self._rewards['transaction'] /= trade_risk

    def reset_reward_dict(self):
        self._rewards = {'transaction': 0}
