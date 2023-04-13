from typing import Optional

from positions import Position


class Reward:
    """Base class for rewards."""

    reward_type = "abstract"

    def __init__(self):
        self.reward = 0

    def calculate_reward(self):
        pass

    def __str__(self):
        return f"{self.reward_type}: {self.reward}"


class TransactionReward(Reward):
    """Reward for wallet actions, such as closing positions."""

    reward_type = "TransactionReward"

    def __init__(self, position=None):
        super().__init__()
        self.position = position
        self.reward = 0 if position is None else self.calculate_reward()

    def calculate_reward(self):
        if self.position.is_stop_profit and self.position.is_closed:
            return self.position.risk_reward_ratio * self.position.profit
        elif self.position.is_stop_loss and self.position.is_closed:
            return self.position.profit
        elif self.position.is_closed:
            return self.position.profit
        else:
            return 0


class ActionReward(Reward):
    """Reward for taking proper actions."""

    reward_type = "ActionReward"

    def __init__(self, reward):
        super().__init__()
        self.reward = reward


class IntermediateReward(Reward):
    reward_type = "IntermediateReward"

    def __init__(self, position: Optional[Position], scaling_factor: Optional[float] = None):
        super().__init__()
        self.unrealized_profit = 0 if position is None else position.unrealized_profit
        self.scaling_factor = 0.001 if scaling_factor is None else scaling_factor
        self.reward = 0 if position is None else self.calculate_reward()

    def calculate_reward(self):
        return round(self.unrealized_profit * self.scaling_factor, 5)
