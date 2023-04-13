from positions import Position


class Reward:
    """Base class for rewards."""

    reward_type = "abstract"

    def __init__(self):
        self._reward = 0

    @property
    def reward(self):
        return self._reward

    def calculate_reward(self):
        pass

    def __str__(self):
        return f"{self.reward_type}: {self.reward}"


class WalletReward(Reward):
    """Reward for wallet actions, such as closing positions."""

    reward_type = "WalletReward"

    def __init__(self, position: Position = None):
        super().__init__()
        self.position = position
        self._reward = 0 if position is None else self.calculate_reward()

    def calculate_reward(self):
        if self.position.is_stop_profit:
            return self.position.risk_reward_ratio * self.position.profit
        elif self.position.is_stop_loss:
            return self.position.profit


class ActionReward(Reward):
    """Reward for taking proper actions."""

    reward_type = "ActionReward"

    def __init__(self, reward):
        super().__init__()
        self._reward = reward
