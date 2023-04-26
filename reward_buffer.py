from pprint import pprint
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from positions import Position


class Reward:
    """Base class for rewards."""

    reward_type = "abstract"

    def __init__(self):
        self.reward = 0

    @property
    def value(self) -> float:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.reward_type}: {self.value}"

    def __float__(self) -> float:
        return float(self.value)


class TransactionReward(Reward):
    """Reward for wallet actions, such as closing positions."""

    reward_type = "TransactionReward"

    def __init__(self, position: Optional[Position] = None):
        super().__init__()
        self.position = position

    @property
    def value(self) -> float:
        if not self.position or not self.position.is_closed:
            return 0.0
        if self.position.is_stop_profit:
            # return 1.5 * self.position.profit
            return self.position.risk_reward_ratio * self.position.profit
        elif self.position.is_stop_loss:
            return self.position.risk_reward_ratio * 0.5 * self.position.profit
        else:
            return self.position.profit


class ActionReward(Reward):
    """Reward for taking proper actions."""

    reward_type = "ActionReward"

    def __init__(self, reward: float):
        super().__init__()
        self._reward = reward

    @property
    def value(self) -> float:
        return float(self._reward)


class IntermediateReward(Reward):
    reward_type = "IntermediateReward"

    def __init__(self, position: Optional[Position] = None, scaling_factor: Optional[float] = None):
        super().__init__()
        self.unrealized_profit = 0.0 if position is None else position.unrealized_profit
        self.scaling_factor = 0.001 if scaling_factor is None else scaling_factor

    @property
    def value(self) -> float:
        return round(self.unrealized_profit * self.scaling_factor, 5)


class SharpeReward(Reward):
    # Rewarding after position closing
    reward_type = "SharpeReward"

    """
    # Assuming you have a list of position returns
    position_returns = [...]

    # Initialize the SharpeReward class
    sharpe_reward = SharpeReward(position_returns)
    """

    def __init__(self, returns: list, risk_free_rate: float = 0.0):
        super().__init__()
        self.returns = returns
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(self, returns: list, risk_free_rate: float) -> float:
        excess_returns = np.array(returns) - risk_free_rate
        avg_excess_return = np.mean(excess_returns)
        excess_return_std = np.std(excess_returns)

        if excess_return_std == 0:
            return 0.0

        sharpe_ratio = avg_excess_return / excess_return_std
        return sharpe_ratio

    @property
    def value(self) -> float:
        sharpe_ratio = self.calculate_sharpe_ratio(self.returns, self.risk_free_rate)
        return sharpe_ratio


class DrawdownReward(Reward):
    reward_type = "DrawdownReward"
    """
    # Assuming you have a list of position returns
    position_returns = [...]

    # Initialize the DrawdownReward class
    drawdown_reward = DrawdownReward(position_returns)
    """

    def __init__(self, returns: list):
        super().__init__()
        self.returns = returns

    def calculate_max_drawdown(self, returns: list) -> float:
        cumulative_returns = np.cumsum(returns)
        cumulative_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_max - cumulative_returns) / cumulative_max
        max_drawdown = np.max(drawdowns)
        return max_drawdown

    @property
    def value(self) -> float:
        max_drawdown = self.calculate_max_drawdown(self.returns)
        drawdown_reward = -max_drawdown  # Invert the max_drawdown to get the reward value
        return drawdown_reward


class RewardBuffer:
    possible_rewards = [TransactionReward, ActionReward,
                        IntermediateReward, SharpeReward,
                        DrawdownReward]

    def __init__(self):
        self.previous_history: dict = None
        self.current_rewards: List[Reward] = []
        self.rewards_history: pd.DataFrame = pd.DataFrame(
            columns=[reward_class.reward_type for reward_class in self.possible_rewards], index=[])
        self.reset()

    def reward_transaction(self, position: Position):
        self.current_rewards.append(TransactionReward(position=position))

    def reward_action(self, reward):
        self.current_rewards.append(ActionReward(reward=reward))

    def reward_intermediate(self, position: Position):
        self.current_rewards.append(IntermediateReward(position=position))

    def reward_sharpe(self, returns):
        self.current_rewards.append(SharpeReward(returns=returns))

    def reward_drawdown(self, returns):
        self.current_rewards.append(DrawdownReward(returns=returns))

    def yield_rewards(self):
        self.update_history()
        total_reward = sum(map(float, self.current_rewards))
        self.reset()
        return total_reward

    def update_history(self):
        update_dict = {reward_class.reward_type: reward_class.value for reward_class in self.current_rewards}
        self.rewards_history = pd.concat([self.rewards_history, pd.DataFrame(update_dict, index=[0])],
                                         axis=0, ignore_index=True)

    @property
    def state(self):
        return list(map(float, self.current_rewards))

    def info(self):
        current_history = self.rewards_history.sum(axis=0)
        if self.previous_history is None:
            self.previous_history = current_history
        delta = {key: round(current_history[key] - self.previous_history[key], 3) for key, value in
                 current_history.items()}
        out = {key: f"{current_history[key]} - Δ:{delta[key]}" for key, value in self.previous_history.items()}
        self.previous_history = current_history
        print(out)

    def get_log_info(self):
        return {str(k): float(v) for k, v in self.rewards_history.sum(axis=0).items()}

    def reset(self, history=False):
        if history:
            self.previous_history = None
            self.rewards_history: pd.DataFrame = pd.DataFrame(
                columns=[reward_class.reward_type for reward_class in self.possible_rewards], index=[])
        self.current_rewards = []

    def __repr__(self):
        return f"<RewardBuffer: Rewards={[str(reward) for reward in self.current_rewards]}>"


class ConsistencyReward(Reward):
    reward_type = "ConsistencyReward"

    """ 
    Calculate the number of winning trades and total trades
    winning_trades = sum(1 for return_ in position_returns if return_ > 0)
    total_trades = len(position_returns)

    Initialize the ConsistencyReward class
    consistency_reward = ConsistencyReward(winning_trades, total_trades)
    """

    def __init__(self, winning_trades: int, total_trades: int):
        super().__init__()
        self.winning_trades = winning_trades
        self.total_trades = total_trades

    @property
    def value(self) -> float:
        if self.total_trades == 0:
            return 0.0
        consistency_score = self.winning_trades / self.total_trades
        return consistency_score


class ProfitFactorReward(Reward):
    reward_type = "ProfitFactorReward"

    """
    # Assuming you have a list of position returns
    position_returns = [...]

    # Calculate the gross profit and gross loss
    gross_profit = sum(return_ for return_ in position_returns if return_ > 0)
    gross_loss = abs(sum(return_ for return_ in position_returns if return_ < 0))

    # Initialize the ProfitFactorReward class
    profit_factor_reward = ProfitFactorReward(gross_profit, gross_loss)
    """

    def __init__(self, gross_profit: float, gross_loss: float):
        super().__init__()
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss

    @property
    def value(self) -> float:
        if self.gross_loss == 0:
            return float('inf')
        profit_factor = self.gross_profit / self.gross_loss
        return profit_factor
