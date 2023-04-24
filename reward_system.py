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
            return self.position.risk_reward_ratio * self.position.profit
        elif self.position.is_stop_loss:
            return self.position.profit
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


class TrendReward(Reward):
    # This should be incorporated into position directly.
    # The unrealized profit from the
    # position unrealized profit history vs. close history throughout period X

    reward_type = "TrendReward"

    """
    # Assuming you have a list of position returns and market returns
    position_returns = [...]
    market_returns = [...]

    # Initialize the TrendReward class
    trend_reward = TrendReward(position_returns, market_returns)
    """

    def __init__(self, position_returns: list, market_returns: list, rolling_average=1):
        super().__init__()
        self.rolling = rolling_average
        self.position_returns = self.rolling_average(position_returns) if rolling_average > 1 else position_returns
        self.market_returns = self.rolling_average(market_returns) if rolling_average > 1 else market_returns

    def rolling_average(self, arr):
        kernel = np.ones(self.rolling) / self.rolling
        return np.convolve(arr, kernel, mode='valid')

    @property
    def value(self) -> float:
        trend_correlation, _ = pearsonr(self.position_returns, self.market_returns)
        return trend_correlation


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


possible_rewards = [TransactionReward, ActionReward, IntermediateReward,
                    SharpeReward, DrawdownReward, TrendReward]


class RewardBuffer:
    possible_rewards = [TransactionReward, ActionReward,
                        IntermediateReward, SharpeReward,
                        DrawdownReward, TrendReward]

    def __init__(self):
        self.rewards: List[Reward] = []
        self.rewards_history: pd.DataFrame = pd.DataFrame(
            columns=[reward_class.reward_type for reward_class in possible_rewards], index=[])
        self.reset()

    def reward_transaction(self, position: Position):
        self.rewards.append(TransactionReward(position=position))

    def reward_action(self, reward):
        self.rewards.append(ActionReward(reward=reward))

    def reward_intermediate(self, position: Position):
        self.rewards.append(IntermediateReward(position=position))

    def reward_trend(self, position: Position, rolling_average):
        self.rewards.append(TrendReward(position_returns=position.profit_history,
                                        market_returns=position.market_history,
                                        rolling_average=rolling_average))

    def reward_sharpe(self, returns):
        self.rewards.append(SharpeReward(returns=returns))

    def reward_drawdown(self, returns):
        self.rewards.append(DrawdownReward(returns=returns))

    def yield_rewards(self):
        self.update_history()
        total_reward = sum(map(float, self.rewards))
        self.reset()
        return total_reward

    def update_history(self):
        update_dict = {reward_class.reward_type: reward_class.value for reward_class in self.rewards}
        self.rewards_history = pd.concat([self.rewards_history, pd.DataFrame(update_dict, index=[0])],
                                         axis=0, ignore_index=True)

    @property
    def state(self):
        return self.__dict__['rewards']

    def info(self, last_rows: int = 100):
        temp = self.rewards_history.sum(axis=0)
        pprint(temp.to_dict())

    def reset(self):
        self.rewards = []

    def __repr__(self):
        return f"<RewardBuffer: Rewards={[str(reward) for reward in self.rewards]}>"


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
