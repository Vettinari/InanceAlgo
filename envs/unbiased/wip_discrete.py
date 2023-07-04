from datetime import timedelta
import gym
import numpy as np
import pandas as pd
from typing import Dict, Optional

from DataProcessing.datastream import DataStream
from Archive.positions.discrete import DiscretePosition
from Utils.xtb import XTB


class DiscretePosition:

    def __init__(self,
                 ticker: str,
                 scaler: float,
                 stop_loss_pips: int,
                 stop_profit_pips: int,
                 risk: int,
                 manual_close: bool = False):
        self.scaler: float = scaler
        self.ticker: str = ticker
        self._stop_loss_pips: int = stop_loss_pips
        self._stop_profit_pips: int = stop_profit_pips
        self.manual_close: bool = manual_close
        self.risk: float = risk

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value: int = XTB[ticker]['one_lot_value']

        # Position dynamic values
        self.stop_loss: float = None
        self.stop_profit: float = None
        self.open_price: float = None
        self.position_type: str = None
        self.position_margin: float = None
        self.volume: float = None
        self.contract_value: float = None
        self.status: str = None
        self.trade_profit: float = None
        self.pips_profit: int = None

    def calculate_pip_profit(self, current_price: float) -> int:
        """Returns the profit in pips given a current_price."""
        out = (current_price - self.open_price) / self.one_pip if self.position_type == 'long' else \
            (self.open_price - current_price) / self.one_pip

        return int(out)

    def calculate_trade_profit(self, current_price: float) -> float:
        """Returns the profit in currency given a current_price."""
        current_value = current_price * self.volume * self.one_lot_value
        acquire_value = self.open_price * self.volume * self.one_lot_value
        return current_value - acquire_value if self.position_type == 'long' else acquire_value - current_value

    def check_stops(self, ohlc_dict: Dict[str, float]) -> Optional[dict]:
        """Returns True and position history(dict) if the current price hits any stop."""

        def _check_stop(stop, comparison):
            return any(comparison(value, stop) for value in ohlc_dict.values())

        # Check stop loss
        hit_stop_loss = _check_stop(self.stop_loss, lambda x, y: x <= y if self.position_type == 'long' else x >= y)
        if hit_stop_loss:
            print("stop_loss")
            self.status = 'stop_loss'
            self.pips_profit = self._stop_loss_pips
            self.trade_profit = self.risk
            history = self.close_position()
            return history

        # Check stop profit
        hit_stop_profit = _check_stop(self.stop_profit, lambda x, y: x >= y if self.position_type == 'long' else x <= y)
        if hit_stop_profit:
            print("stop_profit")
            self.status = 'stop_profit'
            self.pips_profit = self._stop_profit_pips
            self.trade_profit = self.risk * (self._stop_profit_pips / self._stop_loss_pips)
            history = self.close_position()
            return history

        return None

    def close_position(self, manual_close_price: Optional[bool] = None) -> Dict[str, float]:
        if manual_close_price:
            self.status = "manual_close"
            self.pips_profit = self.calculate_pip_profit(current_price=manual_close_price)
            self.trade_profit = self.calculate_trade_profit(current_price=manual_close_price)

        out = self.__dict__().copy()
        self.reset_position()
        return out

    def open_position(self, open_price: float, position_type: str) -> float:
        """
        Opens a position and returns margin that need to be reserved. Calculation of margin is based on the risk.
        :param open_price: Current price
        :param position_type: 'short' or 'long'
        :return: Returns margin that need to be reserved.
        """
        self.status = 'open'
        self.open_price = open_price
        self.position_type = position_type
        self._set_stop_loss()
        self._set_stop_profit()
        self.position_margin = self.required_margin(current_price=open_price)
        return self.position_margin

    def required_margin(self, current_price: float) -> float:
        """
        Calculates the required margin for a position given a current_price.
        :param current_price: Current price
        :return: Returns the required margin.
        """
        stop_loss = current_price - self._stop_loss_pips * self.one_pip if self.position_type == 'long' else \
            current_price + self._stop_loss_pips * self.one_pip
        self.volume = round(
            (self.risk / ((current_price * self.one_lot_value) - (stop_loss * self.one_lot_value))) * current_price, 2)
        self.contract_value = round(self.volume * self.one_lot_value, 2)
        self.position_margin = round(self.contract_value / self.leverage, 2)
        return self.position_margin

    def info(self):
        pprint(self.__dict__())

    def _set_stop_loss(self):
        self.stop_loss = self.open_price - (
                self._stop_loss_pips * self.one_pip) if self.position_type == 'long' else self.open_price + (
                self._stop_loss_pips * self.one_pip)

    def _set_stop_profit(self):
        self.stop_profit = self.open_price + (
                self._stop_profit_pips * self.one_pip) if self.position_type == 'long' else self.open_price - (
                self._stop_profit_pips * self.one_pip)

    def reset_position(self):
        self.stop_loss: float = None
        self.stop_profit: float = None
        self.open_price: float = None
        self.position_type: str = None
        self.position_margin: float = None
        self.volume: float = None
        self.contract_value: float = None
        self.status: str = None
        self.trade_profit: float = None
        self.pips_profit: int = None

    def state(self, current_price: float) -> Optional[List[float]]:
        """
        Returns the state of the position given a current_price. If manual_close is True, it will return the profit
        and other metrics for agent to learn. Otherwise, it will return None as the environment will handle the closing.
        :param current_price:
        :return:
        """
        if self.manual_close:
            return [
                self.calculate_pip_profit(current_price=current_price)
            ]
        else:
            return

    def __dict__(self) -> Dict[str, float]:
        return {
            'ticker': self.ticker,
            'open_price': self.open_price,
            'position_type': self.position_type,
            'stop_loss': self.stop_loss,
            'stop_profit': self.stop_profit,
            'margin': self.position_margin,
            'volume': self.volume,
            'contract_value': self.contract_value,
            'pip_profit': self.pips_profit,
            'trade_profit': self.trade_profit,
            'status': self.status
        }


class DiscreteTradingEnv(gym.Env):
    def __init__(self,
                 datastream: DataStream,
                 test: bool,
                 initial_balance=100000,
                 scale: bool = True,
                 stop_loss_pips: int = 80,
                 stop_profit_pips: int = 20,
                 risk: float = 0.02):
        super().__init__()
        self.scaler = 0.0000001
        self.bad_action_penalty = -0.1
        self.scale = scale
        self.test: bool = test
        self.datastream: DataStream = datastream
        self.initial_balance: float = initial_balance
        self.current_action: float = None
        self.leverage: int = XTB['EURUSD']['leverage'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'leverage']
        self.spread: float = XTB['EURUSD']['spread'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'spread']
        self.pip_value: float = XTB['EURUSD']['one_pip'] if self.datastream.ticker == 'TEST' else \
            XTB[self.datastream.ticker]['one_pip']
        self.history: pd.DataFrame = pd.DataFrame(columns=DiscretePosition.log_info, index=[])

        # Current data
        self.done = False
        self.current_step: int = 0
        self.current_date: pd.Timestamp = self.datastream.generator.start_cursor
        self.current_state: np.array = np.array([])
        self.current_price: float = 0
        self.current_ohlc: Dict[str, float] = dict()
        self.current_extremes_data: pd.DataFrame = None

        self.balance: float = self.initial_balance
        self.stop_loss_pips: int = stop_loss_pips
        self.stop_profit_pips: int = stop_profit_pips
        self.risk: int = risk
        self.position: DiscretePosition = DiscretePosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                                           scaler=self.scaler,
                                                           stop_loss_pips=self.stop_loss_pips,
                                                           stop_profit_pips=self.stop_profit_pips,
                                                           risk=self.initial_balance * self.risk)
        self.reward = 0
        self.reset()

        # Define the observation space
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self.current_state.shape,
                                                dtype=np.float32)
        # Define the action space
        self.action_dict = {i: action for i, action in enumerate(['long', 'short', 'hold'])}
        self.action_space = gym.spaces.Box(low=0, high=len(self.action_dict.values()), shape=(1,), dtype=float)

    def reset(self, **kwargs):
        self.done = False
        self.reward = 0
        self.current_step = 0
        self.current_price = 0
        self.current_action = None
        self.current_date = self.datastream.generator.start_cursor
        self.current_extremes_data: pd.DataFrame = None
        self.position = DiscretePosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                         scaler=self.scaler,
                                         stop_loss_pips=self.stop_loss_pips,
                                         stop_profit_pips=self.stop_profit_pips,
                                         risk=self.initial_balance * self.risk)
        self.balance: float = self.initial_balance
        self.history = pd.DataFrame(columns=self.__dict__, index=[])
        # Generate state after restarting
        self.current_state = self.update_state_price_positions()

    @property
    def total_balance(self) -> float:
        """
        Returns the sum of position margin and cash in wallet.
        Returns: Total agent balance float
        """
        return round(self.balance + self.position.position_margin, 2)

    @property
    def cash_in_hand(self) -> float:
        return round(self.balance, 2)

    def update_state_price_positions(self) -> np.array:
        """
        Updates current_price, current_ohlc and generates new state.
        It is important to update the data in datastream generator
        with new date.

        Returns: State in np.array.
        """
        # Get data corresponding to the current_date
        data = self.datastream.generator[self.current_date]
        # split the data into price_data and scaled_price_data
        price_data = data[data.columns[:6]]
        scaled_data = data[data.columns[6:]]
        # Update current price
        self.current_price = price_data.iloc[-1][f'{self.datastream.step_size}_close']
        # Update current price generate current_ohlc
        self.current_ohlc = price_data.iloc[-1].drop([f'{self.datastream.step_size}_datetime',
                                                      f'{self.datastream.step_size}_volume']).to_dict()
        self.current_ohlc = {k.split("_")[-1]: v for k, v in self.current_ohlc.items()}
        # Concatenate to create a stack

        state = np.hstack([
            self.cash_in_hand * self.scaler / 100,
            *self.position.state(current_price=self.current_price),
            *scaled_data.values.flatten()
        ])

        return state

    def calculate_reward(self, previous_balance: float, hindsight_reward: bool = False) -> float:
        balance_reward = round(self.total_balance - previous_balance, 2)
        if hindsight_reward:  # TO IMPLEMENT
            pass
        return balance_reward

    def validate_action(self, action: float) -> bool:
        return False

    def step(self, action: int):
        self.current_action = action
        previous_balance = self.total_balance

        # Apply action masking
        if self.validate_action(action):  # Invalid action, skip the step and return the current state with 0 reward
            self.reward = self.bad_action_penalty
            return self.current_state, self.reward, self.done, {}

        # Take action
        if self.action_dict[action] == 'long':  # Long
            self.open_position(position_type='long')

        elif self.action_dict[action] == 'short':  # Short
            self.open_position(position_type='short')

        elif self.action_dict[action] == 'hold':  # Hold
            pass

        position_history = None
        while position_history is None:
            self.current_step += 1
            self.current_date += timedelta(minutes=self.datastream.step_size)
            self.current_state = self.update_state_price_positions()
            position_history = self.position.check_stops(ohlc_dict=self.current_ohlc)

        self.balance += position_history['position_margin']
        self.balance += position_history['trade_profit']

        # Calculate the reward
        self.reward = self.calculate_reward(previous_balance=previous_balance)

        # Update history
        self.history.loc[self.current_step] = self.__dict__()

        self.done = self.is_done()

        if self.test:
            print("Environment:")
            self.current_info()
            print("Position:")
            self.position.info()

        return self.current_state, self.reward, self.done, {}

    def is_done(self):
        return self.current_step >= self.datastream.length or self.total_balance <= 0

    def open_position(self, position_type: str):
        if position_type == 'long':
            current_price = self.current_price + self.spread
        else:
            current_price = self.current_price - self.spread

        self.balance -= self.position.open_position(open_price=current_price, position_type=position_type)

    def current_info(self):
        print(f"Current price:", self.current_price, "Balance:", self.total_balance, "Reward:", self.reward)

    def render(self, mode='human'):
        pass

    def __dict__(self) -> dict:
        out = {
            'step': self.current_step,
            'action': self.current_action,
            'balance': self.total_balance,
            'reward': self.reward,
        }
        return out.update(self.position.log_info)
