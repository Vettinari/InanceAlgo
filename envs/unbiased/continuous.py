from datetime import timedelta
import gym
import numpy as np
import pandas as pd
from DataProcessing.datastream import DataStream
from typing import Dict, List, Optional
from Utils.xtb import XTB


class ContinuousTradingEnv(gym.Env):
    def __init__(self,
                 datastream: DataStream,
                 test: bool,
                 initial_balance=100000,
                 scaler: Optional[float] = None):
        super().__init__()
        self.scaler = 0.0000001
        self.bad_action_penalty = -0.1
        self.scaler = scaler
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

        # Current data
        self.done = False
        self.current_action: float = None
        self.current_step: int = 0
        self.current_date: pd.Timestamp = self.datastream.generator.start_cursor
        self.current_state: np.array = np.array([])
        self.current_price: float = 0
        self.current_ohlc: Dict[str, float] = dict()
        self.current_extremes_data: pd.DataFrame = None
        self.balance: float = self.initial_balance
        self.position: ContinuousPosition = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker)
        self.reward = 0
        self.history_columns = list(self.log_info().keys()) + list(
            self.position.log_info(current_price=self.current_price).keys())
        self.history: pd.DataFrame = pd.DataFrame(columns=self.history_columns, index=[])
        self.reset()

        # Define the observation space
        # print('State size:', self.current_state.shape)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self.current_state.shape,
                                                dtype=np.float32)
        # Define the action space
        self.action_space = gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float)

    def reset(self, **kwargs):
        # Current data
        self.done = False
        self.current_step: int = 0
        self.current_date: pd.Timestamp = self.datastream.generator.start_cursor
        self.current_state: np.array = np.array([])
        self.current_price: float = 0
        self.current_ohlc: Dict[str, float] = dict()
        self.current_extremes_data: pd.DataFrame = None
        self.balance: float = self.initial_balance
        self.position = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker)
        self.history: pd.DataFrame = pd.DataFrame(columns=self.history_columns, index=[])
        self.reward = 0
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

    def modify_position(self, volume: float):
        self.balance += self.position.modify_position(current_price=self.current_price, volume=volume)

    def is_done(self):
        return self.current_step >= self.datastream.length or self.total_balance <= 0

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
            self.cash_in_hand * self.scaler,
            *self.position.state(current_price=self.current_price, scaler=self.scaler),
            *scaled_data.values.flatten()
        ])

        return state

    def calculate_reward(self, previous_balance: float, hindsight_reward: bool = False) -> float:
        balance_reward = round(self.total_balance - previous_balance, 2)
        if hindsight_reward:  # TO IMPLEMENT
            pass
        return balance_reward

    def step(self, action: float):
        self.current_action = action

        # Update history
        env_log = self.log_info()
        env_log.update(self.position.log_info(current_price=self.current_price))
        self.history.loc[self.current_step] = env_log

        # Apply action masking.
        if self.position.validate_action(action=action, cash_in_hand=self.cash_in_hand) is False:
            self.reward = self.bad_action_penalty
            return self.current_state, self.reward, self.done, {}

        previous_balance = self.total_balance

        # Determine current action
        self.modify_position(volume=action)

        # Increase step and date.
        self.current_step += 1
        self.current_date += timedelta(minutes=self.datastream.step_size)
        # Update current state
        self.current_state = self.update_state_price_positions()
        # Calculate the reward
        self.reward = self.calculate_reward(previous_balance=previous_balance)

        self.done = self.is_done()

        return self.current_state, self.reward, self.done, {}

    def info(self):
        print("Step=", self.current_step,
              "\nEnv_cash=", self.cash_in_hand,
              "Env_Balance=", self.total_balance,
              "\nP_profit=", self.position.profit,
              "P_margin=", self.position.position_margin,
              "P_volume=", self.position.total_volume,
              )

    def log_info(self):
        return {
            'step': self.current_step,
            'balance': self.total_balance,
        }

    def render(self, mode='human'):
        pass


class ContinuousPosition:

    def __init__(self,
                 ticker: str):
        self.ticker: str = ticker
        self._order_number: Optional[str] = None

        # Continuous values
        self.profit = 0
        self.position_margin: float = 0
        self.total_volume = 0
        self.avg_price: float = 0
        self.contract_value: float = 0
        self.current_bias = 'neutral'

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value = XTB[ticker]['one_lot_value']

    def modify_position(self, current_price: float, volume: float):
        """
        Modify the current ContinuousPosition.
        """
        delta_volume = round(self.total_volume + volume, 2)

        # Decreasing longs
        if self.total_volume > 0 > volume and delta_volume >= 0:
            # print('Decreasing longs'.upper())
            released_margin_and_profit = self.liquidate(current_price=current_price,
                                                        volume=abs(volume),
                                                        order_type='long')
            return released_margin_and_profit
        # Decreasing shorts
        elif self.total_volume < 0 < volume and delta_volume <= 0:
            # print('Decreasing shorts'.upper())
            released_margin_and_profit = self.liquidate(current_price=current_price,
                                                        volume=volume,
                                                        order_type='short')
            return released_margin_and_profit
        # Liquidating shorts and buying longs
        elif self.total_volume < 0 < delta_volume:
            # print('Liquidating shorts and buying longs'.upper())
            released_margin_and_profit = self.liquidate(current_price=current_price,
                                                        volume=abs(self.total_volume),
                                                        order_type='short')
            required_margin = self.buy(current_price=current_price,
                                       volume=delta_volume,
                                       order_type='long')
            return released_margin_and_profit - required_margin
        # Liquidating longs and buying shorts
        elif self.total_volume > 0 > delta_volume:
            # print('Liquidating longs and buying shorts'.upper())
            released_margin_and_profit = self.liquidate(current_price=current_price,
                                                        volume=self.total_volume,
                                                        order_type='long')
            required_margin = self.buy(current_price=current_price,
                                       volume=abs(delta_volume),
                                       order_type='short')
            return released_margin_and_profit - required_margin
        # Increasing shorts
        elif self.total_volume <= 0 and delta_volume < 0:
            # print('INCREASE SHORTS')
            required_margin = self.buy(current_price=current_price,
                                       volume=abs(volume),
                                       order_type='short')
            return -required_margin
        # Increasing longs
        elif self.total_volume >= 0 and delta_volume > 0:
            # print('INCREASE LONGS')
            required_margin = self.buy(current_price=current_price,
                                       volume=abs(volume),
                                       order_type='long')
            return -required_margin

    def buy(self, current_price: float, volume: float, order_type: str):
        if order_type == 'long':
            total_value = (self.total_volume * self.avg_price) + (volume * current_price)
            self.total_volume = round(self.total_volume + volume, 3)
        else:
            total_value = (abs(self.total_volume) * self.avg_price) + (abs(volume) * current_price)
            self.total_volume = round(self.total_volume - volume, 3)

        self.avg_price = round(total_value / abs(self.total_volume), 5)
        self.contract_value = round(abs(self.total_volume) * self.one_lot_value, 2)
        required_margin = round(self.required_margin(volume=abs(volume)), 2)
        self.position_margin = round(self.position_margin + required_margin, 2)
        # print('Buying', order_type, "with volume", volume, "margin_required", self.required_margin(volume=volume), )
        return required_margin

    def liquidate(self, current_price: float, volume: float, order_type: str):
        margin_released = self.required_margin(volume=volume)
        profit = self.trade_profit(current_price=current_price, volume=volume, order_type=order_type)

        self.total_volume = round(abs(self.total_volume) - volume, 3)
        if self.total_volume > 0:
            self.position_margin = round(self.position_margin - margin_released, 2)
            self.contract_value = round(self.total_volume * self.one_lot_value, 2)
        else:
            self.avg_price = 0
            self.position_margin = 0
            self.contract_value = 0
        # print('Liquidating', order_type, "with volume", volume, "margin released", margin_released, "profit", profit)
        return round(margin_released + profit, 2)  # returns the margin released + profit

    def required_margin(self, volume) -> float:
        """Return the required margin to open a position calculation based on the volume passed.
        Returns:
            float: The required margin to open a position.
        """

        return round((volume * self.one_lot_value) / self.leverage, 2)

    def pip_profit(self, current_price: float) -> float:
        """Return the profit in pips.
        Returns:
            float: The profit in pips.
        """
        if self.avg_price:
            out = (current_price - self.avg_price) / self.one_pip \
                if self.total_volume >= 0 else (self.avg_price - current_price) / self.one_pip
            return round(out, 3)
        else:
            return 0

    def trade_profit(self, current_price: float, volume: float, order_type: str) -> float:
        """Returns real trade profit.
        Returns:
            float: Trade profit in currency.
        """
        cur_val = current_price * volume * self.one_lot_value  # Calculate the full contract value
        open_val = self.avg_price * volume * self.one_lot_value  # Calculate the open contract value
        return round(cur_val - open_val, 3) if order_type == 'long' else round(open_val - cur_val, 3)

    def total_position_profit(self, current_price: float) -> float:
        """Returns real total profit.
        Returns:
            float: Total profit in currency.
        """
        delta = round((current_price - self.avg_price) * self.total_volume * self.one_lot_value, 2)
        return delta if self.total_volume >= 0 else -delta

    def info(self) -> None:
        """
        Print position info.
        """
        print(
            f'INFO: '
            f'Avg_price = {self.avg_price}, '
            f'Volume = {self.total_volume}, '
            f'Value = {self.contract_value}, '
            f'Margin = {self.position_margin}')

    def state(self, current_price: float, scaler: float) -> List[float]:
        """Return position state.
        Returns:
            list: [pip_profit, position_margin, total_volume]
        """
        return [self.pip_profit(current_price=current_price) * scaler,
                self.position_margin * scaler,
                self.total_volume * scaler]

    def log_info(self, current_price: float) -> dict:
        """
        Return position as dictionary.
        Returns:
            dict of all position arguments that are important.
        """
        return {
            "total_position_profit": self.total_position_profit(current_price=current_price),
            "position_margin": self.position_margin,
            "total_volume": self.total_volume,
            "avg_price": self.avg_price,
            "position_type": "long" if self.total_volume >= 0 else 'short'
        }

    def validate_action(self, action: float, cash_in_hand: float) -> bool:
        """
        Validate if the action is possible.
        Returns:
            True if the action is possible, False otherwise.
        """
        enough_cash_flag = cash_in_hand >= round(self.required_margin(volume=abs(action)), 2)
        return enough_cash_flag
