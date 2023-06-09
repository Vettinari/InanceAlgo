from datetime import timedelta
import gym
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from DataProcessing.datastream import DataStream
from Utils.xtb import XTB


class ContinuousTradingEnv(gym.Env):
    def __init__(self,
                 datastream: DataStream,
                 test: bool,
                 agent_bias: str,
                 initial_balance=100000,
                 scaler: Optional[float] = None):
        super().__init__()
        self.test: bool = test
        self.scaler = scaler
        self.bad_action_penalty = -0.1
        self.datastream: DataStream = datastream
        self.initial_balance: float = initial_balance
        self.agent_bias: str = agent_bias

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
        self.position: ContinuousPosition = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                                               order_type=self.agent_bias)
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
        self.position: ContinuousPosition = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                                               order_type=self.agent_bias)
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
        if self.agent_bias == 'long':
            current_price = self.current_price + self.spread
        else:
            current_price = self.current_price - self.spread

        self.balance += self.position.modify_position(current_price=current_price, volume=volume)

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
        log = self.log_info()
        log.update(self.position.log_info(current_price=self.current_price))
        self.history.loc[self.current_step] = log

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
              "Cash_in_hand=", self.cash_in_hand,
              "Balance=", self.total_balance,
              "Current_ohlc=", self.current_ohlc,
              )

    def log_info(self) -> dict:
        return {
            'step': self.current_step,
            'balance': self.total_balance,
        }

    def render(self, mode='human'):
        pass


class ContinuousPosition:

    def __init__(self,
                 order_type: str,
                 ticker: str):
        self.ticker: str = ticker
        self.order_type: str = order_type.lower()
        self._order_number: Optional[str] = None

        # Continuous values
        self.profit = 0
        self.position_margin: float = 0
        self.total_volume = 0
        self.avg_price: float = 0
        self.contract_value: float = 0

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value = XTB[ticker]['one_lot_value']

    def modify_position(self, current_price: float, volume: float) -> float:
        """Modify the current ContinuousPosition.

        Given a current_price and volume position changes.
        If the volume is positive agent buys lots if negative he sells them.

        Parameters:
            current_price (float): Current price at selected step_size timeframe.
            volume (float): Amount of the lots the agent wants to acquire/sell.

        Returns:
            float: The amount to update the balance of the ContinuousEnvironment
        """
        volume = round(volume, 2)  # Round down to 2 decimal places
        if volume > 0:  # Agent is increasing his biased position
            total_value = (self.total_volume * self.avg_price) + (volume * current_price)
            self.total_volume = round(self.total_volume + volume, 3)
            self.avg_price = round(total_value / self.total_volume, 5)

            self.contract_value = round(self.total_volume * self.one_lot_value, 2)

            margin = self.required_margin(volume=volume)
            self.position_margin = round(self.position_margin + margin, 2)
            return -round(margin, 2)  # returns the margin required to open the position

        elif volume < 0:  # Agent is decreasing his biased position
            if volume < -self.total_volume:  # If agent wants to sell more than he has
                volume = -self.total_volume

            profit = self.trade_profit(current_price=current_price, volume=abs(volume))
            margin_released = round((abs(volume) * self.one_lot_value) / self.leverage, 2)

            self.total_volume = round(self.total_volume + volume, 3)

            # If all shares are sold, average acquisition price becomes 0
            if self.total_volume > 0:
                self.position_margin = round(self.position_margin - margin_released, 2)
                self.contract_value = round(self.total_volume * self.one_lot_value, 2)
            else:
                self.avg_price = 0
                self.position_margin = 0
                self.contract_value = 0

            return round(margin_released + profit, 2)  # returns the margin released + profit
        # Return 0 as no changes were made
        return 0

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
                if self.order_type == 'long' else (self.avg_price - current_price) / self.one_pip
            return round(out, 3)
        else:
            return 0

    def trade_profit(self, current_price: float, volume: float) -> float:
        """Returns real trade profit.
        Returns:
            float: Trade profit in currency.
        """
        cur_val = current_price * volume * self.one_lot_value  # Calculate the full contract value
        open_val = self.avg_price * volume * self.one_lot_value  # Calculate the open contract value
        return round(cur_val - open_val, 2) if self.order_type == 'long' else round(open_val - cur_val, 2)

    def total_position_profit(self, current_price: float) -> float:
        """Returns real total profit.
        Returns:
            float: Total profit in currency.
        """
        delta = round((current_price - self.avg_price) * self.total_volume * self.one_lot_value, 2)
        return delta if self.order_type == 'long' else -delta

    def info(self) -> None:
        """
        Print position info.
        """
        print(f'INFO: '
              f'Order_type = {self.order_type.capitalize()}: '
              f'Avg_price = {self.avg_price} '
              f'Volume = {self.total_volume} '
              f'Value = {self.contract_value} '
              f'Margin = {self.position_margin}\n')

    def state(self, current_price: float, scaler: float) -> List[float]:
        """Return position state.
        Returns:
            list: [profit, position_margin, total_volume]
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
            "position_type": self.order_type
        }

    def validate_action(self, action: float, cash_in_hand: float) -> bool:
        """
        Validate if the action is possible.
        Returns:
            True if the action is possible, False otherwise.
        """
        # True if agent wants to sell more than he has
        cannot_sell_flag = action < 0 and (self.total_volume == 0 or abs(action) > self.total_volume)
        # True if agent wants to buy without enough cash
        buy_without_enough_cash_flag = cash_in_hand < self.required_margin(volume=action)
        return not (cannot_sell_flag or buy_without_enough_cash_flag)
