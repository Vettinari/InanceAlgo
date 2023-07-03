from datetime import timedelta
import gym
import numpy as np
import pandas as pd
from typing import Dict

from DataProcessing.datastream import DataStream
from positions.continuous import ContinuousPosition
from xtb import XTB


class ContinuousTradingEnv(gym.Env):
    def __init__(self,
                 datastream: DataStream,
                 test: bool,
                 agent_bias: str,
                 initial_balance=100000,
                 scale: bool = True):
        super().__init__()
        self.scaler = 0.0000001
        self.bad_action_penalty = -0.1
        self.scale = scale
        self.test: bool = test
        self.datastream: DataStream = datastream
        self.initial_balance: float = initial_balance
        self.agent_bias: str = agent_bias
        self.current_action: float = None
        self.leverage: int = XTB['EURUSD']['leverage'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'leverage']
        self.spread: float = XTB['EURUSD']['spread'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'spread']
        self.pip_value: float = XTB['EURUSD']['one_pip'] if self.datastream.ticker == 'TEST' else \
            XTB[self.datastream.ticker]['one_pip']
        self.history: pd.DataFrame = pd.DataFrame(columns=ContinuousPosition.__dict__, index=[])

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
                                                               scaler=self.scaler,
                                                               order_type=self.agent_bias)
        self.reward = 0
        self.reset()

        # Define the observation space
        # print('State size:', self.current_state.shape)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self.current_state.shape,
                                                dtype=np.float32)
        # Define the action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)

    def reset(self, **kwargs):
        self.done = False
        self.reward = 0
        self.current_step = 0
        self.current_price = 0
        self.current_action = None
        self.current_date = self.datastream.generator.start_cursor
        self.current_extremes_data: pd.DataFrame = None
        self.position = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                           order_type=self.agent_bias,
                                           scaler=self.scaler)
        self.balance: float = self.initial_balance
        self.history = pd.DataFrame(columns=ContinuousPosition.__dict__, index=[])
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
            self.cash_in_hand * self.scaler / 100,
            *self.position.state(current_price=self.current_price, scaled=True),
            *scaled_data.values.flatten()
        ])

        return state

    def calculate_reward(self, previous_balance: float, hindsight_reward: bool = False) -> float:
        balance_reward = round(self.total_balance - previous_balance, 2)
        if hindsight_reward:  # TO IMPLEMENT
            pass
        return balance_reward

    def validate_action(self, action: float) -> bool:
        sell_zero_flag = action < 0 and self.position.total_volume == 0
        buy_without_enough_cash_flag = self.cash_in_hand < self.position.required_margin(volume=action)
        sell_more_than_own_flag = action < 0 and abs(action) > self.position.total_volume
        return sell_zero_flag or buy_without_enough_cash_flag or sell_more_than_own_flag

    def step(self, action: float):
        self.current_action = action

        previous_balance = self.total_balance

        # Apply action masking
        if self.validate_action(action):  # Invalid action, skip the step and return the current state with 0 reward
            self.reward = self.bad_action_penalty
            return self.current_state, self.reward, self.done, {}

        # Determine current action
        self.modify_position(volume=action)
        # Increase step and date.
        self.current_step += 1
        self.current_date += timedelta(minutes=self.datastream.step_size)
        # Update current state
        self.current_state = self.update_state_price_positions()
        # Calculate the reward
        self.reward = self.calculate_reward(previous_balance=previous_balance)

        # Update history
        self.history.loc[self.current_step] = self.__dict__

        self.done = self.is_done()

        if self.test:
            print("Environment:")
            self.current_info()
            print("Position:")
            self.position.info()

        return self.current_state, self.reward, self.done, {}

    def current_info(self):
        print(f"Current price:", self.current_price,
              "Cash_in_hand:", self.cash_in_hand,
              "Balance:", self.total_balance,
              "Reward:", self.reward)

    def render(self, mode='human'):
        pass

    def __dict__(self) -> dict:
        out = {
            'step': self.current_step,
            'action': self.current_action,
            'balance': self.total_balance,
            'reward': self.reward,
        }
        return out.update(self.position.__dict__)
