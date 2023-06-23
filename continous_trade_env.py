from datetime import timedelta
import gymnasium
import numpy as np
import pandas as pd
from typing import Dict
from DataProcessing.datastream import DataStream
from position import ContinuousPosition
from xtb import XTB


class ContinuousTradingEnv(gymnasium.Env):
    def __init__(self,
                 datastream: DataStream,
                 test: bool,
                 agent_bias: str,
                 initial_balance=100000):
        super().__init__()
        self.test: bool = test
        self.datastream: DataStream = datastream
        self.initial_balance: float = initial_balance
        self.agent_bias: str = agent_bias
        self.leverage: int = XTB['EURUSD']['leverage'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'leverage']
        self.spread: float = XTB['EURUSD']['spread'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'spread']
        self.pip_value: float = XTB['EURUSD']['one_pip'] if self.datastream.ticker == 'TEST' else \
            XTB[self.datastream.ticker]['one_pip']
        self.history: pd.DataFrame = pd.DataFrame(columns=ContinuousPosition.__dict__, index=[])

        # Current data
        self.current_step: int = 0
        self.current_date: pd.Timestamp = self.datastream.generator.start_cursor
        self.current_state: np.array = None
        self.current_price: float = 0
        self.current_ohlc: Dict[str, float] = None
        self.done = False

        self.balance: float = self.initial_balance
        self.position: ContinuousPosition = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                                               order_type=self.agent_bias)
        self.reward = 0
        self.reset()

    def reset(self, **kwargs):
        self.done = False
        self.reward = 0
        self.current_step = 0
        self.current_date = self.datastream.generator.start_cursor
        self.current_price = 0
        self.position: ContinuousPosition = ContinuousPosition(ticker='EURUSD' if self.test else self.datastream.ticker,
                                                               order_type=self.agent_bias)
        self.balance: float = self.initial_balance
        self.history = pd.DataFrame(columns=ContinuousPosition.__dict__, index=[])
        # Generate state after restarting
        self.current_state = self.update_state_price_positions()

    @property
    def total_balance(self):
        return round(self.balance + self.position.position_margin, 2)

    @property
    def position_profit(self):
        return self.position.calculate_profit(current_price=self.current_price)

    def modify_position(self, volume: float):
        if self.agent_bias == 'long':
            current_price = self.current_price + self.spread
        else:
            current_price = self.current_price - self.spread

        self.balance += self.position.modify_position(current_price=current_price, volume=volume)

    def is_done(self):
        return self.current_step >= self.datastream.length or self.total_balance <= 0

    def update_state_price_positions(self) -> np.array():
        """
        Updates current_price, current_ohlc and generates new state.
        It is important to update the data in datastream generator
        with new date.

        Returns: State in np.array.
        """

        data = self.datastream.generator[self.current_date]
        price_data = data[data.columns[:6]]
        self.current_price = price_data.iloc[-1][f'{self.datastream.step_size}_close']
        self.current_ohlc = price_data.iloc[-1].drop([f'{self.datastream.step_size}_datetime',
                                                      f'{self.datastream.step_size}_volume']).to_dict()
        self.current_ohlc = {k.split("_")[-1]: v for k, v in self.current_ohlc.items()}

        scaled_data = data[data.columns[6:]]

        state = np.hstack([self.total_balance, self.position.state, *scaled_data.values])

        return state

    def step(self, action: float):
        previous_balance = self.total_balance

        # Determine current action
        self.modify_position(volume=action)

        self.current_step += 1
        self.current_date += timedelta(minutes=self.datastream.step_size)
        self.current_state = self.update_state_price_positions()

        self.reward = round(self.total_balance - previous_balance, 2)

        # Update history
        self.history.loc[self.current_step] = {
            'step': self.current_step,
            'balance': self.total_balance,
            'action': action,
            'reward': self.reward
        }

        self.done = self.is_done()

        if self.test:
            print("Environment:")
            self.current_info()
            print("Position:")
            self.position.info()

        return self.current_state, self.reward, self.done, {}

    def current_info(self):
        # print(f"Time: {self.current_date}")
        print(f"Current price:", self.current_price)
        print(f"State:", self.current_state[:2])
        print(f"Reward:", self.reward)
