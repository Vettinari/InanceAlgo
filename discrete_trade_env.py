from datetime import timedelta
from pprint import pprint

import gymnasium
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

from DataProcessing.datastream import DataStream
from position import Position

XTB = {"EURUSD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001,
                  "spread": 0.00008},
       "EURCAD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURCHF": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURGBP": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01}}


class DiscreteTradingEnv(gymnasium.Env):
    def __init__(self,
                 datastream: DataStream,
                 stop_loss_pips,
                 stop_profit_pips,
                 test,
                 initial_balance=100000,
                 risk=0.1):
        super().__init__()
        self.current_ohlc: Dict[str, float] = None
        self.test: bool = test
        self.datastream: DataStream = datastream
        self.initial_balance: float = initial_balance
        self.leverage: int = XTB['EURUSD']['leverage'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'leverage']
        self.spread: float = XTB['EURUSD']['spread'] if self.datastream.ticker == 'TEST' else XTB[datastream.ticker][
            'spread']
        self.pip_value: float = XTB['EURUSD']['one_pip'] if self.datastream.ticker == 'TEST' else \
            XTB[datastream.ticker]['one_pip']
        self.history: pd.DataFrame = pd.DataFrame(columns=Position.__dict__, index=[])

        # Basic risk management
        self.risk: float = risk
        self.position_margin: float = self.initial_balance * self.risk
        self.stop_loss_pips = stop_loss_pips
        self.stop_profit_pips = stop_profit_pips

        # Current data
        self.current_step: int = 0
        self.current_date: pd.Timestamp = self.datastream.generator.cursor
        self.current_state: np.array = None
        self.current_price: float = 0
        self.balance: float = self.initial_balance
        self.positions: List[Position] = []
        self.reward = 0
        self.reset()

    def reset(self, **kwargs):
        self.reward = 0
        self.current_step = 0
        self.current_date = self.datastream.generator.cursor
        self.current_price = 0
        self.positions: List[Position] = []
        self.balance: float = self.initial_balance
        self.history = pd.DataFrame(columns=Position.__dict__, index=[])

        self.current_state = self.update_state_price_positions()

        if self.test:
            print(self.__repr__())
            self.current_info()

    @property
    def total_balance(self):
        positions_margin = sum([position.margin for position in self.positions])
        return self.balance + positions_margin

    def positions_update(self):
        for index, position in enumerate(self.positions):
            position.check_stops(ohlc_dict=self.current_ohlc)  # Check if position is terminated
            if position.realized_profit != 0:
                self.close_position(position=position)  # Close the position
                self.history.loc[self.history.shape[0]] = position.__dict__()  # Archive the position
                self.positions.pop(index)  # Remove the position from the list

    def open_position(self, position_type, current_price):
        if position_type == 'long':
            current_price = current_price + self.spread
        elif position_type == 'short':
            current_price = current_price - self.spread

        position = Position(order_type=position_type,
                            ticker=self.datastream.ticker,
                            open_time=self.current_date,
                            open_price=current_price,
                            stop_loss_pips=self.stop_loss_pips,
                            stop_profit_pips=self.stop_profit_pips,
                            leverage=self.leverage,
                            one_pip=self.pip_value,
                            position_margin=self.position_margin)

        # Update balance
        self.balance -= self.position_margin
        self.positions.append(position)

    def close_position(self, position: Position):
        self.balance += position.margin
        self.balance += position.realized_profit

    def calculate_reward(self, previous_balance: int, current_balance: int):
        balance = current_balance - previous_balance
        return balance

    def is_done(self):
        return self.current_step >= self.datastream.length or self.total_balance <= 0

    def update_state_price_positions(self):
        data = self.datastream.generator[self.current_date]
        price_data = data[data.columns[:6]]
        self.current_price = price_data.iloc[-1][f'{self.datastream.step_size}_close']
        self.current_ohlc = price_data.iloc[-1].drop([f'{self.datastream.step_size}_datetime',
                                                      f'{self.datastream.step_size}_volume']).to_dict()
        self.current_ohlc = {k.split("_")[-1]: v for k, v in self.current_ohlc.items()}

        scaled_data = data[data.columns[6:]]
        longs = len([position for position in self.positions if position.order_type == 'long'])
        shorts = len([position for position in self.positions if position.order_type == 'short'])

        self.positions_update()
        positions_margin = sum([position.margin for position in self.positions])

        state = np.hstack([self.balance, positions_margin, longs, shorts, *scaled_data])

        return state

    def step(self, action: int):
        previous_balance = self.total_balance

        # Determine current action
        if action == 0:
            self.open_position('long', self.current_price)

        elif action == 1:
            # Code to execute buy (long)
            self.open_position('short', self.current_price)

        self.current_step += 1
        self.current_date += timedelta(minutes=self.datastream.step_size)
        self.update_state_price_positions()

        current_balance = self.total_balance
        self.reward = self.calculate_reward(previous_balance=previous_balance,
                                            current_balance=current_balance)

        # Update history
        self.history.loc[self.current_step] = {
            'step': self.current_step,
            'balance': current_balance,
            'action': action,
            'reward': self.reward
        }

        self.done = self.is_done()

        # Generate state and save it as current state
        self.current_state = self.update_state_price_positions()

        if self.test:
            print()
            print(self.__repr__())
            self.current_info()

        return self.current_state, self.reward, self.done, {}

    def current_info(self):
        print(f"Current time: {self.current_date}")
        print(f"Current ohlc:")
        pprint(self.current_ohlc)
        print(f"Current price:", self.current_price)
        print(f"Current reward:", self.reward)

    def __repr__(self):
        return f"{self.__class__.__name__}: Balance={self.balance} " \
               f"Longs={len([position for position in self.positions if position.order_type == 'long'])} " \
               f"Shorts={len([position for position in self.positions if position.order_type == 'short'])}"
