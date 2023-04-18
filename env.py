from abc import ABC
from typing import Tuple, List, Dict

import gym
import numpy as np
from gym.core import ObsType

from DataProcessing.data_pipeline import DataPipeline
from DataProcessing.timeframe import OHLCT, TimeFrame
from risk_manager import RiskManager


class TradeGym(gym.Env, ABC):
    def __init__(self,
                 risk_manager: RiskManager,
                 data_pipeline: DataPipeline,
                 reward_scaling: float):
        self.state = None
        self.wallet_state = None
        self.data_state = None
        self.tech_state = None
        self.chart_state = None
        self.timeframes: Dict[int, TimeFrame] = None

        self.done = False
        self.current_step = 0
        self.current_ohlct: OHLCT = None
        self.current_date = None

        # Rewards
        self.reward_scaling = reward_scaling
        self.reward = 0

        # Objects
        self.risk_manager: RiskManager = risk_manager
        self.data_pipeline: DataPipeline = data_pipeline

        # TODO
        self.rewards_history = []
        self.continuous = False
        self.max_steps = -999
        self.reset()

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        self.done = self.current_step >= (self.max_steps - 1) or self.risk_manager.wallet.game_over  # OK

        if self.done:
            return self.state, self.reward, self.done, {}

        else:
            # if self.current_step % 1000 == 0:
            print(f"Env:\n"
                  f"reward: {self.reward}\n"
                  f"{self.risk_manager.wallet}")

            self.rewards_history.append(self.reward)

            start_reward = self.reward
            self.risk_manager.execute_action(action_index=int(action),
                                             current_atr=self.get_atr())
            # state: S -> S+1
            self.current_step += 1
            self.data_pipeline.step_forward()
            # Update wallet and environment elements for state generation
            self.update_wallet_and_env()
            # Calculate transition reward
            end_reward = self.risk_manager.current_rewards
            self.reward = (end_reward - start_reward) * self.reward_scaling

            return self.state, self.reward, self.done, {}

    def get_atr(self, interval=15):
        return float(self.timeframes[interval].tech()['atrr_14'])

    def update_wallet_and_env(self):
        self.current_ohlct = self.data_pipeline.ohlct
        self.current_date = self.data_pipeline.current_date
        self.risk_manager.wallet_step(ohlct=self.current_ohlct)
        self.update_states()
        self.generate_state()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]:
        self.done = False
        self.reward = 0
        self.current_step = 0
        self.data_pipeline.reset_step()
        self.current_ohlct = self.data_pipeline.ohlct
        self.current_date = self.data_pipeline.current_date
        self.risk_manager.wallet.reset(ohlct=self.current_ohlct)
        self.update_states()
        self.generate_state()
        self.max_steps = self.data_pipeline.dataframe.shape[0] - self.data_pipeline.current_step
        return self.state

    def update_states(self):
        self.wallet_state = self.risk_manager.wallet.state
        self.process_pipeline_timeframes()

    def process_pipeline_timeframes(self):
        data = []
        tech = []
        self.timeframes = self.data_pipeline.current_data()
        for interval, timeframe in self.timeframes.items():
            data.append(timeframe.data().values)
            tech.append(timeframe.tech().values)

        self.data_state = data
        self.tech_state = tech

    def generate_state(self):
        self.state = [*self.wallet_state, *self.data_state, *self.tech_state]

    def __repr__(self):
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_pipeline={self.data_pipeline}\n' \
               f'OHLCT={self.current_ohlct}\n>'

# class XTBTradingEnv(gym.Env):
#     """
#     A standard stock trading environment for OpenAI gym.
#     Allows to trade one ticker and open long or short position for this ticker.
#     Positions can be either interchangeable or simultaneous - depending on what wallet you will pick
#     to describe trading strategy.
#     """
#
#     metadata = {"render.modes": ["human"]}
#
#     def __init__(
#             self,
#             ticker,
#             wallet: XTBWallet,
#             step_size: int,
#             data_dict: dict,
#             example_data: dict,
#             reward_scaling: float,
#             strategy_trading: bool = False,
#             mode='train',
#             test=False
#     ):
#         self.name = ticker
#         self.mode = mode
#         self.test = test
#         self.strategy_trading = strategy_trading
#
#         # Full data arrays
#         self.step_size = step_size
#         self.price_data = data_dict['data']
#         self.tech_data = data_dict['tech']
#         self.time_data = data_dict['time']
#         self.atr_data = data_dict['atr']
#         self.max_steps = len(self.price_data)
#
#         # FOR DEBUGGING
#         self.example_data = example_data
#
#         # Current data
#         self.current_step = 0  # RESET INIT
#         self.current_date = None  # RESET INIT
#         self.current_price_data = None  # RESET INIT
#         self.current_tech_data = None  # RESET INIT
#         self.current_time_data = None  # RESET INIT
#         self.terminal = False  # RESET INIT
#         self.current_ohlct = None  # RESET INIT
#
#         self.flat_data_size = self.price_data[0].shape[0] * self.price_data[0].shape[1]
#         self.flat_ta_size = self.tech_data[0].shape[0] * self.tech_data[0].shape[1]
#         self.flat_time_size = self.time_data.iloc[0].shape[0]
#
#         self.wallet = wallet
#         self.reward_scaling = reward_scaling
#         self.state_space = self.flat_data_size + self.flat_ta_size + self.flat_time_size + self.wallet.get_state(
#             state_space=True)
#
#         if strategy_trading:
#             self.action_space = spaces.Box(low=0,
#                                            high=2,
#                                            shape=(self.wallet.get_n_actions(),),
#                                            dtype=np.int16)
#         else:
#             self.action_space = spaces.Box(low=0,
#                                            high=4,
#                                            shape=(self.wallet.get_n_actions(),),
#                                            dtype=np.int16)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
#
#         # Initialize state
#         self.state = None
#
#         # initialize reward
#         self.reward = 0  # RESET
#         self.cost = 0  # RESET
#         self.trades = 0  # RESET
#         self.episode = 0  # RESET
#
#         # Memorize all the total balance change
#         self._seed()
#
#         # SCALERS
#         self.data_scaler = None
#         self.tech_scaler = None
#
#         # Initialize all values
#         self.get_data_window_scaler()
#         self.get_tech_window_scaler()
#
#         # Previous
#         self.previous_balance = 0
#         self.fast_forward = False
#
#         # LAST RESET
#         self.reset()
#
#     def step(self, action):
#         self.terminal = self.current_step >= (self.max_steps - 1) or self.wallet.game_over  # OK
#
#         if self.terminal:
#             if self.wallet.long_position:
#                 self.wallet.close_long(close_price=self.current_ohlct['close'],
#                                        close_time=self.current_ohlct['time'])
#             if self.wallet.short_position:
#                 self.wallet.close_short(close_price=self.current_ohlct['close'],
#                                         close_time=self.current_ohlct['time'])
#             return self.state, self.reward, self.terminal, {}  # OK
#         else:
#             if (self.wallet.total_trades + 1) % 50 == 0 and self.mode == 'train':
#                 current_balance = round(self.wallet.get_total_balance(), 3)
#                 print(f'{self.current_step}/{self.max_steps}',
#                       f'Wallet: {current_balance}$',
#                       f'Trades: {self.wallet.total_trades}',
#                       f'Delta: {round(current_balance - self.previous_balance, 3)}$')
#                 self.previous_balance = current_balance
#
#             # Cast action to int
#             action = int(action)  # convert into integer because we can't buy fraction of shares
#             # Get s total value
#             begin_total_asset = self.get_total_balance()
#             # Trade
#             if self.strategy_trading:
#                 self.strategy_trade(action=action)
#             else:
#                 self.full_trade(action=action)
#
#             if self.strategy_trading and action in [0, 1]:
#                 cur_date = self.current_date
#                 while self.wallet.short_position is not None or self.wallet.long_position is not None:
#                     if self.current_step > self.max_steps - 2 or self.wallet.game_over:
#                         self.terminal = True
#                         break
#                     self.current_step += 1
#                     self.update_wallet_and_env()
#                 if self.test:
#                     ff_date = self.current_date
#                     print(f"Fastforward from {cur_date} to {ff_date}")
#             else:
#                 # state: s -> s+1
#                 self.current_step += 1
#                 # update env data (state and ohlct_dict)
#                 self.update_wallet_and_env()
#
#             # Get s+1 value
#             end_total_asset = self.get_total_balance()
#
#             self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
#
#         return self.state, self.reward, self.terminal, {}
#
#     def full_trade(self, action: int):
#         action = self.action_validate(action)
#
#         # Open short position
#         if action == 0:
#             # Close long first
#             if self.wallet.long_position is not None:
#                 self.wallet.close_long(close_price=self.current_ohlct['close'],
#                                        close_time=self.current_date)
#             # Open short secondly
#             if self.wallet.short_position is None:
#                 self.wallet.open_short(open_time=self.current_date,
#                                        open_price=self.current_ohlct['close'],
#                                        current_atr=self._get_atrr())
#                 self.trades += 1
#
#         # Close short position
#         elif action == 1:
#             if self.wallet.short_position is not None:
#                 self.wallet.close_short(close_price=self.current_ohlct['close'],
#                                         close_time=self.current_date)
#
#         # Open long position
#         elif action == 2:
#             # Close short first
#             if self.wallet.short_position is not None:
#                 self.wallet.close_short(close_price=self.current_ohlct['close'],
#                                         close_time=self.current_date)
#             # Open long position
#             if self.wallet.long_position is None:
#                 self.wallet.open_long(open_time=self.current_date,
#                                       open_price=self.current_ohlct['close'],
#                                       current_atr=self._get_atrr())
#                 self.trades += 1
#
#         # Close long position
#         elif action == 3:
#             if self.wallet.long_position is not None:
#                 self.wallet.close_long(close_price=self.current_ohlct['close'],
#                                        close_time=self.current_date)
#
#         else:
#             pass
#
#     def strategy_trade(self, action: int):
#         if self.wallet.short_position is None and self.wallet.long_position is None:
#             # Open short position
#             if action == 0:
#                 self.wallet.open_short(open_time=self.current_date,
#                                        open_price=self.current_ohlct['close'],
#                                        current_atr=self._get_atrr())
#                 self.fast_forward = True
#                 self.trades += 1
#             # Open long position
#             elif action == 1:
#                 self.wallet.open_long(open_time=self.current_date,
#                                       open_price=self.current_ohlct['close'],
#                                       current_atr=self._get_atrr())
#                 self.fast_forward = True
#                 self.trades += 1
#             else:
#                 pass
#
#     # OK
#     def reset(self, **kwargs):
#         # Initiate all state_values to start
#         self.current_step = 0
#         self.current_date = str(self.time_data.index[self.current_step])
#         self.current_price_data = self.price_data[self.current_step]
#         self.current_tech_data = self.tech_data[self.current_step]
#         self.current_time_data = self.time_data.iloc[self.current_step]
#
#         self.terminal = False
#         self.reward = 0
#         self.cost = 0
#         self.trades = 0
#         self.episode = 0
#
#         # Reset wallet
#         self.current_ohlct = self.update_ohlct_dict()
#         self.wallet.reset(current_ohlct=self.current_ohlct)
#
#         # Generate state from wallet
#         self.state = self._update_state()
#
#         return self.state
#
#     # OK
#     def get_total_balance(self):
#         return self.wallet.get_total_balance()
#
#     # OK
#     def _get_atrr(self, step_size=None):
#         step_size = self.step_size if step_size is None else step_size
#         return self.atr_data.loc[str(self.time_data.index[self.current_step])][f'atrr_14_{step_size}min']
#
#     # OK
#     def _get_current_data_value(self, column) -> dict:
#         col_name = f'{column}_{self.step_size}min'
#         col_index = list(self.example_data['data'].columns).index(col_name)
#         return self.current_price_data[-1, col_index]
#
#     # OK
#     def _update_state(self) -> np.array:
#         wallet_state = self.wallet.get_state()
#         state = np.concatenate([
#             wallet_state,
#             window_to_array(
#                 self.data_scaler.transform(np.expand_dims(window_to_array(self.current_price_data), axis=0))),
#             window_to_array(
#                 self.tech_scaler.transform(np.expand_dims(window_to_array(self.current_tech_data), axis=0))),
#             self.current_time_data.values
#         ])
#         return state
#
#     # OK
#     def get_data_window_scaler(self, scaler='minmax'):
#         out = []
#
#         for window in self.price_data:
#             out.append(window_to_array(window=window))
#
#         sc = MinMaxScaler() if scaler == "minmax" else StandardScaler()
#         sc.fit(np.array(out))
#         self.data_scaler = sc
#
#     # OK
#     def get_tech_window_scaler(self, scaler='minmax'):
#         out = []
#         for window in self.tech_data:
#             out.append(window_to_array(window=window))
#
#         sc = MinMaxScaler() if scaler == "minmax" else StandardScaler()
#         sc.fit(np.array(out))
#         self.tech_scaler = sc
#
#     # OK
#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     # OK
#     def render(self, mode="human", close=False):
#         return self.state
#
#     # OK
#     def update_ohlct_dict(self):
#         return {'open': self._get_current_data_value(column='open'),
#                 'high': self._get_current_data_value(column='high'),
#                 'low': self._get_current_data_value(column='low'),
#                 'close': self._get_current_data_value(column='close'),
#                 'time': self.current_date}
#
#     def update_wallet_and_env(self):
#         self.current_date = str(self.time_data.index[self.current_step])
#         self.current_price_data = self.price_data[self.current_step]
#         self.current_tech_data = self.tech_data[self.current_step]
#         self.current_time_data = self.time_data.iloc[self.current_step]
#
#         # Render state and ohlct dict
#         self.current_ohlct = self.update_ohlct_dict()
#         self.wallet.step(ohlct_dict=self.current_ohlct)
#
#         # Generate state from wallet
#         self.state = self._update_state()
#
#     def save_scalers(self, path):
#         Utils.save_object(object_to_save=self.data_scaler, path=path, filename="data.scaler")
#         Utils.save_object(object_to_save=self.tech_scaler, path=path, filename="tech.scaler")
#
#     def load_scalers(self, path):
#         self.data_scaler = Utils.load_object(path=path, filename="data.scaler")
#         self.tech_scaler = Utils.load_object(path=path, filename="tech.scaler")
#
#     def action_validate(self, action):
#         action = int(action)
#
#         if self.wallet.short_position is None and self.wallet.long_position is None and action in [0, 2, 4]:
#             return action
#
#         elif self.wallet.short_position is not None and self.wallet.long_position is None and action in [1, 2, 4]:
#             return action
#
#         elif self.wallet.short_position is None and self.wallet.long_position is not None and action in [0, 3, 4]:
#             return action
#
#         else:
#             return 4
