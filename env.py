import time
from pprint import pprint
from typing import Tuple, List, Dict, Optional
import gymnasium
import numpy as np
import pandas as pd
import wandb
from gym.core import ObsType
from gymnasium import spaces

from DataProcessing.data_pipeline import DataPipeline
from DataProcessing.timeframe import OHLCT, TimeFrame
from risk_manager import RiskManager


class TradeGym(gymnasium.Env):
    def __init__(self,
                 risk_manager: RiskManager,
                 data_pipeline: DataPipeline,
                 reward_scaling: float,
                 verbose: Optional[int] = None):
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
        self.rewards = []
        # Rewards Detail
        self.transaction_rewards = []
        self.action_rewards = []
        self.intermediate_rewards = []
        self.sharpe_rewards = []
        self.drawdown_rewards = []

        # Objects
        self.risk_manager: RiskManager = risk_manager
        self.data_pipeline: DataPipeline = data_pipeline

        self.max_steps = 0
        self.reset()
        self.verbose = verbose or 200

        # Spaces
        self.action_space = spaces.Discrete(n=len(risk_manager.action_dict.values()), start=0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state),))

        self.connection = wandb.init(entity="miloszbertman", project=f'TradingGym')
        self.wandb_config = self.connection.config
        self.wandb_config.log_interval = verbose

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:

        if self.current_step >= (self.max_steps - 1) \
                or self.risk_manager.wallet.game_over \
                or sum(self.rewards) <= -5000:
            self.done = 1

        if self.done:
            return self.state, self.reward, False, self.done, {}

        else:
            # if self.current_step % 1000 == 0:
            self.risk_manager.execute_action(action_index=int(action),
                                             current_atr=self.get_atr())
            # state: S -> S+1
            self.current_step += 1
            self.data_pipeline.step_forward()

            # Update wallet and environment elements for state generation
            self.update_wallet_and_env()

            log_dict = self.risk_manager.reward_buffer.get_log_info()
            log_dict.update({'Score': round(sum(self.rewards), 3),
                             'Balance': self.risk_manager.wallet.total_balance,
                             'Positions': len(self.risk_manager.wallet.closed_positions)})

            self.connection.log(log_dict)
            self.reward = self.risk_manager.reward_buffer.yield_rewards()
            self.rewards.append(self.reward)

            if self.current_step % self.verbose == 0:
                print(f"{self.current_step}/{self.max_steps} "
                      f"Score: {round(sum(self.rewards), 3)} | "
                      f"Balance={self.risk_manager.wallet.total_balance}$ | "
                      f"MaxBalance={self.risk_manager.wallet.max_balance}$ | "
                      f"Positions={len(self.risk_manager.wallet.closed_positions)}")
                pprint(self.risk_manager.reward_buffer.get_info())
                print()

            return self.state, self.reward, False, self.done, {}

    def get_atr(self, interval=15):
        return float(self.timeframes[interval].tech()['atrr_14'])

    def update_wallet_and_env(self):
        self.current_ohlct = self.data_pipeline.ohlct
        self.current_date = self.data_pipeline.current_date
        self.risk_manager.wallet_step(ohlct=self.current_ohlct)
        self.update_states()
        self.generate_state()

    def reset(self) -> tuple[ObsType, dict]:
        self.done = False
        self.rewards = []
        self.reward = 0
        self.current_step = 0
        self.data_pipeline.reset_step()
        self.current_ohlct = self.data_pipeline.ohlct
        self.current_date = self.data_pipeline.current_date
        self.risk_manager.wallet.reset(ohlct=self.current_ohlct)
        self.update_states()
        self.generate_state()
        self.max_steps = self.data_pipeline.dataframe.shape[0] - self.data_pipeline.current_step
        self.risk_manager.reward_buffer.reset(history=True)

        return self.state, {}

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

        self.data_state = np.hstack(data).flatten()
        self.tech_state = np.hstack(tech).flatten()

    def generate_state(self, include_ohlct_data=False):
        if include_ohlct_data:
            self.state = np.hstack([self.wallet_state, self.data_state, self.tech_state])

        self.state = np.hstack([self.wallet_state, self.tech_state])

    def __repr__(self):
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_pipeline={self.data_pipeline}\n' \
               f'OHLCT={self.current_ohlct}\n>'
