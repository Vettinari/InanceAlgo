from typing import Tuple, List, Dict
import gymnasium
import numpy as np
from gym.core import ObsType
from gymnasium import spaces

from DataProcessing.data_pipeline import DataPipeline
from DataProcessing.timeframe import OHLCT, TimeFrame
from risk_manager import RiskManager


class TradeGym(gymnasium.Env):
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

        # Spaces
        self.action_space = spaces.Discrete(n=len(risk_manager.action_dict.values()), start=0)

        self.state_dimension = len(self.risk_manager.wallet.state) + self.data_pipeline.state_size()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dimension,))

        self.max_steps = 0
        self.reset()

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        self.done = self.current_step >= (self.max_steps - 1) or self.risk_manager.wallet.game_over  # OK

        if self.done:
            return self.state, self.reward, self.done, {}

        else:
            # if self.current_step % 1000 == 0:
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

            # print(f"Env_reward: {self.reward}\n"
            #       f"{self.risk_manager.wallet}")

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

        self.data_state = data
        self.tech_state = tech

    def generate_state(self):
        self.state = [*self.wallet_state, *self.data_state, *self.tech_state]

    def __repr__(self):
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_pipeline={self.data_pipeline}\n' \
               f'OHLCT={self.current_ohlct}\n>'
