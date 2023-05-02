import time
from pprint import pprint
from typing import Tuple, List, Dict, Optional
import gymnasium
import numpy as np
import pandas as pd
import wandb
from gym.core import ObsType
from gymnasium import spaces

from DataProcessing.data_stream import DataStream
from old_DataProcessing.data_pipeline import DataPipeline
from old_DataProcessing.timeframe import OHLCT, TimeFrame
from reward_buffer import RewardBuffer
from risk_manager import RiskManager, Action, ActionValidator


class TradeGym(gymnasium.Env):
    def __init__(self,
                 risk_manager: RiskManager,
                 data_stream: DataStream,
                 action_validator: ActionValidator,
                 reward_buffer: RewardBuffer,
                 reward_scaling: float,
                 verbose: Optional[int] = 250,
                 wandb_logger: bool = False,
                 env_type: str = 'train',
                 test=False):
        self.env_type = env_type
        self.test = test
        # Total state
        self.state = None
        # Separate states
        self.wallet_state = None
        self.data_state = None
        self.tech_state = None
        self.chart_state = None

        self.done = False
        self.current_step = 0
        self.current_date = None
        self.current_ohlct: OHLCT = None

        # Rewards
        self.reward_scaling = reward_scaling
        self.reward = 0
        self.rewards = []

        # Objects
        self.risk_manager: RiskManager = risk_manager
        self.data_stream: DataStream = data_stream
        self.action_validator: ActionValidator = action_validator
        self.reward_buffer: RewardBuffer = reward_buffer

        self.max_steps = self.data_stream.max_steps(data_type=self.env_type)
        self.verbose = verbose

        self.wandb_logger = wandb_logger
        self.connection = None
        self.reset()

        # Spaces generation after reset
        self.action_space = spaces.Discrete(n=len(risk_manager.action_dict.values()), start=0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state),))

    def step(self, agent_action) -> Tuple[np.array, float, bool, bool, dict]:
        if self.current_step >= (self.max_steps - 1) or self.risk_manager.wallet.game_over:
            self.done = 1

        if self.done:
            self.connection.finish()
            return self.state, -10.0, False, True, {}

        log_dict = {}
        action_class = self.risk_manager.get_action_object(action_index=agent_action)

        # if action is valid create continue else return bad action reward and same state
        if self.action_validator.validate_action(position=self.risk_manager.wallet.position,
                                                 action=action_class.action):

            self.risk_manager.execute_action(action=action_class,
                                             current_atr=self.get_atr())

            # state: S -> S+1
            self.current_step += 1

            # Update wallet and environment elements for state generation
            self.update_wallet_and_env()

            if self.wandb_logger:
                log_dict.update({'total_score': round(sum(self.rewards), 3),
                                 'wallet_balance': self.risk_manager.wallet.total_balance,
                                 'position_count': len(self.risk_manager.wallet.closed_positions)})
                log_dict.update(self.risk_manager.action_history)
                log_dict.update({'sentiment': self.risk_manager.sentiment})
                log_dict.update({'price_action': self.current_ohlct.close})
                returns = self.risk_manager.wallet.returns
                if len(returns) > 1:
                    log_dict.update(self.risk_manager.wallet.evaluation(returns=returns))
            if self.current_step % self.verbose == 0:
                print(f"{self.current_step}/{self.max_steps} "
                      f"Score: {round(sum(self.rewards), 3)} | "
                      f"Balance={self.risk_manager.wallet.total_balance}$ | "
                      f"Positions={len(self.risk_manager.wallet.closed_positions)}")

        current_rewards = self.reward_buffer.yield_rewards()

        if self.test:
            pprint("REWARDS:", current_rewards)
            print()

        self.reward = sum([reward for reward in current_rewards.values()])
        self.rewards.append(self.reward)

        if self.wandb_logger:
            log_dict.update(current_rewards)
            self.connection.log(log_dict)

        if self.test:
            print("Current state:", self.state[:6])

        return self.state, self.reward, False, self.done, {}

    def reset(self) -> tuple[ObsType, dict]:
        self.reward_buffer.reset()
        self.risk_manager.reset()

        self.done = False
        self.reward = 0
        self.rewards = []
        self.current_step = 0

        # Initialize data state
        self.generate_data_state()
        self.update_OHLCT()

        # Initialize wallet state
        self.risk_manager.wallet.reset(ohlct=self.current_ohlct)
        self.wallet_state = self.risk_manager.wallet.state(include_balance=False)

        if self.wandb_logger:
            self.connection = wandb.init(entity="miloszbertman", project='TradingGym')
            self.connection.config.log_interval = self.verbose

        return self.state, {}

    def update_OHLCT(self):
        self.current_ohlct = OHLCT(self.data_state[self.data_stream.step_size].iloc[-1],
                                   timeframe=self.data_stream.step_size)
        if self.test:
            print("OHLCT:", self.current_ohlct)

    def generate_data_state(self):
        key = self.env_type + f"_{self.current_step}"
        self.data_state = self.data_stream[key]

    def __repr__(self):
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_stream={self.data_stream}\n' \
               f'OHLCT={self.current_ohlct}\n>'
