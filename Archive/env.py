import collections
from datetime import timedelta
from pprint import pprint
from typing import Tuple, Optional, Dict
import gymnasium
import numpy as np
import pandas as pd
import wandb
from gym.core import ObsType
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import dill

from DataProcessing.data_stream import DataStream
from DataProcessing.ohlct import OHLCT
from reward_buffer import RewardBuffer
from risk_manager import RiskManager


class TradeGym(gymnasium.Env):
    def __init__(self,
                 risk_manager: RiskManager,
                 data_stream: DataStream,
                 reward_buffer: RewardBuffer,
                 reward_scaling: float,
                 verbose: Optional[int] = 250,
                 wandb_logger: bool = False,
                 env_type: str = 'train',
                 state_stacking: int = 4,
                 full_control: Optional[bool] = True,
                 test: Optional[bool] = False,
                 predictor_path: str = None):

        self.env_type: str = env_type
        self.test: bool = test

        # Total state
        self.state: np.array = None

        # Separate states
        self.state_stacking: int = state_stacking
        self.current_timeframe_dict: Dict[int, pd.DataFrame] = None
        self.tech_state: np.array = None
        self.chart_state = None
        self.full_control: bool = full_control

        self.done: bool = False
        self.current_step: int = 0
        self.current_date = None
        self.current_ohlct: OHLCT = None

        # Rewards
        self.reward: float = 0
        self.reward_scaling: float = reward_scaling

        # Objects
        self.risk_manager: RiskManager = risk_manager
        self.data_stream: DataStream = data_stream
        self.reward_buffer: RewardBuffer = reward_buffer

        self.max_steps: int = self.data_stream.max_steps(data_type=self.env_type)
        self.verbose: int = verbose

        self.wandb_logger: bool = wandb_logger
        self.connection = None
        self.states_stack: np.array = None
        self.reset()

        # Spaces generation after reset
        self.action_space = spaces.Discrete(n=len(risk_manager.action_dict.values()), start=0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state * self.state_stacking),))
        self.predictor = dill.load(open(f"{predictor_path}", "rb"))

    def step(self, agent_action) -> Tuple[np.array, float, bool, bool, dict]:
        log_dict = {}

        if self.current_step >= (self.max_steps - 1) or self.risk_manager.wallet.game_over:
            self.done = 1

        if self.done:
            if self.wandb_logger:
                self.connection.finish()
            return self.state, self.reward, False, self.done, {}

        action_class = self.risk_manager.get_action_object(action_index=agent_action)
        self.risk_manager.execute_action(action=action_class)

        if self.full_control:
            self.current_step += 1
            # Update data state
            self.data_state_from_datastream()
            # Update OHLCT state
            self.update_OHLCT()
            # Update date
            self.current_date = self.current_ohlct.time + timedelta(minutes=self.data_stream.step_size)
            # Update risk manager
            self.risk_manager.update_ohlc_buffer(ohlct=self.current_ohlct)
            self.risk_manager.update_wallet(ohlct=self.current_ohlct)
            self.update_state()
            self.update_state_stack()
            self.state = self.states_stack.flatten()
        else:
            while self.risk_manager.wallet.position and self.current_step < self.max_steps:
                self.current_step += 1
                # Update data state
                self.data_state_from_datastream()
                # Update OHLCT state
                self.update_OHLCT()
                # Update date
                self.current_date = self.current_ohlct.time + timedelta(minutes=self.data_stream.step_size)
                # Update risk manager
                self.risk_manager.update_ohlc_buffer(ohlct=self.current_ohlct)
                self.risk_manager.update_wallet(ohlct=self.current_ohlct)

            self.update_state()
            self.update_state_stack()
            self.state = self.states_stack.flatten()

        current_rewards, rewards_info = self.reward_buffer.yield_rewards(max_gain=self.risk_manager.max_gain,
                                                                         trade_risk=self.risk_manager.trade_risk)

        if self.wandb_logger:
            log_dict.update(rewards_info)
            log_dict.update(current_rewards)

            returns = self.risk_manager.wallet.returns
            if len(returns) > 1:
                # Wallet logs
                log_dict.update(self.risk_manager.wallet.log_info(returns))

            # Action logs
            log_dict.update(self.risk_manager.get_log_info())
            self.connection.log(log_dict)

        self.reward = sum(list(current_rewards.values()))

        if self.test:
            print("Current date:", self.current_date)
            print("Current rewards:", current_rewards, "Total reward:", self.reward)
            print("Current state:", self.state)

        if self.current_step % self.verbose == 0:
            print(
                "Progress {}/{} - ${}".format(self.current_step, self.max_steps,
                                              self.risk_manager.wallet.total_balance))

        return self.state, self.reward, False, self.done, {}

    def reset(self) -> tuple[ObsType, dict]:
        self.reward_buffer.reset()

        self.done = False
        self.reward = 0
        self.current_step = 0

        # Initialize data state
        self.data_state_from_datastream()
        self.update_OHLCT()
        self.current_date = self.current_ohlct.time

        # Initialize wallet state
        self.risk_manager.reset(dataframe=self.current_timeframe_dict[self.data_stream.step_size].iloc[-15:])
        self.risk_manager.wallet.reset(ohlct=self.current_ohlct)

        if self.wandb_logger:
            self.connection = wandb.init(entity="miloszbertman", project=f'TradingGym_{self.env_type}')
            self.connection.config.log_interval = self.verbose

        self.update_state()
        self.states_stack = np.zeros((self.state_stacking, len(self.state)), dtype=float)
        self.update_state_stack()
        self.state = self.states_stack.flatten()

        return self.state, {}

    def update_OHLCT(self) -> None:
        self.current_ohlct = OHLCT(self.current_timeframe_dict[self.data_stream.step_size].iloc[-1],
                                   timeframe=self.data_stream.step_size)

    def data_state_from_datastream(self) -> None:
        key = self.env_type + f"_{self.current_step}"
        self.current_timeframe_dict = self.data_stream[key]
        self.current_timeframe_dict = collections.OrderedDict(sorted(self.current_timeframe_dict.items()))

    def update_state(self) -> None:
        # Generate wallet state with scaled values
        wallet_state = self.risk_manager.wallet.state(include_balance=False,
                                                      max_gain=self.risk_manager.max_gain,
                                                      trade_risk=self.risk_manager.trade_risk)
        # Generate data state from timeframe_dict
        data_state = np.array(
            [timeframe_state.iloc[-1].values for timeframe_state in self.current_timeframe_dict.values()]).flatten()

        # Collapse states to horizontal stack
        self.state = np.hstack([*data_state, *wallet_state]).astype(float)

    def update_state_stack(self):
        self.states_stack = np.roll(a=self.states_stack, shift=1, axis=0)
        self.states_stack[0] = self.state

    def __repr__(self) -> str:
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_stream={self.data_stream}\n' \
               f'OHLCT={self.current_ohlct}\n>'
