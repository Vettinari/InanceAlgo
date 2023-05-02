import collections
from datetime import timedelta
from pprint import pprint
from typing import Tuple, Optional, Dict
import gymnasium
import numpy as np
import wandb
from gym.core import ObsType
from gymnasium import spaces

from DataProcessing.data_stream import DataStream
from DataProcessing.ohlct import OHLCT
from reward_buffer import RewardBuffer
from risk_manager import RiskManager, ActionValidator


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
        self.state: np.array = None
        # Separate states
        self.timeframe_dict = None
        self.tech_state = None
        self.chart_state = None

        self.done = False
        self.current_step = 0
        self.current_date = None
        self.current_ohlct: OHLCT = None

        # Rewards
        self.reward = 0
        self.reward_scaling = reward_scaling

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
        log_dict = {}

        if self.current_step >= (self.max_steps - 1) or self.risk_manager.wallet.game_over:
            self.done = 1

        if self.done:
            self.connection.finish()
            return self.state, self.reward, False, True, {}

        action_class = self.risk_manager.get_action_object(action_index=agent_action)

        # if action is valid create continue else return bad action reward and same state
        if self.action_validator.validate_action(position=self.risk_manager.wallet.position,
                                                 action=action_class.action):
            self.risk_manager.execute_action(action=action_class)

            # state: S -> S+1
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

            if self.wandb_logger:
                returns = self.risk_manager.wallet.returns
                if len(returns) > 1:
                    log_dict.update(self.risk_manager.wallet.log_info(returns))
                log_dict.update(self.reward_buffer.log_info(sum_only=True))

                if self.current_step % self.verbose == 0:
                    print(f"{self.current_step}/{self.max_steps}")

        current_rewards = self.process_rewards(current_rewards=self.reward_buffer.yield_rewards())
        self.reward = sum(list(current_rewards.values()))

        if self.test:
            pprint(current_rewards)
            print()

        if self.wandb_logger:
            log_dict.update(current_rewards)
            self.connection.log(log_dict)

        if self.test:
            print("Current date:")
            print(self.current_date)
            print("Current ohlct:")
            print(self.current_ohlct)

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
        self.risk_manager.reset(dataframe=self.timeframe_dict[self.data_stream.step_size].iloc[-15:])
        self.risk_manager.wallet.reset(ohlct=self.current_ohlct)

        if self.wandb_logger:
            self.connection = wandb.init(entity="miloszbertman", project='TradingGym')
            self.connection.config.log_interval = self.verbose

        self.update_state()

        return self.state, {}

    def update_OHLCT(self):
        self.current_ohlct = OHLCT(self.timeframe_dict[self.data_stream.step_size].iloc[-1],
                                   timeframe=self.data_stream.step_size)

    def data_state_from_datastream(self):
        key = self.env_type + f"_{self.current_step}"
        self.timeframe_dict = self.data_stream[key]
        self.timeframe_dict = collections.OrderedDict(sorted(self.timeframe_dict.items()))

    def update_state(self):
        wallet_state = self.risk_manager.wallet.state(include_balance=False)
        data_state = np.array(
            [timeframe_state.iloc[-1].values for timeframe_state in self.timeframe_dict.values()]).flatten()
        self.state = np.hstack([*data_state, *wallet_state])

    def process_rewards(self, current_rewards: Dict[str, float]) -> Dict[str, float]:
        out = current_rewards

        if current_rewards['transaction'] > 0:
            max_gain = abs(max(self.risk_manager.risk_reward_ratios) * self.risk_manager.trade_risk)
            out['transaction'] = current_rewards['transaction'] / max_gain
        elif current_rewards['transaction'] <= 0:
            out['transaction'] = current_rewards['transaction'] / self.risk_manager.trade_risk
            out['drawdown'] = 0

        return out

    def __repr__(self):
        return f'<TradeGym:\n' \
               f'risk_manager={self.risk_manager}\n' \
               f'data_stream={self.data_stream}\n' \
               f'OHLCT={self.current_ohlct}\n>'
