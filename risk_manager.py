from pprint import pprint

import numpy as np
import pandas as pd
import pandas_ta as ta

import positions
from DataProcessing.ohlct import OHLCT
from positions import Position
from reward_buffer import RewardBuffer
from wallet import Wallet
from typing import List, Optional


class Action:
    def __init__(self, action: str, stop_loss: Optional[float], risk_reward: Optional[float]):
        self.action = action
        self.stop_loss = stop_loss
        self.risk_reward = risk_reward

    def __repr__(self):
        return f"Action({self.action}, sl: {self.stop_loss}, rr: {self.risk_reward})"

    def __str__(self):
        return f"{self.action}_sl:{self.stop_loss}_rr:{self.risk_reward}"

    @property
    def sentiment(self):
        if self.action == 'long':
            return self.risk_reward
        elif self.action == 'short':
            return -self.risk_reward
        else:
            return 0


class ActionValidator:

    def __init__(self,
                 reward_buffer: RewardBuffer,
                 position_reversing: Optional[bool] = None,
                 position_closing: Optional[bool] = None,
                 action_penalty: Optional[float] = None):
        self.position_reversing = position_reversing or False
        self.position_pre_closing = position_closing or False
        self.action_penalty = action_penalty or -0.001
        self.reward_buffer = reward_buffer

    def validate_action(self, position: Position, action: str) -> bool:
        flag = True
        reward = 0

        # Standard verification
        if position is None and (action == 'long' or action == 'short' or action == 'hold'):
            reward = 0
            return flag

        # Position cloning
        if position and position.type == action:
            reward = self.action_penalty
            flag = False
            return flag

        # Closing blank position
        if not position and action == 'close':
            reward = self.action_penalty
            flag = False
            return flag

        # Position reversing
        if not self.position_reversing:
            if position and action not in ['hold', 'close']:
                reward = self.action_penalty
                flag = False
                return flag

        if not self.position_pre_closing:
            if position and action != 'hold':
                reward = self.action_penalty
                flag = False
                return flag

        self.reward_buffer.add_reward(reward_name='action', reward_value=reward)
        return flag


class RiskManager:
    def __init__(
            self,
            wallet: Wallet,
            reward_buffer: RewardBuffer,
            use_atr: bool = True,
            stop_loss_ratios: Optional[List[float]] = None,
            risk_reward_ratios: Optional[List[int]] = None,
            portfolio_risk: Optional[float] = None,
            base_pip_loss: Optional[float] = None
    ):
        self.wallet: Wallet = wallet
        self.use_atr: bool = use_atr
        self.base_pip_loss: Optional[float] = (base_pip_loss or 15)
        self.initial_balance: float = wallet.initial_balance
        self.atr_stop_loss_ratios: list = stop_loss_ratios or [2]
        self.risk_reward_ratios: list = risk_reward_ratios or [2]
        self.trade_risk: float = self.initial_balance * (portfolio_risk or 0.02)
        self.action_dict: dict = self._generate_action_space()
        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}
        self._current_sentiment: float = 0
        self.ohlc_buffer = pd.DataFrame(columns=['Datetime', 'open', 'high', 'low', 'close', 'volume'], index=[])
        self.reward_buffer = reward_buffer
        self.reward_buffer.set_trade_risk(self.trade_risk)

    def reset(self, dataframe: pd.DataFrame):
        self._current_sentiment = 0
        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}
        self.ohlc_buffer = dataframe

    @property
    def sentiment(self):
        out = self._current_sentiment
        self._current_sentiment = 0
        return out

    @property
    def action_history(self):
        return self._action_history

    def _generate_action_space(self) -> dict:
        actions = ['long', 'short', 'hold']
        action_space = []

        for action in actions:
            if action in ['long', 'short']:
                for risk_reward in self.risk_reward_ratios:
                    for sl in self.atr_stop_loss_ratios:
                        action_space.append(Action(action=action, stop_loss=sl, risk_reward=risk_reward))

        action_space.append(Action(action='hold', stop_loss=None, risk_reward=None))
        action_space.append(Action(action='close', stop_loss=None, risk_reward=None))

        action_space = dict(zip(range(0, len(action_space)), action_space))

        return action_space

    def get_action_object(self, action_index) -> Action:
        return self.action_dict[action_index]

    def execute_action(self, action: Action):
        if self.use_atr:
            current_atr = self.get_atr()
        else:
            current_atr = self.base_pip_loss * positions.XTB[self.wallet.ticker]['one_pip_size']

        # Update action history
        self._action_history[str(action)] += 1
        # Update current_sentiment
        self._current_sentiment = action.sentiment

        if action.action == "close":
            self.wallet.position_close()

        elif action.action == "long":
            stop_loss_delta = round(current_atr * action.stop_loss, 5)
            self.wallet.open_long(stop_loss_delta=stop_loss_delta,
                                  risk_reward_ratio=action.risk_reward,
                                  position_risk=self.trade_risk)

        elif action.action == "short":
            stop_loss_delta = round(current_atr * action.stop_loss, 5)
            self.wallet.open_short(stop_loss_delta=stop_loss_delta,
                                   risk_reward_ratio=action.risk_reward,
                                   position_risk=self.trade_risk)

    def info(self):
        print('Risk manager:')
        print('Stop losses:', self.atr_stop_loss_ratios)
        print('Risk/Reward ratios:', self.risk_reward_ratios)
        print('Trade risk:', self.trade_risk)
        pprint(self._generate_action_space())

    def log_info(self):
        out = self._action_history
        out.update({'current_sentiment': self._current_sentiment})
        return out

    def update_ohlc_buffer(self, ohlct: OHLCT):
        self.ohlc_buffer = pd.concat([self.ohlc_buffer, ohlct.dataframe],
                                     ignore_index=False).iloc[-15:]
        # self.ohlc_buffer = self.ohlc_buffer.append(ohlct.dataframe, ignore_index=False).iloc[-15:]

    def get_atr(self):
        atr = self.ohlc_buffer.ta.atr(length=14, close='close', high='high', low='low').values[-1]
        return round(atr, 5)

    def update_wallet(self, ohlct: OHLCT):
        self.wallet.update_wallet(ohlct=ohlct)

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
               f"atr_stop_losses={self.atr_stop_loss_ratios}, " \
               f"risk_reward_ratios={self.risk_reward_ratios}, " \
               f"trade_risk={self.trade_risk}, " \
               f"action_space_size={len(self.action_dict.keys())}>"
