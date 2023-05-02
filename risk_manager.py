from pprint import pprint

import pandas as pd

from old_DataProcessing.timeframe import OHLCT
from positions import Position
from reward_buffer import RewardBuffer
from wallet import Wallet
from typing import List, Optional, Union


class Action:
    def __init__(self, action: str, stop_loss: Optional[float], risk_reward: Optional[float], period: Optional[int]):
        self.action = action
        self.stop_loss = stop_loss
        self.risk_reward = risk_reward

    def __repr__(self):
        return f"Action({self.action}, sl: {self.stop_loss}, rr: {self.risk_reward})"

    def __str__(self):
        return f"{self.action}_sl:{self.stop_loss}_rr:{self.risk_reward}"

    def __float__(self):
        return

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
        self._reward = 0

    def yield_rewards(self) -> float:
        out = {"action": self._reward}
        self._reward = 0
        return out

    def validate_action(self, position: Position, action: str) -> bool:
        flag = True

        # Standard verification
        if position is None and (action == 'long' or action == 'short' or action == 'hold'):
            self._reward = 0
            return flag

        # Position cloning
        if position and position.type == action:
            self._reward = self.action_penalty
            flag = False
            return flag

        # Closing blank position
        if not position and action == 'close':
            self._reward = self.action_penalty
            flag = False
            return flag

        # Position reversing
        if not self.position_reversing:
            if position and action not in ['hold', 'close']:
                self._reward = self.action_penalty
                flag = False
                return flag

        if not self.position_pre_closing:
            if position and action != 'hold':
                self._reward = self.action_penalty
                flag = False
                return flag

        return flag


class RiskManager:
    def __init__(
            self,
            wallet: Wallet,
            atr_stop_loss_ratios: Optional[List[float]] = None,
            risk_reward_ratios: Optional[List[int]] = None,
            portfolio_risk: Optional[float] = None,
    ):
        self.wallet = wallet
        self._reward: float = 0
        self.initial_balance = wallet.initial_balance
        self.atr_stop_loss_ratios = atr_stop_loss_ratios or [2, 3]
        self.risk_reward_ratios = risk_reward_ratios or [1.5, 2, 3]
        self.trade_risk = self.initial_balance * (portfolio_risk or 0.02)
        self.action_dict = self._generate_action_space()
        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}
        self._current_sentiment: float = 0

    def reset(self):
        self._reward = 0
        self._current_sentiment = 0
        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}

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
                        action_space.append(Action(action=action, stop_loss=sl, risk_reward=risk_reward, period=0))

        action_space.append(Action(action='hold', stop_loss=None, risk_reward=None, period=1))
        action_space.append(Action(action='close', stop_loss=None, risk_reward=None, period=0))

        action_space = dict(zip(range(0, len(action_space)), action_space))

        return action_space

    def get_action_object(self, action_index) -> Action:
        return self.action_dict[action_index]

    def execute_action(self, action: Action, current_atr: float):
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

    def wallet_step(self, ohlct: OHLCT):
        self.wallet.update_wallet(ohlct=ohlct)

    def info(self):
        print('Risk manager:')
        print('Initial balance:', self.initial_balance)
        print('Stop losses:', self.atr_stop_loss_ratios)
        print('Risk/Reward ratios:', self.risk_reward_ratios)
        print('Trade risk:', self.trade_risk)
        print('Action space:', len(self._generate_action_space()))
        pprint(self._generate_action_space())

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
               f"Balance={self.initial_balance}, " \
               f"atr_stop_losses={self.atr_stop_loss_ratios}, " \
               f"risk_reward_ratios={self.risk_reward_ratios}, " \
               f"trade_risk={self.trade_risk}, " \
               f"action_space_size={len(self.action_dict.keys())}>"
