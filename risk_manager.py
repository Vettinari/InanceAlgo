from pprint import pprint

import pandas as pd

from DataProcessing.timeframe import OHLCT
from reward_buffer import ActionReward, TransactionReward, IntermediateReward, RewardBuffer
from wallet import Wallet
from typing import List, Optional, Union


class Action:
    def __init__(self, action: str, stop_loss: Optional[float], risk_reward: Optional[float]):
        self.action = action
        self.stop_loss = stop_loss
        self.risk_reward = risk_reward

    def __repr__(self):
        return f"Action({self.action}, sl: {self.stop_loss}, rr: {self.risk_reward})"

    def __str__(self):
        return f"{self.action}_sl:{self.stop_loss}_rr:{self.risk_reward}"

    def __float__(self):
        return


class RiskManager:
    def __init__(
            self,
            ticker: str,
            initial_balance: float,
            reward_buffer: RewardBuffer,
            atr_stop_loss_ratios: Optional[List[float]] = None,
            risk_reward_ratios: Optional[List[int]] = None,
            position_closing: bool = True,
            portfolio_risk: Optional[float] = None,
            stop_loss_type: str = 'atr',
            wallet: Optional[Wallet] = None):
        self.initial_balance = initial_balance
        self.atr_stop_loss_ratios = atr_stop_loss_ratios or [2, 3]
        self.risk_reward_ratios = risk_reward_ratios or [1, 2, 3]
        self.trade_risk = initial_balance * (portfolio_risk or 0.02)
        self.position_closing = position_closing
        self.wallet = wallet or Wallet(ticker=ticker,
                                       initial_balance=initial_balance,
                                       reward_buffer=reward_buffer)
        self.action_dict = self._generate_action_space()
        self.stop_loss_type = stop_loss_type
        self.reward_buffer: RewardBuffer = reward_buffer
        self.idle_time = 0

        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}
        self._current_sentiment: dict = dict(zip(['short', 'long', 'close', 'hold'], [0, 0, 0, 0]))

    def reset(self):
        self._current_sentiment = dict(zip(['short', 'long', 'close', 'hold'], [0, 0, 0, 0]))
        self._action_history: dict = {str(action): 0 for action in self.action_dict.values()}

    @property
    def sentiment(self):
        out = self._current_sentiment
        self._current_sentiment = dict(zip(['short', 'long', 'close', 'hold'], [0, 0, 0, 0]))
        return out

    @property
    def action_history(self):
        return self._action_history

    def _generate_action_space(self, ) -> dict:
        actions = ['long', 'short', 'hold']
        action_space = []

        for action in actions:
            if action in ['long', 'short']:
                for risk_reward in self.risk_reward_ratios:
                    for sl in self.atr_stop_loss_ratios:
                        action_space.append(Action(action=action, stop_loss=sl, risk_reward=risk_reward))
            else:
                action_space.append(Action(action='hold', stop_loss=None, risk_reward=None))

        if self.position_closing:
            action_space.append(Action(action='close', stop_loss=None, risk_reward=None))

        action_space = dict(zip(range(0, len(action_space)), action_space))

        return action_space

    def _validate_action(self, action: str) -> bool:
        flag = True
        position = self.wallet.position

        if (position and position.type == action) or (position is None and action == 'close'):
            self.reward_buffer.reward_action(reward=-100.0)
            flag = False

        if action == 'hold' and self.wallet.position is None:
            self.reward_buffer.reward_action(reward=0 - 1 * self.idle_time)
            self.idle_time += 1

        if flag:
            self.reward_buffer.reward_action(reward=1)
            self.idle_time = 0

        return flag

    def execute_action(self, action_index: int, current_atr: float):
        # Get Action object
        action = self.action_dict[action_index]
        # Update action history
        self._action_history[str(action)] += 1
        # Update current_sentiment
        self._current_sentiment[action.action] = 1 * action.risk_reward if action.risk_reward else 1

        if self._validate_action(action=action.action):
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
