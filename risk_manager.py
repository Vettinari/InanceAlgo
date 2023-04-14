from pprint import pprint
from DataProcessing.data_processor import OHLCT
from reward_system import ActionReward, TransactionReward, IntermediateReward
from wallet import Wallet
from typing import List, Optional, Union


class Action:
    def __init__(self, action: str, stop_loss: Optional[float], risk_reward: Optional[float]):
        self.action = action
        self.stop_loss = stop_loss
        self.risk_reward = risk_reward

    def __repr__(self):
        return f"Action({self.action}, sl: {self.stop_loss}, rr: {self.risk_reward})"


class RiskManager:
    def __init__(
            self,
            ticker: str,
            initial_balance: float,
            atr_stop_loss_ratios: Optional[List[float]] = None,
            risk_reward_ratios: Optional[List[int]] = None,
            manual_position_closing: bool = True,
            portfolio_risk: Optional[float] = None,
            stop_loss_type: str = 'atr',
            wallet: Optional[Wallet] = None
    ):
        self.initial_balance = initial_balance
        self.atr_stop_loss_ratios = atr_stop_loss_ratios or [2, 3]
        self.risk_reward_ratios = risk_reward_ratios or [1, 2, 3]
        self.trade_risk = initial_balance * (portfolio_risk or 0.02)
        self.manual_position_closing = manual_position_closing
        self.wallet = wallet or Wallet(ticker=ticker, initial_balance=initial_balance)
        self.action_dict = self._generate_action_space()
        self.stop_loss_type = stop_loss_type
        self.action_reward: ActionReward = None
        self.transaction_reward: TransactionReward = None
        self.intermediate_reward: IntermediateReward = None
        self._initialize_rewards()

    def _initialize_rewards(self):
        self.action_reward = ActionReward(reward=0)
        self.transaction_reward = TransactionReward()
        self.intermediate_reward = IntermediateReward(position=None, scaling_factor=None)

    def _generate_action_space(self) -> dict:
        actions = ['long', 'short', 'hold']
        action_space = []

        for action in actions:
            if action in ['long', 'short']:
                for risk_reward in self.risk_reward_ratios:
                    for sl in self.atr_stop_loss_ratios:
                        action_space.append(Action(action=action, stop_loss=sl, risk_reward=risk_reward))
            else:
                action_space.append(Action(action='hold', stop_loss=None, risk_reward=None))

        if self.manual_position_closing:
            action_space.append(Action(action='close', stop_loss=None, risk_reward=None))

        action_space = dict(zip(range(0, len(action_space)), action_space))

        return action_space

    def _validate_action(self, action: str) -> bool:
        position = self.wallet.position
        if position and position.type == action or (not position and action == 'close'):
            self.action_reward = ActionReward(reward=-1)
            return False
        self.action_reward = ActionReward(reward=0.1)
        return True

    def execute_action(self, action_index: int, current_atr: float):
        action = self.action_dict[action_index]
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
        self._update_rewards()

    def _update_rewards(self):
        self.transaction_reward = self.wallet.transaction_reward
        self.intermediate_reward = self.wallet.intermediate_reward

    def yield_rewards(self) -> List[Union[ActionReward, TransactionReward, IntermediateReward]]:
        return [self.action_reward, self.transaction_reward, self.intermediate_reward]

    @property
    def current_rewards(self):
        return sum([reward.reward for reward in self.yield_rewards()])

    def wallet_reset(self, ohlct):
        self.wallet.reset(ohlct)

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

    def rewards_info(self):
        print('RiskManager Rewards:', [str(reward) for reward in self.yield_rewards()])

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
               f"Balance={self.initial_balance}, " \
               f"atr_stop_losses={self.atr_stop_loss_ratios}, " \
               f"risk_reward_ratios={self.risk_reward_ratios}, " \
               f"trade_risk={self.trade_risk}, " \
               f"action_space_size={len(self.action_dict.keys())}>"
