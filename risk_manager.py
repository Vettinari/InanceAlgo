from pprint import pprint
from typing import Optional

from DataProcessing.data_processor import OHLCT
from reward_system import ActionReward, TransactionReward, IntermediateReward
from wallet import Wallet


class Action:
    def __init__(self, action: str, sl: Optional[float], rr: Optional[float]):
        self.action = action
        self.sl = sl
        self.rr = rr

    def __repr__(self):
        return f"Action({self.action}, sl: {self.sl}, rr: {self.rr})"


class RiskManager:
    def __init__(self, ticker: str, initial_balance: float, atr_stop_loss_ratios: list,
                 risk_reward_ratios: list = None, manual_position_closing: bool = True,
                 portfolio_risk: float = None, stop_loss_type='atr', wallet=None):
        self.initial_balance = initial_balance
        self.atr_stop_loss_ratios = [2,
                                     3] if atr_stop_loss_ratios is None else atr_stop_loss_ratios  # ATR multiplier/ percent
        self.risk_reward_ratios = [1, 2, 3] if risk_reward_ratios is None else risk_reward_ratios  # [1,2,3,4,5]
        self.trade_risk = initial_balance * 0.02 if portfolio_risk is None else initial_balance * portfolio_risk
        self.manual_position_closing = manual_position_closing
        self.wallet = Wallet(ticker=ticker, initial_balance=initial_balance) if wallet is None else wallet
        self.action_dict = self.generate_action_space()
        self.stop_loss_type = stop_loss_type
        # Rewards
        self.action_reward: ActionReward = ActionReward(reward=0)
        self.transaction_reward: TransactionReward = TransactionReward()
        self.intermediate_reward: IntermediateReward = IntermediateReward(position=None, scaling_factor=None)

    def generate_action_space(self):
        actions = ['long', 'short', 'hold']
        action_space = []

        for action in actions:
            if action in ['long', 'short']:
                for risk_reward in self.risk_reward_ratios:
                    for sl in self.atr_stop_loss_ratios:
                        action_space.append(Action(action=action, sl=sl, rr=risk_reward))
            else:
                action_space.append(Action(action='hold', sl=None, rr=None))

        if self.manual_position_closing:
            action_space.append(Action(action='close', sl=None, rr=None))

        action_space = dict(zip(range(0, len(action_space)), action_space))

        return action_space

    def validate_action(self, action: str):
        if self.wallet.position is not None and self.wallet.position.type == action:
            self.action_reward = ActionReward(reward=-1)
            return False  # if current action is the same as the position open

        elif self.wallet.position is None and action == 'close':
            self.action_reward = ActionReward(reward=-1)
            return False  # if current action is close and no positions are open

        self.action_reward = ActionReward(reward=0.1)
        return True

    def execute_action(self, action_index: int, current_atr):
        action = self.action_dict[action_index]
        validation = self.validate_action(action=action.action)

        if validation:
            if action.action == "hold":
                pass
            elif action.action == "close":
                self.wallet.position_close()
            elif action.action == "long":
                stop_loss_delta = round(current_atr * action.sl, 5)
                self.wallet.open_long(stop_loss_delta=stop_loss_delta,
                                      risk_reward_ratio=action.rr,
                                      position_risk=self.trade_risk)
            elif action.action == "short":
                stop_loss_delta = round(current_atr * action.sl, 5)
                self.wallet.open_short(stop_loss_delta=stop_loss_delta,
                                       risk_reward_ratio=action.rr,
                                       position_risk=self.trade_risk)

        self.transaction_reward = self.wallet.transaction_reward
        self.intermediate_reward = self.wallet.intermediate_reward

    def yield_rewards(self, objects=True):
        if objects:
            return [self.action_reward, self.transaction_reward, self.intermediate_reward]
        else:
            return self.action_reward.reward, self.transaction_reward.reward, self.intermediate_reward.reward

    def calculate_stop_loss_delta(self, sl_value, current_atr):
        return

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
        print('Action space:', len(self.generate_action_space()))
        pprint(self.generate_action_space())

    def rewards_info(self):
        print('RiskManager Rewards:', [str(reward) for reward in self.yield_rewards(True)])
