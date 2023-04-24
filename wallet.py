from pprint import pprint

import Utils
import pandas as pd
from DataProcessing.timeframe import OHLCT
from positions import Position, Long, Short
from reward_system import RewardBuffer

template_position = Long(ticker='EURUSD',
                         open_time='0',
                         open_price=1,
                         stop_loss=0.99,
                         risk_reward_ratio=1.5,
                         position_risk=100)

dataframe_columns = list(template_position.__dict__.keys())
dataframe_columns = [column for column in dataframe_columns if not column.endswith("_history")]


class Wallet:
    def __init__(self,
                 ticker: str,
                 initial_balance: float,
                 reward_buffer: RewardBuffer):
        self.ticker = ticker.upper()
        self.initial_balance = initial_balance
        self.margin_balance = {"free": self.initial_balance, "margin": 0}
        self.position = None
        self.current_ohlct = None
        self.game_over = False
        self.transaction_reward = None
        self.intermediate_reward = None
        self.closed_positions = []
        self.history_dataframe = pd.DataFrame(columns=dataframe_columns, index=[])
        self.reward_buffer: RewardBuffer = reward_buffer
        self.total_trades = 0

    def reserve_margin(self, amount):
        amount = round(amount, 2)
        if amount > self.margin_balance['free']:
            self.game_over = True
        else:
            self.margin_balance['free'] -= amount
            self.margin_balance['margin'] = amount

    def free_margin(self):
        self.margin_balance['free'] += self.margin_balance['margin']
        self.margin_balance['free'] = round(self.margin_balance['free'], 2)
        self.margin_balance['margin'] = 0

    def open_long(self, stop_loss_delta: float, risk_reward_ratio: float, position_risk: float):
        self._prepare_to_open_position(close_position_type='short')  # Generates wallet reward
        self.position = Long(ticker=self.ticker,
                             open_time=self.current_ohlct.time,
                             open_price=self.current_ohlct.close,
                             stop_loss=self.current_ohlct.close - stop_loss_delta,
                             risk_reward_ratio=risk_reward_ratio,
                             position_risk=position_risk)
        self.reserve_margin(amount=self.position.margin)

    def open_short(self, stop_loss_delta: float, risk_reward_ratio: float, position_risk: float):
        self._prepare_to_open_position(close_position_type='long')  # Generates wallet reward
        self.position = Short(ticker=self.ticker,
                              open_time=self.current_ohlct.time,
                              open_price=self.current_ohlct.close,
                              stop_loss=self.current_ohlct.close + stop_loss_delta,
                              risk_reward_ratio=risk_reward_ratio,
                              position_risk=position_risk)
        self.reserve_margin(amount=self.position.margin)

    def position_close(self):
        self.position.is_closed = True
        self.position.close_price = self.current_ohlct.close
        self.position.close_time = self.current_ohlct.time
        self.margin_balance['free'] += self.position.profit
        self.free_margin()
        self.__update_dataframe(position=self.position)
        self.generate_rewards()
        self.position = None

    def generate_rewards(self):
        self.reward_buffer.reward_transaction(position=self.position)
        self.reward_buffer.reward_sharpe(returns=self.returns)
        self.reward_buffer.reward_drawdown(returns=self.returns)

    def update_wallet(self, ohlct: OHLCT):
        self.is_game_over()
        self.current_ohlct = ohlct
        self.update_position()
        self.check_and_close_position()

    def update_position(self):
        if self.position is not None:
            self.position.update_position(ohlct=self.current_ohlct)
            self.reward_buffer.reward_intermediate(position=self.position)

    @property
    def returns(self):
        return self.history_dataframe.profit.values

    def check_and_close_position(self):
        if self.position is not None and self.position.is_closed:
            self.margin_balance['free'] += self.position.profit
            self.free_margin()
            self.__update_dataframe(position=self.position)
            self.generate_rewards()
            self.position = None

    @property
    def state(self, include_balance=True):
        balance = self.total_balance / (self.initial_balance * 10)
        position_type = [0, 0] if self.position is None else [int(self.position.type == 'long'),
                                                              int(self.position.type == 'short')]
        position_state = self.position.state if self.position is not None else [0, 0, 0]

        if include_balance:
            return [balance, *position_type, *position_state]
        else:
            return [*position_type, *position_state]

    @property
    def cash(self) -> float:
        return self.margin_balance['free']

    @property
    def total_balance(self) -> float:
        return self.margin_balance['free'] + self.margin_balance['margin']

    @property
    def unrealized_profit(self):
        if self.position is None:
            return 0
        else:
            return self.position.unrealized_profit

    def info(self) -> None:
        if not self.game_over:
            Utils.printline(text='Wallet info', title=False, line_char=":", size=60)
            print(
                f"Free: {self.margin_balance['free']}$ "
                f"| Margin: {self.margin_balance['margin']}$ "
                f"| Total: {self.total_balance}$")
            if self.position:
                self.position.info()
                Utils.printline(text='', title=False, line_char=":", size=60, blank=True)
            else:
                Utils.printline(text='No opened positions', title=False, line_char=":", size=60)
        else:
            Utils.printline(text='GAME OVER', title=True, line_char="=")

    def reset(self, ohlct: OHLCT):
        self.position = None
        self.game_over = False
        self.margin_balance = {"free": self.initial_balance, "margin": 0}
        self.update_wallet(ohlct=ohlct)
        self.total_trades = 0
        self.closed_positions = []
        self.history_dataframe = pd.DataFrame(columns=dataframe_columns, index=[])

    def cancel_position(self):
        self.margin_balance['free'] += self.position.margin
        self.position = None

    def _prepare_to_open_position(self, close_position_type):
        if self.position is not None and self.position.type == close_position_type:
            self.position_close()

    def __update_dataframe(self, position: Position):
        string_args = ['open_time', 'close_time', 'type', 'ticker', 'order_number']

        position_dict = {argument: float(position.__dict__[argument]) for argument in dataframe_columns if
                         argument not in string_args}

        position_dict['open_time'] = position.open_time
        position_dict['close_time'] = position.close_time
        position_dict['type'] = position.type
        position_dict['ticker'] = position.ticker
        position_dict['order_number'] = position.order_number

        self.history_dataframe = pd.concat([self.history_dataframe,
                                            pd.DataFrame(position_dict, index=[0])],
                                           axis=0, ignore_index=True)
        self.closed_positions.append(position)
        self.total_trades += 1

    def __repr__(self):
        return f"<Wallet: margin_balance={self.margin_balance}>"

    def is_game_over(self):
        if self.total_balance < self.initial_balance / 10:
            self.game_over = True
