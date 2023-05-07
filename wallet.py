from typing import Union

import numpy as np
import Utils
import pandas as pd
from DataProcessing.ohlct import OHLCT
from positions import Position, Long, Short
from reward_buffer import RewardBuffer

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
        self.reward_buffer = reward_buffer
        self.max_balance = initial_balance
        self.ticker = ticker.upper()
        self.initial_balance = initial_balance
        self.margin_balance = {"free": self.initial_balance, "margin": 0}
        self.position: Union[Short, Long] = None
        self.current_ohlct: OHLCT = None
        self.game_over: bool = False
        self.closed_positions = []
        self.history_dataframe = pd.DataFrame(columns=dataframe_columns, index=[])
        self.total_trades = 0
        self.cost = 0.8

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
        self._prepare_to_open_position()
        self.position = Long(ticker=self.ticker,
                             open_time=self.current_ohlct.time,
                             open_price=self.current_ohlct.close,
                             stop_loss=self.current_ohlct.close - stop_loss_delta,
                             risk_reward_ratio=risk_reward_ratio,
                             position_risk=position_risk)
        self.reserve_margin(amount=self.position.margin)

    def open_short(self, stop_loss_delta: float, risk_reward_ratio: float, position_risk: float):
        self._prepare_to_open_position()
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
        self.free_margin()
        self.margin_balance['free'] += (self.position.get_real_profit() - self.cost)
        self.__update_dataframe(position=self.position)
        self.get_position_rewards()
        self.position = None

    def update_wallet(self, ohlct: OHLCT):
        self.is_game_over()
        self.current_ohlct = ohlct
        self.update_position()
        self.check_and_close_position()

    def update_position(self):
        if self.position is not None:
            self.position.update_position(ohlct=self.current_ohlct)

    @property
    def returns(self):
        self.history_dataframe[
            'total_balance'] = self.history_dataframe.profit.cumsum() + 10000 - self.history_dataframe.profit
        self.history_dataframe['returns'] = round(self.history_dataframe.profit / self.history_dataframe.total_balance,
                                                  4)
        return self.history_dataframe.returns.values

    def check_and_close_position(self):
        if self.position and self.position.is_closed:
            self.free_margin()
            self.margin_balance['free'] += (self.position.profit - self.cost)
            self.__update_dataframe(position=self.position)
            self.get_position_rewards()
            self.position = None

    def state(self, max_gain: float, trade_risk: float, include_balance: bool = False) -> list:
        long_position = 1 if self.position and self.position.type == 'long' else 0
        short_position = 1 if self.position and self.position.type == 'short' else 0
        position_profit = self.position.get_profit() if self.position else 0
        position_profit = position_profit / max_gain if position_profit > 0 else position_profit / trade_risk

        if include_balance:
            return [self.total_balance / 100000, short_position, long_position, position_profit]
        else:
            return [short_position, long_position, position_profit]

    @property
    def total_balance(self) -> float:
        return self.margin_balance['free'] + self.margin_balance['margin']

    def info(self, short=False) -> None:
        if short:
            print(self.__repr__())
        else:
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

    def _prepare_to_open_position(self):
        if self.position is not None:
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
        return f"<Wallet: margin_balance={self.margin_balance} " \
               f"position_profit={self.position.get_profit() if self.position else None}>"

    def evaluation(self, returns) -> dict:
        """
        Evaluate wallet performance
        :param returns: portfolio returns
        :return: dict with evaluation metrics
        """
        return {
            'wallet_sharpe_ratio': self.sharpe_ratio(returns=returns),
            'wallet_sortino_ratio': self.sortino_ratio(returns=returns),
            'wallet_maximum_drawdown': self.maximum_drawdown(returns=returns),
            'wallet_profit_factor': self.profit_factor(returns=returns),
            'wallet_average_trade_return': self.average_trade_return(returns=returns),
            'wallet_win_rate': self.win_rate(returns=returns)
        }

    def log_info(self, returns) -> dict:
        """
        Log wallet info and evaluation metrics
        :param returns: portfolio returns
        :return: dict with evaluation metrics and wallet
        info that consists of total balance and total trades
        """
        out = self.evaluation(returns)
        out.update({"wallet_balance": self.total_balance,
                    "wallet_total_trades": self.total_trades})
        return out

    def is_game_over(self):
        if self.total_balance < 0:
            self.game_over = True

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.01):
        """
        Calculate Sharpe ratio for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.
        - risk_free_rate (float): The risk-free rate of return.

        Returns:
        - sharpe (float): The Sharpe ratio of the trading strategy.
        """
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.01, mar=0.02):
        """
        Calculate Sortino ratio for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.
        - risk_free_rate (float): The risk-free rate of return.
        - mar (float): The minimum acceptable return.

        Returns:
        - sortino (float): The Sortino ratio of the trading strategy.
        """
        downside_returns = np.minimum(returns - mar, 0)
        downside_volatility = np.std(downside_returns)
        if downside_volatility.any() == 0:
            return 0.0
        sortino = (np.mean(returns) - risk_free_rate) / downside_volatility
        return sortino

    @staticmethod
    def maximum_drawdown(returns):
        """
        Calculate maximum drawdown for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.

        Returns:
        - mdd (float): The maximum drawdown of the trading strategy.
        """
        cum_returns = np.cumprod(1 + returns)
        max_drawdown = np.maximum.accumulate(cum_returns) - cum_returns
        mdd = np.max(max_drawdown)
        return -mdd

    @staticmethod
    def profit_factor(returns):
        """
        Calculate profit factor for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.

        Returns:
        - pf (float): The profit factor of the trading strategy.
        """
        profit = np.sum(returns[returns > 0])
        loss = np.sum(returns[returns < 0])
        if loss > 0:
            pf = profit / -loss
            return pf
        else:
            return profit

    @staticmethod
    def average_trade_return(returns):
        """
        Calculate average trade return for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.

        Returns:
        - atr (float): The average trade return of the trading strategy.
        """
        atr = np.mean(returns)
        return atr

    @staticmethod
    def win_rate(returns):
        """
        Calculate win rate for a trading strategy.

        Parameters:
        - returns (numpy array): Array of returns for the trading strategy.

        Returns:
        - wr (float): The win rate of the trading strategy.
        """
        wins = np.sum(returns > 0)
        total_trades = len(returns)
        wr = wins / total_trades
        return wr

    def position_transaction_reward(self):
        if self.position and self.position.is_closed:
            return self.position.profit
        else:
            return 0

    def get_position_rewards(self):
        self.reward_buffer.add_reward(reward_name='transaction',
                                      reward_value=self.position_transaction_reward())
