from typing import Dict, Optional

import pandas as pd

one_lot_contract_value = 100000


class Position:

    def __init__(self,
                 order_type: str,
                 ticker: str,
                 open_time: pd.Timestamp,
                 open_price: float,
                 stop_loss_pips: int,
                 stop_profit_pips: int,
                 leverage: int,
                 one_pip: float,
                 position_margin: int):
        self.true_profit = 0
        self.order_type: str = order_type.lower()
        self.ticker: str = ticker
        self.open_time: pd.Timestamp = open_time
        self.open_price: float = open_price
        self.leverage: int = leverage
        self.margin: float = position_margin
        self.contract_value: float = self.leverage * position_margin
        self.one_pip: float = one_pip
        self.closed: bool = False
        self.realized_profit = 0
        self._order_number: Optional[str] = None

        # Calculated
        self.volume: float = round(self.contract_value / one_lot_contract_value, 5)
        self._stop_loss_pips = stop_loss_pips
        self._stop_profit_pips = stop_profit_pips
        self._set_stop_loss()
        self._set_stop_profit()

    def current_profit(self, current_price) -> int:
        return int((current_price - self.open_price) / self.one_pip) if self.order_type == 'long' else int(
            (self.open_price - current_price) / self.one_pip)

    def set_order_number(self, new_number):
        self._order_number = new_number

    def _set_stop_loss(self):
        # print("TEST_SL", self._stop_loss_pips)
        # print("TEST_SL", self.one_pip)

        self.stop_loss = self.open_price - (
                self._stop_loss_pips * self.one_pip) if self.order_type == 'long' else self.open_price + (
                self._stop_loss_pips * self.one_pip)

    def _set_stop_profit(self):
        # print("TEST_SP", self._stop_profit_pips)
        # print("TEST_SP", self.one_pip)

        self.stop_profit = self.open_price + (
                self._stop_profit_pips * self.one_pip) if self.order_type == 'long' else self.open_price - (
                self._stop_profit_pips * self.one_pip)

    def check_stops(self, ohlc_dict: Dict[str, float]) -> bool:
        if not all(isinstance(v, float) for v in ohlc_dict.values()):
            raise ValueError("Invalid ohlc_dict values")

        def _check_stop(stop, comparison):
            return any(comparison(value, stop) for value in ohlc_dict.values())

        hit_stop_loss = _check_stop(self.stop_loss, lambda x, y: x <= y if self.order_type == 'long' else x >= y)
        hit_stop_profit = _check_stop(self.stop_profit, lambda x, y: x >= y if self.order_type == 'long' else x <= y)

        if hit_stop_loss or hit_stop_profit:
            self.realized_profit = self.current_profit(
                current_price=(self.stop_loss if hit_stop_loss else self.stop_profit))
            print("stop_loss" if hit_stop_loss else "stop_profit")
            return True
        return False

    def terminate_order(self, current_price):
        profit = self.current_profit(current_price=current_price)
        if profit <= self._stop_loss_pips:
            self.realized_profit = -self._stop_loss_pips
        elif profit >= self._stop_profit_pips:
            self.realized_profit = self._stop_profit_pips

    def info(self):
        print(f'{self.order_type.capitalize()}: '
              f'Lots = {self.volume} '
              f'Contract_value = {self.contract_value} '
              f'Position_margin = {self.margin}')

    def __repr__(self):
        return f"<{self.order_type.capitalize()}: " \
               f"open_price={self.open_price}, " \
               f"stop_loss={self.stop_loss}, " \
               f"stop_profit={self.stop_profit}, " \
               f"realized_pip_profit={self.realized_profit or 0}>"

    @property
    def order_number(self):
        return self._order_number

    @order_number.setter
    def order_number(self, value):
        self._order_number = value

    def __dict__(self) -> dict:
        return {
            "order_type": self.order_type,
            "ticker": self.ticker,
            "open_time": self.open_time,
            "open_price": self.open_price,
            "realized_profit": self.realized_profit,
            "volume": self.volume,
        }
