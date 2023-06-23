from typing import Dict, Optional, List

import pandas as pd
from xtb import XTB


class ContinuousPosition:

    def __init__(self,
                 order_type: str,
                 ticker: str):
        self.ticker: str = ticker
        self.order_type: str = order_type.lower()
        self._order_number: Optional[str] = None

        # Continuous values
        self.profit = 0
        self.position_margin: float = 0
        self.total_volume = 0
        self.avg_price: float = 0
        self.contract_value: float = 0

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value = XTB[ticker]['one_lot_value']

    def check_position(self, current_price, volume):
        pass

    def modify_position(self, current_price: float, volume: float) -> float:
        """Modify the current ContinuousPosition.

        Given a current_price and volume position changes.
        If the volume is positive agent buys lots if negative he sells them.

        Parameters:
            current_price (float): Current price at selected step_size timeframe.
            volume (float): Amount of the lots the agent wants to acquire/sell.

        Returns:
            float: The amount to update the balance of the ContinuousEnvironment
        """
        volume = round(volume, 2)
        if volume > 0:  # Agent is increasing his biased position
            total_value = (self.total_volume * self.avg_price) + (volume * current_price)
            self.total_volume = round(self.total_volume + volume, 3)
            self.avg_price = round(total_value / self.total_volume, 4)

            self.contract_value = round(self.total_volume * self.one_lot_value, 2)
            margin = round((volume * self.one_lot_value) / self.leverage, 2)
            self.position_margin = round(self.position_margin + margin, 2)
            return -margin

        elif volume < 0:  # Agent is decreasing his biased position
            profit = self.calculate_profit(current_price=current_price, volume=abs(volume))
            margin_released = round((abs(volume) * self.one_lot_value) / self.leverage, 2)

            self.total_volume = round(self.total_volume + volume, 3)

            # If all shares are sold, average acquisition price becomes 0
            if self.total_volume != 0:
                self.position_margin = round(self.position_margin - margin_released, 2)
                self.contract_value = round(self.total_volume * self.one_lot_value, 2)
            else:
                self.avg_price = 0
                self.position_margin = 0
                self.contract_value = 0

            return margin_released + profit

        else:
            return 0

    def calculate_profit(self, current_price: float, volume: Optional[float] = None) -> int:
        volume = self.total_volume if volume is None else volume
        cur_value = current_price * volume * self.one_lot_value
        open_value = self.avg_price * volume * self.one_lot_value
        return round(cur_value - open_value, 2) if self.order_type == 'long' else round(open_value - cur_value, 2)

    def info(self) -> None:
        print(f'{self.order_type.capitalize()}: '
              f'Avg_price = {self.avg_price} '
              f'Volume = {self.total_volume} '
              f'Contract_value = {self.contract_value} '
              f'Position_margin = {self.position_margin}\n')

    def __repr__(self) -> str:
        return f"<{self.order_type.capitalize()}: " \
               f"avg_price={self.avg_price}, " \
               f"profit={self.profit or 0}>"

    @property
    def order_number(self) -> str:
        return self._order_number

    @order_number.setter
    def order_number(self, value) -> None:
        self._order_number = value

    def __dict__(self) -> dict:
        return {
            "order_type": self.order_type,
            "ticker": self.ticker,
            "profit": self.profit,
            "position_margin": self.position_margin,
            "total_volume": self.total_volume,
            "avg_price": self.avg_price
        }

    @property
    def state(self) -> List[float]:
        """Return position state.
        Returns:
            list: [profit, position_margin, total_volume]
        """
        return [self.profit, self.position_margin, self.total_volume]


class DiscretePosition:

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

        self.volume: float = round(self.contract_value / XTB[ticker]['one_lot_value'], 5)
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
