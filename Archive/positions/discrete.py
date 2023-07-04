from pprint import pprint
from typing import Dict, List, Optional
from Utils.xtb import XTB


class DiscretePosition:

    def __init__(self,
                 ticker: str,
                 scaler: float,
                 stop_loss_pips: int,
                 stop_profit_pips: int,
                 risk: int,
                 manual_close: bool = False):
        self.scaler: float = scaler
        self.ticker: str = ticker
        self._stop_loss_pips: int = stop_loss_pips
        self._stop_profit_pips: int = stop_profit_pips
        self.manual_close: bool = manual_close
        self.risk: float = risk

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value: int = XTB[ticker]['one_lot_value']

        # Position dynamic values
        self.stop_loss: float = None
        self.stop_profit: float = None
        self.open_price: float = None
        self.position_type: str = None
        self.position_margin: float = None
        self.volume: float = None
        self.contract_value: float = None
        self.status: str = None
        self.trade_profit: float = None
        self.pips_profit: int = None

    def calculate_pip_profit(self, current_price: float) -> int:
        """Returns the profit in pips given a current_price."""
        out = (current_price - self.open_price) / self.one_pip if self.position_type == 'long' else \
            (self.open_price - current_price) / self.one_pip

        return int(out)

    def calculate_trade_profit(self, current_price: float) -> float:
        """Returns the profit in currency given a current_price."""
        current_value = current_price * self.volume * self.one_lot_value
        acquire_value = self.open_price * self.volume * self.one_lot_value
        return current_value - acquire_value if self.position_type == 'long' else acquire_value - current_value

    def check_stops(self, ohlc_dict: Dict[str, float]) -> Optional[dict]:
        """Returns True and position history(dict) if the current price hits any stop."""

        def _check_stop(stop, comparison):
            return any(comparison(value, stop) for value in ohlc_dict.values())

        # Check stop loss
        hit_stop_loss = _check_stop(self.stop_loss, lambda x, y: x <= y if self.position_type == 'long' else x >= y)
        if hit_stop_loss:
            print("stop_loss")
            self.status = 'stop_loss'
            self.pips_profit = self._stop_loss_pips
            self.trade_profit = self.risk
            history = self.close_position()
            return history

        # Check stop profit
        hit_stop_profit = _check_stop(self.stop_profit, lambda x, y: x >= y if self.position_type == 'long' else x <= y)
        if hit_stop_profit:
            print("stop_profit")
            self.status = 'stop_profit'
            self.pips_profit = self._stop_profit_pips
            self.trade_profit = self.risk * (self._stop_profit_pips / self._stop_loss_pips)
            history = self.close_position()
            return history

        return None

    def close_position(self, manual_close_price: Optional[bool] = None) -> Dict[str, float]:
        if manual_close_price:
            self.status = "manual_close"
            self.pips_profit = self.calculate_pip_profit(current_price=manual_close_price)
            self.trade_profit = self.calculate_trade_profit(current_price=manual_close_price)

        out = self.__dict__().copy()
        self.reset_position()
        return out

    def open_position(self, open_price: float, position_type: str) -> float:
        """
        Opens a position and returns margin that need to be reserved. Calculation of margin is based on the risk.
        :param open_price: Current price
        :param position_type: 'short' or 'long'
        :return: Returns margin that need to be reserved.
        """
        self.status = 'open'
        self.open_price = open_price
        self.position_type = position_type
        self._set_stop_loss()
        self._set_stop_profit()
        self.position_margin = self.required_margin(current_price=open_price)
        return self.position_margin

    def required_margin(self, current_price: float) -> float:
        """
        Calculates the required margin for a position given a current_price.
        :param current_price: Current price
        :return: Returns the required margin.
        """
        stop_loss = current_price - self._stop_loss_pips * self.one_pip if self.position_type == 'long' else \
            current_price + self._stop_loss_pips * self.one_pip
        self.volume = round(
            (self.risk / ((current_price * self.one_lot_value) - (stop_loss * self.one_lot_value))) * current_price, 2)
        self.contract_value = round(self.volume * self.one_lot_value, 2)
        self.position_margin = round(self.contract_value / self.leverage, 2)
        return self.position_margin

    def info(self):
        pprint(self.__dict__())

    def _set_stop_loss(self):
        self.stop_loss = self.open_price - (
                self._stop_loss_pips * self.one_pip) if self.position_type == 'long' else self.open_price + (
                self._stop_loss_pips * self.one_pip)

    def _set_stop_profit(self):
        self.stop_profit = self.open_price + (
                self._stop_profit_pips * self.one_pip) if self.position_type == 'long' else self.open_price - (
                self._stop_profit_pips * self.one_pip)

    def reset_position(self):
        self.stop_loss: float = None
        self.stop_profit: float = None
        self.open_price: float = None
        self.position_type: str = None
        self.position_margin: float = None
        self.volume: float = None
        self.contract_value: float = None
        self.status: str = None
        self.trade_profit: float = None
        self.pips_profit: int = None

    def state(self, current_price: float) -> Optional[List[float]]:
        """
        Returns the state of the position given a current_price. If manual_close is True, it will return the profit
        and other metrics for agent to learn. Otherwise, it will return None as the environment will handle the closing.
        :param current_price:
        :return:
        """
        if self.manual_close:
            return [
                self.calculate_pip_profit(current_price=current_price)
            ]
        else:
            return

    def __dict__(self) -> Dict[str, float]:
        return {
            'ticker': self.ticker,
            'open_price': self.open_price,
            'position_type': self.position_type,
            'stop_loss': self.stop_loss,
            'stop_profit': self.stop_profit,
            'margin': self.position_margin,
            'volume': self.volume,
            'contract_value': self.contract_value,
            'pip_profit': self.pips_profit,
            'trade_profit': self.trade_profit,
            'status': self.status
        }
