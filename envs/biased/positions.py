from typing import Optional, List
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


class ContinuousPosition:

    def __init__(self,
                 order_type: str,
                 ticker: str,
                 scaler: float):
        self.scaler: float = scaler
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
        volume = round(volume, 2)  # Round down to 2 decimal places
        if volume > 0:  # Agent is increasing his biased position
            total_value = (self.total_volume * self.avg_price) + (volume * current_price)
            self.total_volume = round(self.total_volume + volume, 3)
            self.avg_price = round(total_value / self.total_volume, 5)

            self.contract_value = round(self.total_volume * self.one_lot_value, 2)

            margin = self.required_margin(volume=volume)
            self.position_margin = round(self.position_margin + margin, 2)
            return round(-margin, 2)  # returns the margin required to open the position

        elif volume < 0:  # Agent is decreasing his biased position
            if volume < -self.total_volume:  # If agent wants to sell more than he has
                volume = -self.total_volume

            profit = self.trade_profit(current_price=current_price, volume=abs(volume))
            margin_released = round((abs(volume) * self.one_lot_value) / self.leverage, 2)

            self.total_volume = round(self.total_volume + volume, 3)

            # If all shares are sold, average acquisition price becomes 0
            if self.total_volume > 0:
                self.position_margin = round(self.position_margin - margin_released, 2)
                self.contract_value = round(self.total_volume * self.one_lot_value, 2)
            else:
                self.avg_price = 0
                self.position_margin = 0
                self.contract_value = 0

            return round(margin_released + profit, 2)  # returns the margin released + profit
        # Return 0 as no changes were made
        return 0

    def required_margin(self, volume) -> float:
        """Return the required margin to open a position calculation based on the volume passed.
        Returns:
            float: The required margin to open a position.
        """

        return round((volume * self.one_lot_value) / self.leverage, 2)

    def pip_profit(self, current_price: float) -> float:
        """Return the profit in pips.
        Returns:
            float: The profit in pips.
        """
        out = (current_price - self.avg_price) / self.one_pip \
            if self.order_type == 'long' else (self.avg_price - current_price) / self.one_pip
        return round(out, 3)

    def trade_profit(self, current_price: float, volume: float) -> float:
        """Returns real trade profit.
        Returns:
            float: Trade profit in currency.
        """
        calculated_value = current_price * volume * self.one_lot_value  # Calculate the full contract value
        open_value = self.avg_price * volume * self.one_lot_value  # Calculate the open contract value
        return round(calculated_value - open_value, 2) if self.order_type == 'long' else round(
            open_value - calculated_value, 2)

    def total_position_profit(self, current_price: float) -> float:
        """Returns real total profit.
        Returns:
            float: Total profit in currency.
        """
        delta = round((current_price - self.avg_price) * self.total_volume * self.one_lot_value, 2)
        return delta if self.order_type == 'long' else -delta

    def info(self) -> None:
        """
        Print position info.
        """
        print(f'{self.order_type.capitalize()}: '
              f'Avg_price = {self.avg_price} '
              f'Volume = {self.total_volume} '
              f'Contract_value = {self.contract_value} '
              f'Position_margin = {self.position_margin}\n')

    def state(self, current_price: float, scaled: bool = True) -> List[float]:
        """Return position state.
        Returns:
            list: [profit, position_margin, total_volume, avg_price]
        """
        if scaled:
            return [self.pip_profit(current_price=current_price) * self.scaler,
                    self.position_margin * self.scaler,
                    self.total_volume * self.scaler]
        else:
            return [self.pip_profit(current_price=current_price),
                    self.position_margin,
                    self.total_volume]

    def log_info(self, current_price: float) -> dict:
        """
        Return position as dictionary.
        Returns:
            dict of all position arguments that are important.
        """
        return {
            "total_position_profit": self.total_position_profit(current_price=current_price),
            "position_margin": self.position_margin,
            "total_volume": self.total_volume,
            "avg_price": self.avg_price,
            "position_type": self.order_type
        }

    def validate_action(self, action: float, cash_in_hand: float) -> bool:
        sell_zero_flag = action < 0 and self.total_volume == 0
        buy_without_enough_cash_flag = cash_in_hand < self.required_margin(volume=action)
        sell_more_than_own_flag = action < 0 and abs(action) > self.total_volume
        return sell_zero_flag or buy_without_enough_cash_flag or sell_more_than_own_flag
