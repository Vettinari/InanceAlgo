from typing import Optional, List
from xtb import XTB


class ContinuousPosition:

    def __init__(self,
                 ticker: str,
                 scaler: float):
        self.scaler: float = scaler
        self.ticker: str = ticker
        self._order_number: Optional[str] = None

        # Continuous values
        self.profit = 0
        self.position_margin: float = 0
        self.total_volume = 0
        self.avg_price: float = 0
        self.contract_value: float = 0
        self.current_bias = 'neutral'

        # Calculated
        self.leverage: int = XTB[ticker]['leverage']
        self.one_pip: float = XTB[ticker]['one_pip']
        self.one_lot_value = XTB[ticker]['one_lot_value']

    # def modify_position(self, current_price: float, volume: float) -> float:
    #     """Modify the current ContinuousPosition.
    #
    #     Given a current_price and volume position changes.
    #     If the volume is positive agent buys lots if negative he sells them.
    #
    #     Parameters:
    #         current_price (float): Current price at selected step_size timeframe.
    #         volume (float): Amount of the lots the agent wants to acquire/sell.
    #
    #     Returns:
    #         float: The amount to update the balance of the ContinuousEnvironment
    #     """
    #     volume = round(volume, 2)  # Round down to 2 decimal places
    #
    #     if volume > 0:  # Agent is increasing his biased position
    #         total_value = (self.total_volume * self.avg_price) + (volume * current_price)
    #         self.total_volume = round(self.total_volume + volume, 3)
    #         self.avg_price = round(total_value / self.total_volume, 5)
    #
    #         self.contract_value = round(self.total_volume * self.one_lot_value, 2)
    #
    #         margin = self.required_margin(volume=volume)
    #         self.position_margin = round(self.position_margin + margin, 2)
    #         return round(-margin, 2)  # returns the margin required to open the position
    #
    #     elif volume < 0:  # Agent is decreasing his biased position
    #         if volume < -self.total_volume:  # If agent wants to sell more than he has
    #             volume = -self.total_volume
    #
    #         profit = self.trade_profit(current_price=current_price, volume=abs(volume))
    #         margin_released = round((abs(volume) * self.one_lot_value) / self.leverage, 2)
    #
    #         self.total_volume = round(self.total_volume + volume, 3)
    #
    #         # If all shares are sold, average acquisition price becomes 0
    #         if self.total_volume > 0:
    #             self.position_margin = round(self.position_margin - margin_released, 2)
    #             self.contract_value = round(self.total_volume * self.one_lot_value, 2)
    #         else:
    #             self.avg_price = 0
    #             self.position_margin = 0
    #             self.contract_value = 0
    #
    #         return round(margin_released + profit, 2)  # returns the margin released + profit
    #     # Return 0 as no changes were made
    #     return 0

    def modify_position(self, current_price: float, volume: float):
        """
        Modify the current ContinuousPosition.
        """

        delta_volume = round(volume + self.total_volume, 2)
        if delta_volume < 0 and self.current_bias != 'short':
            self.current_bias = 'short'
        elif delta_volume > 0 and self.current_bias != 'long':
            self.current_bias = 'long'
        else:
            self.current_bias = 'neutral'

        # if volume was negative then we need to sell the shorts and buy longs
        if self.total_volume < 0 < delta_volume:
            # Liquidate the shorts and buy longs
            pass
        elif self.total_volume > 0 > delta_volume:
            # Liquidate the longs and buy shorts
            pass
        elif self.total_volume < 0 and delta_volume < 0:
            # Add shorts
            pass
        elif self.total_volume > 0 and delta_volume > 0:
            # Add longs
            pass


    # if volume > 0:  # Agent is increasing his biased position
    #     total_value = (self.total_volume * self.avg_price) + (volume * current_price)
    #     self.total_volume = round(self.total_volume + volume, 3)
    #     self.avg_price = round(total_value / self.total_volume, 5)
    #
    #     self.contract_value = round(self.total_volume * self.one_lot_value, 2)
    #
    #     margin = self.required_margin(volume=volume)
    #     self.position_margin = round(self.position_margin + margin, 2)
    #     return round(-margin, 2)  # returns the margin required to open the position
    #
    # elif volume < 0:  # Agent is decreasing his biased position
    #     if volume < -self.total_volume:  # If agent wants to sell more than he has
    #         volume = -self.total_volume
    #
    #     profit = self.trade_profit(current_price=current_price, volume=abs(volume))
    #     margin_released = round((abs(volume) * self.one_lot_value) / self.leverage, 2)
    #
    #     self.total_volume = round(self.total_volume + volume, 3)
    #
    #     # If all shares are sold, average acquisition price becomes 0
    #     if self.total_volume > 0:
    #         self.position_margin = round(self.position_margin - margin_released, 2)
    #         self.contract_value = round(self.total_volume * self.one_lot_value, 2)
    #     else:
    #         self.avg_price = 0
    #         self.position_margin = 0
    #         self.contract_value = 0
    #
    #     return round(margin_released + profit, 2)  # returns the margin released + profit
    #     # Return 0 as no changes were made
    # return 0

    def sell_position(self, current_price: float, volume: float) -> float:
        pass

    def buy_position(self, current_price: float, volume: float) -> float:
        pass

    def flip_bias(self):
        self.current_bias = 'long' if self.current_bias == 'short' else 'short'

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


class ContinuousPositionBiased:

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
