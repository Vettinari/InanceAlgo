import operator
from typing import Callable, List

import Utils
from DataProcessing.timeframe import OHLCT

XTB = {"EURUSD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip_size": 0.0001,
                  "spread": 0.00008},
       "EURCAD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURCHF": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURGBP": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.0001},
       "EURJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       # TODO
       "USDJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPUSD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "AUDUSD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "USDCAD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "USDCHF": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "AUDJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "EURAUD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPAUD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPCAD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "EURNZD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "AUDCAD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPCHF": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "AUDCHF": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "CADJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "GBPNZD": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01},
       "CHFJPY": {"one_lot_value": 100000,
                  "leverage": 30,
                  "one_pip": 0.01}}


class Position:

    def __init__(self, ticker, open_time, open_price, position_risk):
        self.ticker = ticker
        self.open_time = open_time
        self.open_price = open_price
        self.one_pip_size = XTB[ticker]['one_pip_size']
        self.leverage = XTB[ticker]['leverage']
        self.position_risk = position_risk
        self.profit = 0
        # Closed
        self.close_time = None
        self.close_price = None
        # Flags
        self.is_closed = False
        self.is_stop_profit = False
        self.is_stop_loss = False
        # Type specific
        self.stop_loss = 0
        self.stop_profit = 0
        self.volume = 0
        self.margin = 0
        self.contract_value = 0
        self.risk_reward_ratio = 0
        # Broker specific
        self.order_number = None
        self.type = "Abstract"
        # TrendReward calculations
        self.steps_count = 0

    def update_position(self, ohlct: OHLCT) -> None:
        pass

    @property
    def unrealized_profit(self):
        if self.is_stop_profit is False and self.is_stop_loss is False and self.is_closed is False:
            return self.profit
        return 0

    def update_profit(self, current_price) -> None:
        pass

    def info(self):
        transaction_type = str(self.__class__.__name__).upper()
        Utils.printline(text=f'{transaction_type} : OPEN',
                        size=60, line_char="-", blank=False, title=False, test=True)
        print(f'Order number: {self.order_number}')
        print(f'Open at: {self.open_time} - 1:{self.leverage} leverage.')
        print(f'Open price: {self.open_price}$ - Volume: {self.volume}')
        print(f'Stop loss: {self.stop_loss}$ - Stop profit: {self.stop_profit}$ - RR: {self.risk_reward_ratio}x')
        print(f'Unrealized profit: {self.unrealized_profit}$ - Position risk: {self.position_risk}$')
        print(f'Margin: {self.margin}$ - Contract value: {self.contract_value}$')
        Utils.printline(text="", size=60, line_char="-", blank=True, title=False, test=True)

    def get_profit(self) -> float:
        return round(self.profit, 5)

    def set_order_number(self, new_number):
        self.order_number = new_number

    def get_order_number(self):
        return self.order_number

    def set_open_price(self, new_price):
        self.open_price = new_price
        self.update_profit(current_price=new_price)

    def _stop_loss_hit(self, ohlct: OHLCT, comparator: Callable) -> bool:
        return any(
            comparator(ohlct.__getattribute__(key), self.stop_loss) for key in ['open', 'high', 'low', 'close']
        )

    def _stop_profit_hit(self, ohlct: OHLCT, comparator: Callable) -> bool:
        return any(
            comparator(ohlct.__getattribute__(key), self.stop_profit) for key in ['open', 'high', 'low', 'close']
        )

    @property
    def state(self) -> List[float]:
        """
        :return: [stop_loss_level, stop_profit level, unrealized_profit]
        """
        return [self.stop_loss, self.stop_profit, self.unrealized_profit]

    def __repr__(self):
        return f"<{self.type.capitalize()} Position: open_price={self.open_price}, " \
               f"stop_loss={self.stop_loss}, " \
               f"stop_profit={self.stop_profit}>"


class Long(Position):
    def __init__(self, ticker, open_time, open_price, stop_loss, risk_reward_ratio, position_risk=100):
        super().__init__(ticker=ticker, open_time=open_time, open_price=open_price, position_risk=position_risk)
        self.stop_loss = stop_loss
        self.position_risk = position_risk

        # Calculated
        self.risk_reward_ratio = risk_reward_ratio
        self.position_gain = self.position_risk * risk_reward_ratio
        self.stop_profit = round(open_price + risk_reward_ratio * (open_price - stop_loss), 5)

        self.volume = round(
            (position_risk / abs(open_price - stop_loss)) * open_price / XTB[self.ticker]['one_lot_value'], 2)

        self.margin = round(self.volume * XTB[self.ticker]['one_lot_value'] / XTB[self.ticker]['leverage'], 2)
        self.contract_value = int(self.margin * self.leverage)
        self.type = 'long'

    def update_position(self, ohlct: OHLCT) -> None:

        # CHECK CONTINUITY
        if self.__stop_loss_hit(ohlct):
            self.is_closed = True
            self.is_stop_loss = True
            self.close_price = self.stop_loss
            self.close_time = ohlct.time
            self.profit = -self.position_risk
        else:
            if self.__stop_profit_hit(ohlct):
                self.is_closed = True
                self.is_stop_profit = True
                self.close_price = self.stop_profit
                self.close_time = ohlct.time
                self.profit = self.position_gain
            else:
                self.update_profit(current_price=ohlct.close)

    def __stop_loss_hit(self, ohlct: OHLCT) -> bool:
        return super()._stop_loss_hit(ohlct, operator.le)

    def __stop_profit_hit(self, ohlct: OHLCT) -> bool:
        return super()._stop_profit_hit(ohlct, operator.ge)

    def update_profit(self, current_price) -> None:
        eur_usd_pip_value = 1 / current_price
        pip_trade_value = XTB["EURUSD"]['one_lot_value'] * self.volume * self.one_pip_size * eur_usd_pip_value
        pips = (current_price - self.open_price) / self.one_pip_size
        self.profit = round(pips * pip_trade_value, 2)


class Short(Position):
    def __init__(self, ticker, open_time, open_price, stop_loss, risk_reward_ratio, position_risk=100):
        super().__init__(ticker=ticker, open_time=open_time, open_price=open_price, position_risk=position_risk)
        self.stop_loss = stop_loss
        self.position_risk = position_risk

        # Calculated
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_profit = round(open_price - risk_reward_ratio * (stop_loss - open_price), 5)
        self.position_gain = self.position_risk * risk_reward_ratio

        self.volume = round(
            (position_risk / abs(open_price - stop_loss)) * open_price / XTB[self.ticker]['one_lot_value'], 2)

        self.margin = round(self.volume * XTB[self.ticker]['one_lot_value'] / XTB[self.ticker]['leverage'], 2)
        self.contract_value = int(self.margin * self.leverage)
        self.type = 'short'

    def update_position(self, ohlct: OHLCT) -> None:

        # CHECK CONTINUITY
        if self.__stop_loss_hit(ohlct):
            self.is_closed = True
            self.is_stop_loss = True
            self.close_price = self.stop_loss
            self.close_time = ohlct.time
            self.profit = -self.position_risk
        else:
            if self.__stop_profit_hit(ohlct):
                self.is_closed = True
                self.is_stop_profit = True
                self.close_price = self.stop_profit
                self.close_time = ohlct.time
                self.profit = self.position_gain
            else:
                self.update_profit(current_price=ohlct.close)

    def __stop_loss_hit(self, ohlct: OHLCT) -> bool:
        return super()._stop_loss_hit(ohlct, operator.ge)

    def __stop_profit_hit(self, ohlct: OHLCT) -> bool:
        return super()._stop_profit_hit(ohlct, operator.le)

    def update_profit(self, current_price) -> None:
        eur_usd_pip_value = 1 / current_price
        pip_trade_value = XTB["EURUSD"]['one_lot_value'] * self.volume * self.one_pip_size * eur_usd_pip_value
        pips = (self.open_price - current_price) / self.one_pip_size
        self.profit = round(pips * pip_trade_value, 2)
