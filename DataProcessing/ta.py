import pandas_ta as ta
import pandas as pd
import numpy as np


class TechnicalIndicator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.ta = None

    def calculate(self):
        pass


class BBands(TechnicalIndicator):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe=dataframe)

    def calculate(self):
        self.ta = ta.bbands(close='close', length=14)


class ATR(TechnicalIndicator):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe=dataframe)

    def calculate(self):
        self.ta = ta.atr(high='high', low='low', close='close', length=14)


class EMA(TechnicalIndicator):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe=dataframe)

    def calculate(self):
        self.ta = ta.ema(close='close', length=20)


class CandlestickPatterns(TechnicalIndicator):
    def __init__(self):
        super().__init__()
        pass

    def calculate(self):
        pass
