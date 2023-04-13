import enum
from datetime import timedelta
from typing import Optional

import pandas as pd

from DataProcessing import ta


class OHLCT:
    def __init__(self, dataframe_row):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.time = dataframe_row.name

    def __repr__(self):
        return f'Open: {self.open}, High: {self.high}, Low: {self.low}, Close: {self.close}, Time: {self.time}'


class DataProcessor:
    def __init__(self,
                 ticker: str,
                 intervals: Optional[list] = None,
                 technical_indicators: Optional[list] = None):
        self.ticker = ticker
        self.intervals = [15, 60, 120] if intervals is None else intervals
        self.dataframe: pd.DataFrame = None
        self.technical_dataframe: pd.DataFrame = None
        self.current_step = 0
        self.total_steps = -999
        self.load_data()
        self.clean_data()

    def load_data(self, path: Optional[str] = None):
        path = f'Datasets/forex/intraday/{self.ticker}.csv' if path is None else path
        self.dataframe = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
        self.dataframe = self.dataframe.groupby(
            pd.Grouper(freq=f'{min(self.intervals)}T')).agg({'open': 'first',
                                                             'close': 'last',
                                                             'low': 'min',
                                                             'high': 'max',
                                                             'volume': 'sum'})
        self.intervals.pop(0)
        self.total_steps = self.dataframe.shape[0]

    def clean_data(self):
        pass

    def generate_technical_indicators(self):
        pass

    def ohlct(self, current_step) -> OHLCT:
        return OHLCT(dataframe_row=self.dataframe.iloc[current_step])

    def info(self):
        print(f'{self.ticker} - Start: {self.dataframe.iloc[0].name} - End: {self.dataframe.iloc[-1].name}')
        print(f'Total steps: {self.total_steps} - Intervals: {self.intervals}')

    def get_state(self, current_step):
        pass
