import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Union, List, Optional

from DataProcessing.timeframe import TimeFrame, OHLCT


class DataPipeline:
    def __init__(self, ticker: str, intervals: list, return_window: int, chart_window: int,
                 test: Optional[bool] = None):
        self.ticker: str = ticker
        self.step_size: int = min(intervals)
        self.intervals: list = sorted(intervals)
        self.return_window = return_window
        self.timeframes: Dict[int, TimeFrame] = {interval: TimeFrame(interval=interval,
                                                                     return_window=return_window,
                                                                     chart_window=chart_window)
                                                 for interval in intervals}
        self.current_step = None
        self.current_date = None
        self.dataframe: pd.DataFrame = None
        self.window = max(self.get_window_size(), chart_window)
        self.data_window = 1
        # Init data pipeline
        self.test = False if test is None else test
        self._load_csv_data()
        self._clean_dataframe()
        self.reset_step()

    def process_data(self):
        window_data = self.dataframe.loc[:self.current_date].iloc[-self.window - 1:-1]
        for interval, timeframe in self.timeframes.items():
            timeframe.process_window_data(window_data=window_data)

    def current_data(self, interval: Optional[int] = None) -> Union[TimeFrame, Dict[int, TimeFrame]]:
        if interval is not None:
            return self.timeframes[interval]
        return self.timeframes

    def reset_step(self):
        self.current_step = int(max(self.intervals) / self.step_size * self.window)
        self.current_date = self.dataframe.index[self.current_step]
        self.process_data()

    def step_forward(self):
        self.current_step += 1
        self.current_date = self.dataframe.index[self.current_step]
        self.process_data()

    def _load_csv_data(self):
        self.step_size = min(self.intervals)
        path = f'/Users/milosz/Documents/Pycharm/InanceAlgo/Datasets/forex/intraday/{self.ticker}.csv'
        if self.test:
            path = '/Users/milosz/Documents/Pycharm/InanceAlgo/EURUSD_short.csv'
        df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
        df = df.groupby(pd.Grouper(freq=f'{self.step_size}T')).agg({'open': 'first',
                                                                    'close': 'last',
                                                                    'low': 'min',
                                                                    'high': 'max',
                                                                    'volume': 'sum'})
        self.dataframe = df

    def _clean_dataframe(self):
        # Adjust to Amsterdam time
        self.dataframe.index = self.dataframe.index + timedelta(hours=1, minutes=0)
        # Filter out only forex days
        self.dataframe['forex_open'] = self.dataframe.apply(self._is_forex_day, axis=1)
        # All non forex days are np.NaN
        self.dataframe[self.dataframe['forex_open'] == False] = np.NaN
        # Save old datatime index and reset index
        self.dataframe.dropna(axis=0, inplace=True)
        self.dataframe.drop(columns=['forex_open'], axis=0, inplace=True)

    @staticmethod
    def _is_forex_day(row):
        start_time = 23
        end_time = 22
        valid_weekdays = [0, 1, 2, 3, 4, 6]  # Monday to Friday and Sunday
        if row.name.dayofweek in valid_weekdays:
            if row.name.dayofweek == 4 and row.name.time().hour >= end_time:
                return False  # Friday after end_time is not a valid day
            if row.name.dayofweek == 6 and row.name.time().hour < start_time:
                return False  # Sunday before start_time is not a valid day
            return True
        else:
            return False

    @property
    def ohlct(self) -> OHLCT:
        return self.timeframes[self.step_size].get_ohlct()

    def __repr__(self):
        return f'<{self.__class__.__name__}: ' \
               f'window={self.window} | ' \
               f'current_step={self.current_step} | ' \
               f'current_date={self.current_date} | ' \
               f'intervals: {self.intervals}>'

    def get_window_size(self):
        max_window_size = 0
        max_timeframe = max(list(self.timeframes.keys()))
        for indicator_vals in self.timeframes[max_timeframe].tech_processor.ema_args.values():
            if indicator_vals['length'] > max_window_size:
                max_window_size = indicator_vals['length']
        return max_window_size
