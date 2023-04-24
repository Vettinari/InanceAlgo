import datetime
from datetime import timedelta
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import Utils


class DataBuffer:
    def __init__(self, ticker, intervals, return_window_size, test=False):
        self.ticker: str = ticker
        self.intervals: List[int] = intervals
        self.return_window_size: int = return_window_size + 1
        self.test: bool = test
        self.step_size: int = min(intervals)
        self.max_timeframe = max(intervals)
        self.max_timeframe_window = int(max(intervals) / self.step_size) * self.return_window_size

        # Calculated
        self.step_dataframe: pd.DataFrame = None
        self.base_dataframe: pd.DataFrame = None

        self._load_csv_data()
        self._clean_dataframe()

        self.start_step: int = self.get_start_step()
        # Returns
        self.processed_data: Dict[int, Union[pd.DataFrame, np.array]] = dict(zip(self.intervals, []))

    def get_start_step(self):
        return int((self.max_timeframe * self.return_window_size) / self.step_size)

    def process_data(self):
        for current_step in tqdm(range(0, self.step_dataframe.shape[0] - self.start_step)):
            self.processed_data[current_step] = {'time': self.step_dataframe.index[current_step + self.start_step],
                                                 'data': self.get_multi_timeframe_data(current_step=current_step)}

    def get_multi_timeframe_data(self, current_step) -> Dict[int, pd.DataFrame]:
        current_step += self.start_step
        start_date = self.step_dataframe.index[current_step]
        temp_df = self.step_dataframe.loc[:start_date].iloc[-self.max_timeframe_window:-1]

        out = {}
        for interval in self.intervals:
            data = temp_df
            if interval != self.step_size:
                data = data.groupby(pd.Grouper(freq=f'{interval}T')).agg({'open': 'first',
                                                                          'close': 'last',
                                                                          'low': 'min',
                                                                          'high': 'max',
                                                                          'volume': 'sum'}).dropna()
            out[interval] = data.iloc[-self.return_window_size:]
        return out

    def _load_csv_data(self):
        self.step_size = min(self.intervals)
        path = f'/Users/milosz/Documents/Pycharm/InanceAlgo/Datasets/forex/intraday/{self.ticker}.csv'
        self.base_dataframe = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')

        if self.test:
            self.base_dataframe = self.base_dataframe.iloc[-int(len(self.base_dataframe) * 0.01):]

        self.step_dataframe = self.base_dataframe.groupby(pd.Grouper(freq=f'{self.step_size}T')).agg({'open': 'first',
                                                                                                      'close': 'last',
                                                                                                      'low': 'min',
                                                                                                      'high': 'max',
                                                                                                      'volume': 'sum'})

    def _clean_dataframe(self):
        # Adjust to Amsterdam time
        self.step_dataframe.index = self.step_dataframe.index + timedelta(hours=1, minutes=0)
        # Filter out only forex days
        self.step_dataframe['forex_open'] = self.step_dataframe.apply(self._is_forex_day, axis=1)
        # All non forex days are np.NaN
        self.step_dataframe[self.step_dataframe['forex_open'] == False] = np.NaN
        # Save old datatime index and reset index
        self.step_dataframe.dropna(axis=0, inplace=True)
        self.step_dataframe.drop(columns=['forex_open'], axis=0, inplace=True)

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

    def __repr__(self):
        return f'<{self.__class__.__name__}: ' \
               f'window={self.return_window_size} | ' \
               f'intervals: {self.intervals}>'
