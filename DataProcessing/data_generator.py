from datetime import timedelta

import numpy as np

from DataProcessing.data_processor import DataProcessor
import pandas as pd


class DataGenerator:
    def __init__(self, ticker: str, intervals: list):
        self.ticker: str = ticker
        self.intervals: list = sorted(intervals)
        self.base_dataframe: pd.DataFrame = None
        self.processors: dict = dict(zip(intervals, list()))
        self.load_data()
        self.clean_data()

    def load_data(self):
        path = f'Datasets/forex/intraday/{self.ticker}.csv'
        self.base_dataframe = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
        self.base_dataframe = self.base_dataframe.groupby(
            pd.Grouper(freq=f'{self.intervals[0]}T')).agg({'open': 'first',
                                                           'close': 'last',
                                                           'low': 'min',
                                                           'high': 'max',
                                                           'volume': 'sum'})
        self.intervals.pop(0)

    def clean_data(self):
        self.base_dataframe.index = self.base_dataframe.index + timedelta(hours=1,
                                                                          minutes=0)  # Adjust to Amsterdam time
        self.base_dataframe['forex_open'] = self.base_dataframe.apply(self.is_forex_day,
                                                                      axis=1)  # Filter out only forex days
        self.base_dataframe[self.base_dataframe['forex_open'] == False] = np.NaN  # All non forex days are np.NaN
        pass

    def build_processors(self):
        for interval in self.intervals:
            self.processors[interval] = DataProcessor(interval=interval)

    @staticmethod
    def is_forex_day(row):
        start_time = 23
        end_time = 22

        # MARK WORKING DAYS
        if row.name.dayofweek == 0:
            return True
        elif row.name.dayofweek == 1:
            return True
        elif row.name.dayofweek == 2:
            return True
        elif row.name.dayofweek == 3:
            return True
        elif row.name.dayofweek == 4 and row.name.time().hour < end_time:
            return True
        elif row.name.dayofweek == 6 and row.name.time().hour >= start_time:
            return True
        else:
            return False
