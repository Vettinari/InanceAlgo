import datetime
from datetime import timedelta
from typing import Dict

import numpy as np
from tqdm import tqdm

import Utils
from DataProcessing.data_processor import DataProcessor, OHLCT, ChartProcessor
import pandas as pd


class DataPipeline:
    def __init__(self, ticker: str, intervals: list, window: int,
                 processor_type: DataProcessor = DataProcessor):
        self.ticker: str = ticker
        self.intervals: list = sorted(intervals)
        self.dataframe: pd.DataFrame = None
        self.step_size: int = None
        self.processors: Dict[int, DataProcessor] = {}
        self.window = window
        self.processor_type: DataProcessor = processor_type
        self.current_step = None
        self.current_date = None
        # Init data pipeline
        self._load_data()
        self._clean_data()
        self.reset_step()
        self._build_processors()

    def reset_step(self):
        self.current_step = int(max(self.intervals) / self.step_size * self.window)
        self.current_date = self.dataframe.index[self.current_step]

    def _load_data(self):
        self.step_size = min(self.intervals)
        path = f'Datasets/forex/intraday/{self.ticker}.csv'
        # path = f'EURUSD_short.csv'
        df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
        df = df.groupby(pd.Grouper(freq=f'{self.step_size}T')).agg({'open': 'first',
                                                                    'close': 'last',
                                                                    'low': 'min',
                                                                    'high': 'max',
                                                                    'volume': 'sum'})
        self.dataframe = df

    def _clean_data(self):
        # Adjust to Amsterdam time
        self.dataframe.index = self.dataframe.index + timedelta(hours=1, minutes=0)
        # Filter out only forex days
        self.dataframe['forex_open'] = self.dataframe.apply(self._is_forex_day, axis=1)
        # All non forex days are np.NaN
        self.dataframe[self.dataframe['forex_open'] == False] = np.NaN
        # Save old datatime index and reset index
        self.dataframe.dropna(axis=0, inplace=True)
        self.dataframe.drop(columns=['forex_open'], axis=0, inplace=True)

    def _build_processors(self):
        for interval in self.intervals:
            self.processors[interval] = self.processor_type(interval=interval, window=self.window)

    def get_current_data(self, save=True):
        data_window = self.dataframe.loc[:self.current_date].iloc[:-1]
        for processor_interval, processor in self.processors.items():
            if processor.processor_type == 'chart':
                processor.process(data=data_window, current_step=self.current_step, save=save)
            else:
                processor.process(data=data_window, current_step=None)

        self.current_step += 1
        self.current_date = self.dataframe.index[self.current_step]

    def render_all_charts(self):
        assert self.processor_type == ChartProcessor, 'use ChartProcessor as processor type to enable renders'
        time = datetime.datetime.now()
        for _ in tqdm(range(0, 1000)):
            self.get_current_data(save=True)

        print(f'Rendering took {datetime.datetime.now() - time}')

    def get_ohlct(self, current_step) -> OHLCT:
        return OHLCT(dataframe_row=self.dataframe.iloc[current_step])

    def reset(self):
        self.current_step = self.window

    def __repr__(self):
        return f'<{self.__class__.__name__}: ' \
               f'current_step={self.current_step} | ' \
               f'current_date={self.current_date} | ' \
               f'intervals: {self.intervals}>'

    @staticmethod
    def _is_forex_day(row):
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
