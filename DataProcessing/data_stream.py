from multiprocessing import Pool

import Utils
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from DataProcessing.processors import DataProcessor


class DataStream:
    def __init__(self, ticker: str,
                 timeframes: list,
                 data_processor_type: DataProcessor,
                 technicals: List[str],
                 processor_window_output: int,
                 test: bool = False,
                 data_split: Optional[float] = 0.9,
                 data_size: Optional[int] = None):
        self.test: bool = test
        self.ticker: str = ticker
        self.dataframe: pd.DataFrame = None
        self.step_size: int = min(timeframes)
        self.timeframes: list = sorted(timeframes)
        self.technicals: list = technicals
        self.processor_window_output: int = processor_window_output
        self.data_processor_type: DataProcessor = data_processor_type
        self.data_processors: Dict[int, DataProcessor] = None
        self.data_window_size = int((max(self.timeframes) / self.step_size) * self.processor_window_output) * 2
        self.data_size: int = data_size
        self.data_split: float = data_split
        # Initialize functions
        self._load_csv_data()
        self._clean_dataframe()
        # self._add_technicals()

        if self.data_size:
            self.dataframe = self.dataframe.iloc[-self.data_size:]

        self.processors: List[DataProcessor] = []
        self.split_index: int = int(len(self.dataframe) * self.data_split) if self.data_split else len(self.dataframe)

    def run_single_timeframe(self, interval: int) -> DataProcessor:

        data_processor = self.data_processor_type(timeframe=interval, window=self.processor_window_output)

        for key, window_start in tqdm(enumerate(range(self.data_window_size, self.split_index)),
                                      desc=f"{interval}_train",
                                      position=self.timeframes.index(interval),
                                      leave=False,
                                      ncols=100):
            window = self.dataframe.iloc[window_start - self.data_window_size: window_start]
            data_processor.process_train_data(key=key, data_window=window)

        for key, window_start in tqdm(enumerate(range(self.split_index, len(self.dataframe))),
                                      desc=f"{interval}_test",
                                      position=self.timeframes.index(interval),
                                      leave=False,
                                      ncols=100):
            window = self.dataframe.iloc[window_start - self.data_window_size: window_start]
            data_processor.process_test_data(key=key, data_window=window)

        return data_processor

    def run_multithread(self) -> None:
        with Pool(3) as pool:
            for result in pool.imap_unordered(self.run_single_timeframe, self.timeframes):
                self.processors.append(result)

        self.data_processors = {processor.timeframe: processor for processor in self.processors}

    def get_hindsight_data(self, current_step: int, horizon: int) -> pd.DataFrame:
        """
        Returns data from the future for training purposes (hindsight rewards).
        :param current_step: int - current step in the data stream
        :param horizon: int - number of steps into the future
        :return: pd.DataFrame - data from the future
        """
        return self.data_processors[self.step_size].train_data[current_step + horizon]

    def save_datastream(self):
        path = f'data_streams/{"_".join(map(str, self.timeframes))}'
        filename = f'{self.ticker}_window{self.processor_window_output}.data_stream'
        if self.data_size:
            path += f"_data{self.data_size}"
        Utils.save_object(object_to_save=self, path=path, filename=filename)

    def max_steps(self, data_type='train') -> int:
        if data_type == 'train':
            return len(self.data_processors[self.step_size].train_data.keys()) - self.data_window_size - 1
        return len(self.data_processors[self.step_size].test_data.keys()) - 1

    def _load_csv_data(self) -> None:
        self.step_size = min(self.timeframes)
        path = f'/Users/milosz/Documents/Pycharm/InanceAlgo/Datasets/forex/intraday/{self.ticker}.csv'
        df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
        df = df.groupby(pd.Grouper(freq=f'{self.step_size}T')).agg({'open': 'first',
                                                                    'close': 'last',
                                                                    'low': 'min',
                                                                    'high': 'max',
                                                                    'volume': 'sum'})
        self.dataframe = df

    def _clean_dataframe(self) -> None:
        # Adjust to Amsterdam time
        self.dataframe.index = self.dataframe.index + timedelta(hours=1, minutes=0)
        # Filter out only forex days
        self.dataframe['forex_open'] = self.dataframe.apply(self._is_forex_day, axis=1)
        # All non forex days are np.NaN
        self.dataframe[self.dataframe['forex_open'] == False] = np.NaN
        # Save old datatime index and reset index
        self.dataframe.dropna(axis=0, inplace=True)
        self.dataframe.drop(columns=['forex_open'], axis=0, inplace=True)

    def info(self) -> None:
        print(self.__repr__())
        print('train_max_steps', self.max_steps(data_type='train'))
        print('test_max_steps', self.max_steps(data_type='test'))

    @staticmethod
    def _is_forex_day(row: pd.DataFrame) -> bool:
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

    @staticmethod
    def load_datastream(path: str):
        return Utils.load_object(path=path, filename=None)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' \
               f'\nprocessors={self.data_processors}'

    def __getitem__(self, key: str) -> Dict[int, DataProcessor]:
        return {timeframe: dataprocessor[key] for timeframe, dataprocessor in self.data_processors.items()}
