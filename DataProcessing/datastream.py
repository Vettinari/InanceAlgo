from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional
import pandas_ta as ta
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler


class OHLCVMinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        self.volume_scaler = MinMaxScaler()

    def fit(self, dataframe: pd.DataFrame, low: str = 'low', high: str = 'high'):
        # Fit self
        self.min_val = dataframe[low].min(axis=0)
        self.max_val = dataframe[high].max(axis=0)
        # Fit volume_scaler
        self.volume_scaler.fit(X=dataframe.volume.values.reshape(-1, 1))

    def transform(self, dataframe: pd.DataFrame):
        # Split to data
        ohlc_df = dataframe.drop(['volume'], axis=1)
        volume_df = dataframe[['volume']]

        # Scale down OHLC data
        ohlc_normalized = (ohlc_df - self.min_val) / (self.max_val - self.min_val)
        ohlc_scaled = ohlc_normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        ohlc_scaled_df = pd.DataFrame(ohlc_scaled,
                                      columns=ohlc_df.columns,
                                      index=dataframe.index)

        # Split to volume
        volume_scaled = self.volume_scaler.transform(volume_df.values.reshape(-1, 1))
        volume_scaled_df = pd.DataFrame(volume_scaled,
                                        columns=['volume'],
                                        index=dataframe.index)

        out = pd.concat([ohlc_scaled_df, volume_scaled_df], axis=1)
        return out[sorted(out.columns)]

    def fit_transform(self, dataframe: pd.DataFrame):
        self.fit(dataframe=dataframe)
        return self.transform(dataframe=dataframe)


class DataStream:
    def __init__(self,
                 ticker: str,
                 timeframes: list,
                 output_window_length: int,
                 # Moving avg
                 ma_lengths: Optional[List[int]] = None,
                 ma_type: Optional[str] = None,
                 # Momentums
                 momentums: Optional[List[int]] = None,
                 momentum_noise_reduction: Optional[int] = None,
                 # Local extreme values
                 local_extreme_orders: Optional[List[int]] = None,
                 # Data split
                 data_split: Optional[float] = 0.8
                 ):
        self.test = ticker == 'test'
        self.ticker: str = ticker.upper()
        self.step_size: int = min(timeframes)
        self.dataframe: pd.DataFrame = None
        self.timeframes: Dict[int, pd.DataFrame] = dict(zip(sorted(timeframes), [None] * len(timeframes)))
        self.output_window_length: int = output_window_length
        self.data_split: float = data_split
        self.ma_lengths: List[int] = ma_lengths
        self.ma_type = 'sma' if ma_type is None else ma_type
        self.momentums: List[int] = momentums
        self.momentum_noise_reduction = 4 if momentum_noise_reduction is None else momentum_noise_reduction
        self.local_extreme_orders = local_extreme_orders

        self._load_csv_data()
        self._clean_dataframe()
        self.prepare_all_timeframes()
        self.generator = DataGenerator(timeframes=self.timeframes,
                                       output=self.output_window_length,
                                       extreme_orders=self.local_extreme_orders)

    def _load_csv_data(self) -> None:
        if self.ticker == 'TEST':
            path = '/Users/milosz/Documents/Pycharm/InanceAlgo/EURUSD_short.csv'
        else:
            path = f'/Users/milosz/Documents/Pycharm/InanceAlgo/Datasets/forex/intraday/{self.ticker}.csv'
        self.dataframe = pd.read_csv(path,
                                     parse_dates=True,
                                     index_col='Datetime')
        self.dataframe = self.dataframe.groupby(
            pd.Grouper(freq=f'{self.step_size}T')).agg({'open': 'first',
                                                        'close': 'last',
                                                        'low': 'min',
                                                        'high': 'max',
                                                        'volume': 'sum'})

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

    def _prepare_timeframe(self, timeframe: int) -> None:
        # Group data to fit the frame
        price_data = self.dataframe.groupby(pd.Grouper(freq=f'{timeframe}T')).agg(
            {
                'open': 'first',
                'close': 'last',
                'low': 'min',
                'high': 'max',
                'volume': 'sum',
            })
        # Drop NaN values
        price_data = price_data.dropna(axis=0)

        # Scale the data fit the data scaler
        data_scaler = OHLCVMinMaxScaler()
        data_scaled = data_scaler.fit_transform(dataframe=price_data)
        data_scaled.columns = [f'scaled_{column}' for column in data_scaled.columns]

        if self.test:
            scaled_cols = set(data_scaled.columns)
            print('Data_scaled columns:\n', *scaled_cols)

        # Calculate moving averages
        if self.ma_lengths is not None:
            for ma_length in self.ma_lengths:
                if self.ma_type == 'hma':
                    data_scaled[f'scaled_{ma_length}'] = ta.hma(length=ma_length, close=data_scaled["scaled_close"])
                elif self.ma_type == 'ema':
                    data_scaled[f'scaled_{ma_length}'] = ta.ema(length=ma_length, close=data_scaled["scaled_close"])
                elif self.ma_type == 'sma':
                    data_scaled[f'scaled_{ma_length}'] = ta.sma(length=ma_length, close=data_scaled["scaled_close"])
            if self.test:
                moving_avg_cols = set(data_scaled.columns) - scaled_cols
                print('Moving avg cols:\n', *moving_avg_cols)

            # Drop nan values due to moving averages and momentum calculations
            data_scaled = data_scaled.dropna(axis=0)

        # Calculate momentums
        if self.momentums is not None:
            for mom in self.momentums:
                data_scaled[f'scaled_momentum_{mom}'] = (
                        data_scaled['scaled_close'] - data_scaled['scaled_close'].shift(mom)).rolling(
                    self.momentum_noise_reduction).mean()
            if self.test:
                momentum_cols = set(data_scaled.columns) - scaled_cols - moving_avg_cols
                print('Momentum cols:\n', *momentum_cols)

            # Drop nan values due to moving averages and momentum calculations
            data_scaled = data_scaled.dropna(axis=0)

        # # Calculate local extreme with different filters
        # if self.local_extreme_orders is not None:
        #     for order in self.local_extreme_orders:
        #         self.local_extrema(df=data_scaled,
        #                            order=order,
        #                            low_col='scaled_low',
        #                            high_col='scaled_high')
        #     if self.test:
        #         extreme_cols = set(data_scaled.columns) - scaled_cols - moving_avg_cols - momentum_cols
        #         print('Extreme cols:\n', *extreme_cols)

        # Drop nan values due to moving averages and momentum calculations
        data_scaled = data_scaled.dropna(axis=0)

        # Copy the index into 'datetime' column for ease of use
        price_data['datetime'] = price_data.index

        # Rename columns for the ease of use
        price_data.columns = [f'{timeframe}_{column}' for column in price_data.columns]
        data_scaled.columns = [f'{timeframe}_{column}' for column in data_scaled.columns]

        # Even out price data according to scaled_data(shorter df due to transformations) index.
        price_data = price_data.loc[data_scaled.index[0]:data_scaled.index[-1]]

        if timeframe == self.step_size:
            self.timeframes[timeframe] = pd.concat([price_data, data_scaled], axis=1)
        else:
            self.timeframes[timeframe] = data_scaled

        if self.test:
            print(f'TF: {timeframe} - TF_shape: {self.timeframes[timeframe].shape}')

    def prepare_all_timeframes(self):
        with ThreadPoolExecutor(max_workers=len(list(self.timeframes.keys()))) as executor:
            executor.map(self._prepare_timeframe, list(self.timeframes.keys()))

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

    @property
    def length(self):
        return self.timeframes[self.step_size].shape[0]

    def info(self) -> None:
        print('DataStream:')
        for tf, data in self.timeframes.items():
            print(f"\nTF: {tf} Shape: {data.shape}")


class DataGenerator:
    def __init__(self, timeframes: Dict[int, pd.DataFrame], output: int,
                 extreme_orders: Optional[List[int]] = None):
        self.timeframes: Dict[int, pd.DataFrame] = timeframes
        self.output: int = output
        self.step_size = min(self.timeframes.keys())
        self.max_timeframe = max(self.timeframes.keys())
        self.start_cursor = None
        self.extreme_orders = extreme_orders
        self.pick_start_date()

    def pick_start_date(self):
        self.start_cursor = self.timeframes[self.max_timeframe].index[self.output] + timedelta(
            minutes=self.max_timeframe)

    def __getitem__(self, item: pd.Timestamp) -> pd.DataFrame:
        out = []
        for timeframe in self.timeframes.keys():
            end_index = self.timeframes[timeframe].index.get_indexer([item], method='ffill')[0]
            temp_df = self.timeframes[timeframe].iloc[end_index - self.output:end_index].copy()

            if timeframe != self.step_size:
                temp_df[f'{timeframe}_update'] = 1 if item.minute % timeframe == 0 else 0

            for order in self.extreme_orders:
                self.local_extrema(df=temp_df,
                                   order=order,
                                   low_col=f'{timeframe}_scaled_low',
                                   high_col=f'{timeframe}_scaled_high')
            out.append(temp_df.reset_index(drop=True))

        return pd.concat(out, axis=1)

    def __repr__(self):
        return f"{self.__class__.__name__}: max_timeframe={self.max_timeframe} start_date={self.start_cursor}"

    def local_extrema(self, df: pd.DataFrame, order: int, low_col: str = 'low', high_col: str = 'high'):
        # Local max
        highs = df.iloc[argrelextrema(df[high_col].values, np.greater_equal, order=order)[0]][high_col].notnull()
        df[f'{high_col}_max_{order}'] = highs
        # Local min
        lows = df.iloc[argrelextrema(df[low_col].values, np.less_equal, order=order)[0]][low_col].notnull()
        df[f'{low_col}_min_{order}'] = lows

        df[f'{high_col}_max_{order}'] = df[f'{high_col}_max_{order}'].map({np.NaN: 0, True: 1})
        df[f'{low_col}_min_{order}'] = df[f'{low_col}_min_{order}'].map({np.NaN: 0, True: 1})

        df.loc[df.index[-1], f'{high_col}_max_{order}'] = 0
        df.loc[df.index[-1], f'{low_col}_min_{order}'] = 0
