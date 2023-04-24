import math

import numpy as np
import pandas as pd
from typing import Optional
from DataProcessing.ta import TechnicalIndicator
import pandas_ta


class DataProcessor:
    def __init__(self, interval: int, window: int = 1):
        self._data: pd.DataFrame = None
        self.interval = interval
        self.window = window

    def process_data(self, data_window: pd.DataFrame):
        self._data = data_window.groupby(
            pd.Grouper(freq=f'{self.interval}T')).agg(
            {
                'open': 'first',
                'close': 'last',
                'low': 'min',
                'high': 'max',
                'volume': 'sum',
            })
        self._data.dropna(inplace=True, axis=0)

    def data(self):
        return self._data.iloc[-self.window:]

    @property
    def state_size(self):
        return self._data.shape[0]


class TechProcessor(DataProcessor):
    def __init__(self, interval: int, window: int):
        super().__init__(interval=interval, window=window)
        self.slope = 2
        self.atr_args = {'length': 14}
        self.rsi_args = {'length': 14 + self.slope}
        self.ema_args = {f'ema_{5 + self.slope}': {'length': 5 + self.slope},
                         f'ema_{13 + self.slope}': {'length': 13 + self.slope},
                         f'ema_{50 + self.slope}': {'length': 50 + self.slope},
                         f'ema_{100 + self.slope}': {'length': 100 + self.slope}}

    def process_data(self, data_window: pd.DataFrame, slope=1):
        out = [
            self.calculate_atr(data_window=data_window),
            *self.calculate_rsi(data_window=data_window),
            *self.calculate_ema(data_window=data_window),
            *self.calculate_volume_ma(data_window=data_window),
        ]
        self._data = pd.concat(out, axis=1)

    def calculate_atr(self, data_window):
        tech = TechnicalIndicator(indicator='atr', params=self.atr_args)
        return np.round(tech.calculate(data=data_window), 5)

    def calculate_volume_ma(self, data_window: pd.DataFrame):
        tech = pandas_ta.ema(close=data_window.volume, length=2)
        slope_tech = np.round(tech.rolling(self.slope).mean().diff(), 6)
        return [tech, slope_tech]

    def calculate_rsi(self, data_window):
        tech = TechnicalIndicator(indicator='rsi', params=self.rsi_args)
        tech = tech.calculate(data=data_window)

        slope_tech = np.round(tech.rolling(self.slope).mean().diff(), 6)
        slope_tech.name = f"{slope_tech.name}_slope"

        return [tech, slope_tech]

    def calculate_ema(self, data_window):
        tech_out = []
        slope_out = []
        for args in self.ema_args.values():
            args['length'] = args['length'] - self.slope

            # calculate emas
            tech = TechnicalIndicator(indicator='ema', params=args)
            tech = tech.calculate(data=data_window)

            # calculate slope
            slope_tech = np.round(tech.rolling(self.slope).mean().diff(), 6)
            slope_tech.name = f"{slope_tech.name}_slope"

            # calculate ema-close delta
            tech_delta = tech - data_window.close
            tech_name = tech.name
            tech = tech.to_frame()
            tech[f"{tech_name}_delta"] = tech_delta
            tech_out.append(tech[[f"{tech_name}_delta"]])

            slope_out.append(slope_tech)
        return tech_out + slope_out
