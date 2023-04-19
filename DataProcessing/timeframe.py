from typing import Optional

from DataProcessing.chart_processors import CandlestickProcessor, VolumeProfile
from DataProcessing.processors import DataProcessor, TechProcessor


class OHLCT:
    def __init__(self, dataframe_row, interval: int):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.volume = dataframe_row.volume
        self.time = dataframe_row.name
        self.interval = interval

    def __repr__(self):
        return f'<{self.__class__.__name__}_{self.interval}T: ' \
               f'Open={self.open}, High={self.high}, Low={self.low}, Close={self.close}, Time={self.time}>'


class TimeFrame:
    def __init__(self,
                 interval: int,
                 chart_window: int,
                 return_window: int = 1,
                 data_types: Optional[list] = None):
        self.interval = interval
        self.data_processor = DataProcessor(interval=interval, window=return_window)
        self.tech_processor = TechProcessor(interval=interval, window=return_window)
        self.candlestick_processor = CandlestickProcessor(interval=interval, window=chart_window)
        self.volume_profile_processor = VolumeProfile(interval=interval, window=chart_window)
        self.data_types = ['data', 'tech'] or data_types

    def process_window_data(self, window_data):
        if 'data' in self.data_types:
            self.data_processor.process_data(data_window=window_data)
        if 'tech' in self.data_types:
            self.tech_processor.process_data(data_window=window_data)
        if 'candlestick' in self.data_types:
            self.candlestick_processor.process_data(data_window=window_data)
        if 'volume_profile' in self.data_types:
            self.volume_profile_processor.process_data(data_window=window_data)

    def get_ohlct(self) -> OHLCT:
        return OHLCT(dataframe_row=self.data_processor.data().iloc[-1], interval=self.interval)

    def data(self):
        return self.data_processor.data()

    def tech(self):
        return self.tech_processor.data()

    def candlestick(self):
        raise NotImplementedError()

    def volume_profile(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.interval}T data\n" \
               f"{self.data()}\n" \
               f"{self.interval}T tech:\n" \
               f"{self.tech()}"

    @property
    def state_size(self):
        data_state = self.data().shape[0] * self.data().shape[1]
        tech_state = self.tech().shape[0] * self.tech().shape[1]
        return data_state + tech_state
