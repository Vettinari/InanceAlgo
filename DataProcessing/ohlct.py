from typing import Union

import pandas as pd


class OHLCT:
    def __init__(self, dataframe_row, timeframe: int):
        self.open: float = dataframe_row.open
        self.high: float = dataframe_row.high
        self.low: float = dataframe_row.low
        self.close: float = dataframe_row.close
        self.volume: float = dataframe_row.volume
        self.time: Union[pd.datetime, str, int] = dataframe_row.name
        self.timeframe: int = timeframe

    @property
    def dataframe(self) -> pd.DataFrame:
        out = {'open': self.open,
               'high': self.high,
               'low': self.low,
               'close': self.close,
               'volume': self.volume}
        return pd.DataFrame(out, index=[self.time])

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}_{self.timeframe}T: ' \
               f'Open={self.open}, High={self.high}, Low={self.low}, Close={self.close}, Time={self.time}>'
