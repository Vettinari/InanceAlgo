import pandas as pd


class OHLCT:
    def __init__(self, dataframe_row, timeframe: int):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.volume = dataframe_row.volume
        self.time = dataframe_row.name
        self.timeframe = timeframe

    @property
    def dataframe(self):
        out = {'open': self.open,
               'high': self.high,
               'low': self.low,
               'close': self.close,
               'volume': self.volume}
        return pd.DataFrame(out, index=[self.time])

    def __repr__(self):
        return f'<{self.__class__.__name__}_{self.timeframe}T: ' \
               f'Open={self.open}, High={self.high}, Low={self.low}, Close={self.close}, Time={self.time}>'
