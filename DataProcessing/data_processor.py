import pandas as pd


class OHLCT:
    def __init__(self, dataframe_row):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.time = dataframe_row.name


class DataProcessor:
    def __init__(self, ticker: str, technical_indicators: list):
        self.ticker = ticker
        self.dataframe: pd.DataFrame = None
        self.technical_dataframe: pd.DataFrame = None
        self.technical_indicators: list = technical_indicators
        self.current_step = 0

    def load_data(self):
        pass

    def clean_data(self):
        pass

    def generate_technical_indicators(self):
        pass

    @property
    def current_ohlct(self) -> OHLCT:
        return OHLCT(dataframe_row=self.dataframe.iloc[self.current_step])

    def step(self):
        self.current_step += 1
