import enum
import pandas as pd
from typing import Optional
from DataProcessing import ta
import plotly.graph_objects as go

from datetime import timedelta


class OHLCT:
    def __init__(self, dataframe_row):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.time = dataframe_row.name

    def __repr__(self):
        return f'Open: {self.open}, High: {self.high}, Low: {self.low}, Close: {self.close}, Time: {self.time}'


class DataProcessor:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 interval: int,
                 technical_indicators: Optional[list] = None):
        self.interval = interval
        self.dataframe: pd.DataFrame = dataframe
        self.technical_dataframe: pd.DataFrame = None
        self.current_step = 0

    def clean_data(self):
        pass

    def generate_technical_indicators(self):
        pass

    def ohlct(self, current_step) -> OHLCT:
        return OHLCT(dataframe_row=self.dataframe.iloc[current_step])

    def info(self):
        print(f'Start: {self.dataframe.iloc[0].name} - End: {self.dataframe.iloc[-1].name}')
        print(f'Total steps: {self.total_steps} - Interval: {self.interval}')

    def get_state(self, current_step):
        pass

    def __call__(self, interval: int, technical_indicators: list):
        self.__init__(interval=interval, technical_indicators=technical_indicators)


class ChartProcessor(DataProcessor):
    def __init__(self, dataframe: pd.DataFrame, interval: int, ):
        super().__init__(dataframe=dataframe, interval=interval)

    def render(self, save: bool = True, inspection: bool = False):

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.dataframe["date"],
                    open=self.dataframe["open"],
                    high=self.dataframe["high"],
                    low=self.dataframe["low"],
                    close=self.dataframe["close"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            ]
        )

        if save is False:
            fig.show()
        else:
            fig.write_image()


