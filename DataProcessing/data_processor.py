import enum
import pandas as pd
from typing import Optional

import Utils
from DataProcessing import ta
import plotly.graph_objects as go


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
    processor_type = 'base'

    def __init__(self, interval: int, window: int):
        self.interval = interval
        self.window = window
        self.data = None

    def process(self, data: pd.DataFrame, current_step: Optional[int], save: Optional[bool]):
        self.data: pd.DataFrame = data.groupby(pd.Grouper(freq=f'{self.interval}T')).agg(
            {'open': 'first',
             'close': 'last',
             'low': 'min',
             'high': 'max',
             'volume': 'sum'})
        self.data.dropna(inplace=True, axis=0)
        self.data = self.data.iloc[-self.window:]

    def __call__(self, interval: int, window: int):
        self.__init__(interval=interval, window=window)


class ChartProcessor(DataProcessor):
    processor_type = 'chart'

    def __init__(self, interval: int, window: int):
        super().__init__(interval=interval, window=window)

    def process(self, data, current_step, save):
        super().process(data=data, current_step=current_step, save=save)
        self.render(current_step=current_step, save=save)

    def render(self, current_step: int, save: bool = True, high_res=1):
        df: pd.DataFrame = self.data.reset_index(drop=True)
        fig = go.Figure(data=[
            go.Candlestick(x=df.index[:],
                           open=df["open"],
                           high=df["high"],
                           low=df["low"],
                           close=df["close"],
                           increasing_line_color="green",
                           decreasing_line_color="red",
                           )
        ])

        fig.add_hline(y=df.iloc[-1].close, line_width=1.5, line_color="Blue")  # Highlight last price

        fig.update_traces({d: {"fillcolor": c, "line": {"color": c}}
                           for d, c in zip(["increasing", "decreasing"], ["green", "red"])})

        fig.update_layout(width=self.window * 8 * high_res,
                          height=400 * high_res,
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
                          )

        yaxis_density = 0.001  # Adjust this value to change the y-axis grid density

        fig.update_xaxes(rangeslider_visible=False,
                         visible=False,
                         showticklabels=False,
                         showgrid=False,
                         gridwidth=0)

        fig.update_yaxes(showgrid=True,
                         gridwidth=.5,
                         zerolinewidth=0,
                         showticklabels=False,
                         gridcolor='White',
                         dtick=yaxis_density)

        if save is False:
            fig.show()
        else:
            save_path = Utils.make_path(f"charts/{self.interval}")
        fig.write_image(f'{save_path}/{current_step}.jpg')


class TechProcessor(DataProcessor):
    processor_type = 'tech'

    def __init__(self, interval, window):
        super().__init__(interval=interval, window=window)
