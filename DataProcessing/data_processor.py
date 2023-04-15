import enum

import numpy as np
import pandas as pd
from typing import Optional

from plotly.subplots import make_subplots

import Utils
from DataProcessing import ta
import plotly.graph_objects as go
import io
from PIL import Image


class OHLCT:
    def __init__(self, dataframe_row):
        self.open = dataframe_row.open
        self.high = dataframe_row.high
        self.low = dataframe_row.low
        self.close = dataframe_row.close
        self.time = dataframe_row.name

    def __repr__(self):
        return f'<{self.__class__.__name__}: ' \
               f'Open={self.open}, High={self.high}, Low={self.low}, Close={self.close}, Time={self.time}>'


class DataProcessor:
    processor_type = 'base'

    def __init__(self, interval: int, window: int):
        self.interval = interval
        self.window = window
        self.data = None

    def process(self, data: pd.DataFrame,
                current_step: Optional[int], save: Optional[bool]):
        self.data: pd.DataFrame = data.groupby(pd.Grouper(freq=f'{self.interval}T')).agg(
            {'open': 'first',
             'close': 'last',
             'low': 'min',
             'high': 'max',
             'volume': 'sum',
             })
        self.data.dropna(inplace=True, axis=0)
        self.data = self.data.iloc[-self.window:]

    def __call__(self, interval: int, window: int):
        self.__init__(interval=interval, window=window)


class ChartProcessor(DataProcessor):
    processor_type = 'chart'

    def __init__(self, interval: int, window: int):
        super().__init__(interval=interval, window=window)
        self.fig: go.Figure = None

    def process(self, data, current_step: int, save: Optional[bool]):
        super().process(data=data, current_step=current_step, save=save)
        self.fig = chart_render(data=self.data, window=self.window, high_res=1)
        if save:
            save_path = Utils.make_path(f'charts/{self.interval}')
            self.fig.write_image(f'{save_path}/{current_step}.jpg', 'jpg')
        else:
            return self.fig_as_array

    @property
    def fig_as_array(self):
        # convert Plotly fig to  an array
        fig_bytes = self.fig.to_image(format="png")
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img)


def chart_render(data: pd.DataFrame, window: int, high_res=1) -> go.Figure:
    data = data.reset_index(drop=True)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0, row_width=[0.2, 0.8],
                        specs=[[{"secondary_y": False, "r": -0.06}],
                               [{"secondary_y": True, "r": -0.06}]]
                        )

    candlestick = go.Candlestick(
        x=data.index,
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close,
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=False)

    volume_bar = go.Bar(
        x=data.index,
        y=data.volume,
        showlegend=False
    )

    volume_line = go.Scatter(
        x=data.index,
        y=data.volume,
        showlegend=False
    )

    close_line = go.Scatter(
        x=data.index,
        y=data.close,
        showlegend=False,
    )

    fig.add_trace(candlestick, row=1, col=1)
    fig.update_traces({d: {"fillcolor": c, "line": {"color": c}}
                       for d, c in zip(["increasing", "decreasing"], ["green", "red"])})
    # fig.add_trace(volume_bar, row=2, col=1, secondary_y=False)

    fig.add_trace(volume_line, row=2, col=1,
                  secondary_y=False
                  )
    fig.add_trace(close_line, row=2, col=1,
                  secondary_y=True
                  )

    fig.add_hline(y=data.iloc[-1].close, line_width=1.5, line_color="Blue", row=1, col=1)  # Highlight last price

    fig.update_layout(width=window * 8 * high_res,
                      height=400 * high_res,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                      )

    yaxis_density = 0.001  # Adjust this value to change the y-axis grid density

    fig.update_xaxes(rangeslider_visible=False,
                     visible=False,
                     showticklabels=False,
                     showgrid=False,
                     gridwidth=0)

    fig['layout'][f'yaxis1'].update(showgrid=True)

    fig.update_yaxes(gridwidth=.5,
                     zerolinewidth=0,
                     showticklabels=False,
                     gridcolor='White',
                     dtick=yaxis_density)

    fig['layout'][f'yaxis3'].update(visible=False,
                                    showticklabels=False,
                                    showgrid=False,
                                    gridwidth=0)

    return fig
