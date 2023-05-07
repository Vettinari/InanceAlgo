import enum
import io
from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Utils


class DataProcessor:
    def __init__(self, timeframe: int, window: int = 1):
        # Parameters
        self.timeframe: int = timeframe
        self.window: int = window
        # Data
        self._train_data: Dict[int, pd.DataFrame] = {}
        self._test_data: Dict[int, pd.DataFrame] = {}

    def process_train_data(self, key: int, data_window: pd.DataFrame) -> None:
        fresh_data = data_window.groupby(pd.Grouper(freq=f'{self.timeframe}T')).agg(
            {
                'open': 'first',
                'close': 'last',
                'low': 'min',
                'high': 'max',
                'volume': 'sum',
            })
        fresh_data.dropna(inplace=True, axis=0)
        self._train_data[key] = fresh_data.iloc[-self.window:]

    def process_test_data(self, key: int, data_window: pd.DataFrame) -> None:
        fresh_data = data_window.groupby(pd.Grouper(freq=f'{self.timeframe}T')).agg(
            {
                'open': 'first',
                'close': 'last',
                'low': 'min',
                'high': 'max',
                'volume': 'sum',
            })
        fresh_data.dropna(inplace=True, axis=0)
        self._test_data[key] = fresh_data.iloc[-self.window:]

    @property
    def train_data(self) -> Dict[int, pd.DataFrame]:
        return self._train_data

    @property
    def test_data(self) -> Dict[int, pd.DataFrame]:
        return self._test_data

    def __repr__(self) -> str:
        return f"DataProcessor: timeframe={self.timeframe} window={self.window} " \
               f"train_data={len(self._train_data)} test_data={len(self._test_data)}"

    def __call__(self, timeframe: int, window: int):
        self.__init__(timeframe=timeframe, window=window)

    def __getitem__(self, key: str) -> pd.DataFrame:
        key = key.split("_")
        data_type = key[0]
        index = int(key[-1])

        assert data_type in ['train', 'test', 'tech'], "Data type not generated!"

        if data_type == 'train':
            return self.train_data[index]
        else:
            return self.test_data[index]


class Renderer(DataProcessor):
    def __init__(self, timeframe: int, window: int):
        super().__init__(timeframe, window)
        self._charts: Dict[int, go.Figure] = {}

    def _render(self, data):
        pass

    @staticmethod
    def fig_as_array(fig):
        fig_bytes = fig.to_image(format="png")
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img) / 255


class CandlestickProcessor(Renderer):

    def process_train_data(self, key, data_window: pd.DataFrame):
        super().process_train_data(key=key, data_window=data_window)
        self._charts[key] = self.render(self._train_data[key])

    def process_test_data(self, key, data_window: pd.DataFrame):
        super().process_test_data(key=key, data_window=data_window)
        self.render(self._test_data[key])

    def render(self, data):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0, row_width=[0.15, 0.85])

        candlestick = go.Candlestick(
            x=data.index,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            increasing_line_color="green",
            decreasing_line_color="red",
            showlegend=False)

        volume_bars = go.Bar(x=data.index, y=data.volume, showlegend=False, opacity=0.5,
                             marker_color='blue', marker_line_width=0)

        fig.add_trace(candlestick, row=1, col=1, )
        fig.update_traces({'increasing': {'fillcolor': 'green',
                                          'line': {'color': 'green',
                                                   'width': 2}},
                           'decreasing': {'fillcolor': 'red',
                                          'line': {'color': 'red',
                                                   'width': 2}}})

        fig.add_trace(volume_bars, row=2, col=1)
        fig.add_hline(y=data.iloc[-1].close, line_width=1.5, line_color="Blue", row=1,
                      col=1)  # Highlight last price

        fig.update_layout(width=self.window * 8,
                          height=400,
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

        fig.update_yaxes(showgrid=True,
                         gridwidth=.5,
                         zerolinewidth=0,
                         showticklabels=False,
                         gridcolor='White',
                         dtick=yaxis_density)

        return fig
