import io
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from DataProcessing.processors import DataProcessor


class Renderer(DataProcessor):
    def __init__(self, interval: int, window: int):
        super().__init__(interval=interval)
        self.fig: go.Figure = None
        self.render_window = window

    def process_data(self, data_window):
        super().process_data(data_window=data_window)
        self._data = self._data.iloc[-self.render_window:].reset_index(drop=True)
        self._render()

    def _render(self):
        pass

    @property
    def fig_as_array(self):
        # convert Plotly fig to  an array
        fig_bytes = self.fig.to_image(format="png")
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img) / 255


class CandlestickProcessor(Renderer):

    def _render(self):
        data = self._data
        interval = self.interval

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

        fig.update_layout(width=self.render_window * 8,
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

        self.fig = fig


class VolumeProfile(Renderer):

    def process_data(self, data_window):
        interval_data = self._data.groupby(pd.Grouper(freq=f'{self.interval}T')).agg({
            'open': 'first',
            'close': 'last',
            'low': 'min',
            'high': 'max',
            'volume': 'sum',
        })
        interval_data.dropna(inplace=True, axis=0)
        interval_data = interval_data.iloc[-self.render_window:]

        self._data.dropna(inplace=True, axis=0)
        data = self._data.loc[interval_data.index[0]:]

        bins = 30
        price_bins = np.linspace(data['close'].min(), data['close'].max(), bins + 1)
        bin_labels = pd.cut(data['close'], bins=price_bins, include_lowest=True)
        self._data = data.groupby(bin_labels)['volume'].sum()

    def _render(self):
        fig = go.Figure(go.Bar(
            x=self._data.values,
            y=[str(interval) for interval in self._data.index],
            orientation='h',
        ))

        fig.update_layout(width=400,
                          height=400,
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                          )

        fig.update_xaxes(rangeslider_visible=False,
                         showgrid=True,
                         visible=False,
                         showticklabels=False,
                         gridwidth=0)

        fig.update_yaxes(showgrid=False,
                         gridwidth=.5,
                         zerolinewidth=0,
                         showticklabels=False,
                         gridcolor='White')

        self.fig = fig
