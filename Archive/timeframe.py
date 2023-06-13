import enum
import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler


class OHLCVMinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        self.volume_scaler = MinMaxScaler()

    def fit(self, dataframe: pd.DataFrame, low: str = 'low', high: str = 'high'):
        # Fit self
        self.min_val = dataframe[low].min(axis=0)
        self.max_val = dataframe[high].max(axis=0)
        # Fit volume_scaler
        self.volume_scaler.fit(X=dataframe.volume.values.reshape(-1, 1))

    def transform(self, dataframe: pd.DataFrame):
        # Split to data
        ohlc_df = dataframe.drop(['volume'], axis=1)
        volume_df = dataframe[['volume']]

        # Scale down OHLC data
        ohlc_normalized = (ohlc_df - self.min_val) / (self.max_val - self.min_val)
        ohlc_scaled = ohlc_normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        ohlc_scaled_df = pd.DataFrame(ohlc_scaled,
                                      columns=ohlc_df.columns,
                                      index=dataframe.index)

        # Split to volume
        volume_scaled = self.volume_scaler.transform(volume_df.values.reshape(-1, 1))
        volume_scaled_df = pd.DataFrame(volume_scaled,
                                        columns=['volume'],
                                        index=dataframe.index)

        out = pd.concat([ohlc_scaled_df, volume_scaled_df], axis=1)
        return out[sorted(out.columns)]

    def fit_transform(self, dataframe: pd.DataFrame):
        self.fit(dataframe=dataframe)
        return self.transform(dataframe=dataframe)


class TimeFrame:
    def __init__(self, timeframe: int, window: int, ma_lengths: Optional[List[int]] = None):
        # Parameters
        self.timeframe: int = timeframe
        self.window: int = window
        self.ma_lengths: List[int] = [4, 8, 15, 30, 50] if ma_lengths is None else ma_lengths

        # Data
        self.price_data: pd.DataFrame = pd.DataFrame()
        self.scaled_data: pd.DataFrame = pd.DataFrame()

        # Timeseries estimators
        self.estimator_window = 100
        self.estimators = None

    def process_data(self, df: pd.DataFrame) -> None:
        price_data = df.groupby(pd.Grouper(freq=f'{self.timeframe}T')).agg(
            {
                'open': 'first',
                'close': 'last',
                'low': 'min',
                'high': 'max',
                'volume': 'sum',
            })
        # Drop NaN values
        self.price_data = price_data.dropna(axis=0)

        # Scale the data fit the data scaler
        data_scaler = OHLCVMinMaxScaler()
        data_scaled = data_scaler.fit_transform(dataframe=self.price_data)
        data_scaled.columns = [f'scaled_{column}' for column in data_scaled.columns]

        # Calculate moving averages
        target_labels = ['low', 'close', 'high']

        for label in target_labels:
            for ma_length in self.ma_lengths:
                data_scaled[f'scaled_{label}_{ma_length}'] = data_scaled.ta.hma(length=ma_length,
                                                                                close=f"scaled_{label}")
        self.scaled_data = data_scaled.dropna(axis=0)

    def __repr__(self) -> str:
        return f"TimeFrame: timeframe={self.timeframe} window={self.window} ma_lengths={self.ma_lengths}"

    def __call__(self, timeframe: int, window: int):
        self.__init__(timeframe=timeframe, window=window)

    def __getitem__(self, key: str) -> pd.DataFrame:
        key = key.split("_")
        index = int(key[-1])

        return self.price_data.iloc[index]

# class Renderer(TimeFrame):
#     def __init__(self, timeframe: int, window: int):
#         super().__init__(timeframe, window)
#         self._charts: Dict[int, go.Figure] = {}
#
#     def _render(self, data):
#         pass
#
#     @staticmethod
#     def fig_as_array(fig):
#         fig_bytes = fig.to_image(format="png")
#         buf = io.BytesIO(fig_bytes)
#         img = Image.open(buf)
#         return np.asarray(img) / 255
#
#
# class CandlestickProcessor(Renderer):
#
#     def process_train_data(self, key, df: pd.DataFrame):
#         super().process_train_data(key=key, df=df)
#         self._charts[key] = self.render(self._train_data[key])
#
#     def process_test_data(self, key, data_window: pd.DataFrame):
#         super().process_test_data(key=key, data_window=data_window)
#         self.render(self._test_data[key])
#
#     def render(self, data):
#         fig = make_subplots(rows=2, cols=1, vertical_spacing=0, row_width=[0.15, 0.85])
#
#         candlestick = go.Candlestick(
#             x=data.index,
#             open=data.open,
#             high=data.high,
#             low=data.low,
#             close=data.close,
#             increasing_line_color="green",
#             decreasing_line_color="red",
#             showlegend=False)
#
#         volume_bars = go.Bar(x=data.index, y=data.volume, showlegend=False, opacity=0.5,
#                              marker_color='blue', marker_line_width=0)
#
#         fig.add_trace(candlestick, row=1, col=1, )
#         fig.update_traces({'increasing': {'fillcolor': 'green',
#                                           'line': {'color': 'green',
#                                                    'width': 2}},
#                            'decreasing': {'fillcolor': 'red',
#                                           'line': {'color': 'red',
#                                                    'width': 2}}})
#
#         fig.add_trace(volume_bars, row=2, col=1)
#         fig.add_hline(y=data.iloc[-1].close, line_width=1.5, line_color="Blue", row=1,
#                       col=1)  # Highlight last price
#
#         fig.update_layout(width=self.window * 8,
#                           height=400,
#                           plot_bgcolor='rgba(0, 0, 0, 0)',
#                           paper_bgcolor='rgba(0, 0, 0, 0)',
#                           margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
#                           )
#
#         yaxis_density = 0.001  # Adjust this value to change the y-axis grid density
#
#         fig.update_xaxes(rangeslider_visible=False,
#                          visible=False,
#                          showticklabels=False,
#                          showgrid=False,
#                          gridwidth=0)
#
#         fig.update_yaxes(showgrid=True,
#                          gridwidth=.5,
#                          zerolinewidth=0,
#                          showticklabels=False,
#                          gridcolor='White',
#                          dtick=yaxis_density)
#
#         return fig
