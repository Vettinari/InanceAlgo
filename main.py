import datetime
import time
from pprint import pprint

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch

import torch
import torch.nn as nn

import Utils
from DataProcessing.data_stream import BaseDataStream
from DataProcessing.processors import DataProcessor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

if __name__ == '__main__':
    ds = BaseDataStream(ticker=ticker,
                        timeframes=[15],
                        data_processor_type=DataProcessor,
                        technicals=['ema', 'bbands', 'rsi', 'stoch'],
                        processor_window_output=1,
                        test=True,
                        data_split=0.8,
                        data_size=None)
