import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch

import Utils
from DataProcessing.data_stream import DataStream
from DataProcessing.processors import DataProcessor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # now = datetime.datetime.now()
    # ds = DataStream.load_datastream(path='data_streams/15_60_240')
    # print(f"This took {datetime.datetime.now() - now}")
    # print(ds)
    ds = DataStream(ticker='EURUSD',
                    timeframes=[15, 60, 240],
                    technicals=['ema_5'],
                    processor_window_output=100,
                    data_processor_type=DataProcessor,
                    data_size=30000,
                    data_split=0.9)
    ds.run_multithread()
    ds.save_datastream()

    # print(ds.processors)
