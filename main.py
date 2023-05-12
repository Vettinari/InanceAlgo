import os

import pandas as pd
import torch

import Utils
from DataProcessing.data_stream import DataStream
from DataProcessing.processors import DataProcessor
from concurrent.futures import ThreadPoolExecutor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

dataframes = {}


def process_df(file_path):
    df = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
    df.dropna(axis=0, inplace=True)
    ticker = file_path.split("/")[-1].rstrip(".csv")
    dataframes[ticker] = df.iloc[-100:]
    print(f"Ticker: {ticker} - finished!")


if __name__ == '__main__':
    dataset_folder = "/Users/milosz/Documents/Pycharm/InanceAlgo/Datasets/forex/intraday"
    file_list = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith(".csv")]
    with ThreadPoolExecutor() as executor:
        executor.map(process_df, file_list)