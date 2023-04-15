import datetime

import pandas as pd
from tqdm import tqdm

import Utils
from DataProcessing.data_pipeline import DataPipeline
from DataProcessing.data_processor import DataProcessor, ChartProcessor
from env import TradeGym
from reward_system import TransactionReward
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # path = 'Datasets/forex/intraday/EURUSD.csv'
    # df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
    # df.iloc[-50000:].to_csv('EURUSD_short.csv')
    data_pipeline = DataPipeline(ticker='EURUSD',
                                 intervals=[15, 60, 240],
                                 window=60,
                                 processor_type=ChartProcessor)
    processor_data = data_pipeline.render_all_charts(limit=100)
