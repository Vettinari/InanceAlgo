from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from DataProcessing.datastream import DataStream
from trade_env import DiscreteTradingEnv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

seed = 42
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
np.set_printoptions(suppress=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    datastream = DataStream(ticker='test',
                            timeframes=[15, 30],
                            output_window_length=2,
                            data_split=0.9,
                            ma_lengths=[5])

    env = DiscreteTradingEnv(datastream=datastream,
                             stop_loss_pips=80,
                             stop_profit_pips=20,
                             initial_balance=10000,
                             risk=0.1,
                             test=True)

    while True:
        action = int(input("Choose action:"))
        env.step(action=action)
