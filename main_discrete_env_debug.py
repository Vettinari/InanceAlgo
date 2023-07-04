import numpy as np
import pandas as pd
import torch

from DataProcessing.datastream import DataStream
from envs.biased.discrete import DiscreteTradingEnv

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
                            timeframes=[5],
                            output_window_length=10,
                            ma_lengths=[5, 10, 15, 20],
                            momentums=[4, 8, 12],
                            momentum_noise_reduction=4,
                            local_extreme_orders=[],
                            data_split=0.8,
                            ma_type='sma',
                            )

    env = DiscreteTradingEnv(datastream=datastream,
                             initial_balance=10000,
                             test=True)

    while True:
        action = float(input("Type action"))
        env.step(action=action)
