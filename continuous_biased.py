from pprint import pprint

import numpy as np
import pandas as pd
import torch
from DataProcessing.datastream import DataStream
from envs.biased.continuous import ContinuousTradingEnv

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
                            timeframes=[15],
                            output_window_length=1,
                            ma_lengths=[],
                            momentums=[],
                            momentum_noise_reduction=4,
                            local_extreme_orders=[],
                            data_split=0.8,
                            ma_type='sma',
                            )

    env = ContinuousTradingEnv(datastream=datastream,
                               initial_balance=10000,
                               agent_bias='short',
                               test=True,
                               scaler=0.0001)

    while True:
        env.info()
        print("State:", env.current_state)
        print("Reward:", env.reward)
        action = float(input("Type volume amount:"))
        env.step(action=action)
        print()
