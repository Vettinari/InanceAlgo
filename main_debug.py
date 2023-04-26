from pprint import pprint

import numpy as np
import pandas as pd
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import gymnasium as gym

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from positions import Position, Long
from reward_buffer import RewardBuffer
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

reward_buffer = RewardBuffer()
risk_manager = RiskManager(ticker=ticker,
                           initial_balance=10000,
                           atr_stop_loss_ratios=[2, 4],
                           risk_reward_ratios=[1.5, 2, 3],
                           position_closing=True,
                           portfolio_risk=0.02,
                           reward_buffer=reward_buffer)

data_pipeline = DataPipeline(ticker=ticker,
                             intervals=[15, 60, 240],
                             return_window=1,
                             chart_window=100,
                             test=True)

env = TradeGym(data_pipeline=data_pipeline,
               risk_manager=risk_manager,
               reward_scaling=0.99,
               verbose=5,
               wandb_logger=False)

seed = 42
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    print(risk_manager.info())
    done = False
    while not done:
        action = int(input("Choose action:"))
        if action == 9:
            break
        env.step(action=action)
        print(env.risk_manager.wallet.info())
