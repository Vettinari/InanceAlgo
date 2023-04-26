from pprint import pprint

from IPython.display import display, HTML
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
from tqdm import tqdm

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from experimental.data_buffering import DataBuffer
from positions import Position, Long
from reward_system import RewardBuffer, SharpeReward, ConsistencyReward, ProfitFactorReward, DrawdownReward, TrendReward
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
                           portfolio_risk=0.01,
                           reward_buffer=reward_buffer)

if __name__ == '__main__':
    print(pd.concat([pd.DataFrame({"action_type": ['a', 'b']}),
                     pd.DataFrame({"action_type": ['c']})],
                    axis=0, ignore_index=True))
