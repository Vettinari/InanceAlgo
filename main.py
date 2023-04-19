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
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    risk_manager = RiskManager(ticker=ticker,
                               initial_balance=10000,
                               atr_stop_loss_ratios=[2],
                               risk_reward_ratios=[1.5, 2, 3],
                               manual_position_closing=True,
                               portfolio_risk=0.01)

    data_pipeline = DataPipeline(ticker=ticker,
                                 intervals=[15, 60, 240],
                                 return_window=1,
                                 chart_window=100)

    env = TradeGym(data_pipeline=data_pipeline,
                   risk_manager=risk_manager,
                   reward_scaling=0.99)
    print(env.observation_space.sample())
    print(set(sorted([env.action_space.sample() for x in range(100)])))

    env = gym.make('CartPole-v1')
    print(env.observation_space.sample())
    print(set(sorted([env.action_space.sample() for x in range(100)])))
