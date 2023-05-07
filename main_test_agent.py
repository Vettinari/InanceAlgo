import gymnasium
import torch as T
from test_agent.agent import Agent
from test_agent.network_builder import NetworkBuilderDDDQN
import numpy as np
import pandas as pd
import torch
from DataProcessing.data_stream import BaseDataStream
from env import TradeGym
from reward_buffer import RewardBuffer
from risk_manager import RiskManager
from wallet import Wallet

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

seed = 40
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
np.set_printoptions(suppress=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

reward_buffer = RewardBuffer()

wallet = Wallet(ticker='EURUSD',
                initial_balance=10000,
                reward_buffer=reward_buffer)

risk_manager = RiskManager(wallet=wallet,
                           reward_buffer=reward_buffer,
                           use_atr=True,
                           stop_loss_ratios=[1.5],
                           risk_reward_ratios=[2],
                           portfolio_risk=0.01)

data_stream = BaseDataStream.load_datastream(
    path='/Users/milosz/Documents/Pycharm/InanceAlgo/data_streams/15_60_240_data30000')

train_env = TradeGym(data_stream=data_stream,
                     risk_manager=risk_manager,
                     reward_buffer=reward_buffer,
                     reward_scaling=0.99,
                     verbose=500,
                     wandb_logger=True,
                     full_control=True,
                     test=False)

# train_env = gymnasium.make('CartPole-v0')

if __name__ == '__main__':
    num_games = 100

    agent = Agent(learning_rate=0.001,
                  n_actions=train_env.action_space.n,
                  input_shape=train_env.observation_space.shape,
                  l1_dims=256,
                  l2_dims=128,
                  memory_size=200000,
                  batch_size=64,
                  epsilon=1.0,
                  eps_min=0.05,
                  eps_dec=0.0005,
                  replace_target_counter=1000,
                  gamma=0.99,
                  )

    # hidden_dims = np.array([256, 128, 64])
    # learning_rate = 0.005
    # agent.Q_next = NetworkBuilderDDDQN(input_shape=env.observation_space.shape,
    #                                    hidden_dims=hidden_dims,
    #                                    activation='relu',
    #                                    weight_init=True,
    #                                    learning_rate=learning_rate,
    #                                    n_actions=env.action_space.n,
    #                                    optimizer='adam',
    #                                    loss='mse',
    #                                    batch_norm=True,
    #                                    dropout=0.1)
    #
    # agent.Q_eval = NetworkBuilderDDDQN(input_shape=env.observation_space.shape,
    #                                    hidden_dims=hidden_dims,
    #                                    activation='relu',
    #                                    weight_init=True,
    #                                    learning_rate=learning_rate,
    #                                    n_actions=env.action_space.n,
    #                                    optimizer='adam',
    #                                    loss='mse',
    #                                    batch_norm=True,
    #                                    dropout=0.1)

    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation, _ = train_env.reset()
        train_score = 0

        while not done:
            observation = T.tensor(observation, dtype=T.float)
            action = agent.choose_action(observation)
            observation_, reward, _, done, info = train_env.step(action)
            agent.learn(state=observation, action=action, reward=reward, state_=observation_, done=done)
            train_score += reward
            observation = observation_

        print('Episode: ', i, 'train_score: %.1f ' % train_score)
