import gymnasium
import numpy as np
import torch as T

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from reward_buffer import RewardBuffer
from risk_manager import RiskManager
from test_agent.agent import Agent
from test_agent.network_builder import NetworkBuilderDDDQN

ticker = 'EURUSD'
device = 'cuda' if T.cuda.is_available() else 'cpu'
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
                             test=False)

env = TradeGym(data_pipeline=data_pipeline,
               risk_manager=risk_manager,
               reward_scaling=0.99,
               verbose=250,
               wandb_logger=True)

seed = 42
np.random.seed(seed=seed)
T.manual_seed(seed=seed)
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    num_games = 100

    agent = Agent(learning_rate=0.001,
                  n_actions=env.action_space.n,
                  input_shape=env.observation_space.shape,
                  l1_dims=256,
                  l2_dims=128,
                  memory_size=200000,
                  batch_size=128,
                  epsilon=1.0,
                  eps_min=0.05,
                  eps_dec=0.0005,
                  replace_target_counter=250,
                  gamma=0.99)

    hidden_dims = np.array([256, 128, 64])
    learning_rate = 0.005

    agent.Q_next = NetworkBuilderDDDQN(input_shape=env.observation_space.shape,
                                       hidden_dims=hidden_dims,
                                       activation='relu',
                                       weight_init=True,
                                       learning_rate=learning_rate,
                                       n_actions=env.action_space.n,
                                       optimizer='adam', loss='mse',
                                       batch_norm=True, dropout=False)

    agent.Q_eval = NetworkBuilderDDDQN(input_shape=env.observation_space.shape,
                                       hidden_dims=hidden_dims,
                                       activation='relu',
                                       weight_init=True,
                                       learning_rate=learning_rate,
                                       n_actions=env.action_space.n,
                                       optimizer='adam', loss='mse',
                                       batch_norm=True, dropout=False)

    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            observation = T.tensor(observation, dtype=T.float)
            action = agent.choose_action(observation)
            observation_, reward, _, done, info = env.step(action)
            agent.learn(state=observation, action=action, reward=reward, state_=observation_, done=done)
            score += reward
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])

        print('Episode: ', i,
              'score: %.1f ' % score,
              'average score: %.1f' % avg_score,
              'wallet balance:', env.risk_manager.wallet.total_balance
              )
