import numpy as np
import torch as T

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from reward_system import RewardBuffer
from risk_manager import RiskManager
from test_agent.agent import Agent
from test_agent.network_builder import NetworkBuilderDDDQN

ticker = 'EURUSD'
device = 'cuda' if T.cuda.is_available() else 'cpu'

risk_manager = RiskManager(ticker=ticker,
                           initial_balance=10000,
                           atr_stop_loss_ratios=[2],
                           risk_reward_ratios=[1.5, 2, 3],
                           manual_position_closing=True,
                           portfolio_risk=0.02)

reward_manager = RewardBuffer()

data_pipeline = DataPipeline(ticker=ticker,
                             intervals=[15, 60, 240],
                             return_window=1,
                             chart_window=100)

env = TradeGym(data_pipeline=data_pipeline,
               risk_manager=risk_manager,
               reward_scaling=0.99)

if __name__ == '__main__':
    num_games = 10
    load_checkpoint = False

    agent = Agent(learning_rate=0.005,
                  n_actions=env.action_space.n,
                  input_shape=env.observation_space.shape,
                  l1_dims=32,
                  l2_dims=16,
                  memory_size=200000,
                  batch_size=512,
                  epsilon=1.0,
                  eps_min=0.01,
                  eps_dec=0.0005,
                  replace_target_counter=1000,
                  gamma=0.99)

    network_eval = NetworkBuilderDDDQN(input_shape=env.observation_space.shape, hidden_dims=[128, 128, 64],
                                       activation='relu', weight_init=True, learning_rate=0.005,
                                       n_actions=env.action_space.n, optimizer='adam', loss='mse', batch_norm=True,
                                       dropout=True)

    network_next = NetworkBuilderDDDQN(input_shape=env.observation_space.shape, hidden_dims=[128, 128, 64],
                                       activation='relu', weight_init=True, learning_rate=0.005,
                                       n_actions=env.action_space.n, optimizer='adam', loss='mse', batch_norm=True,
                                       dropout=True)

    agent.Q_eval = network_eval
    agent.Q_next = network_next

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
        observation_, reward, done, _truncated, info = env.step(action)
        agent.learn(state=observation, action=action, reward=reward, state_=observation_, done=done)

        score += reward
        observation = observation_

    scores.append(score)
    avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
    print('episode: ', i, 'score %.1f ' % score,
          'average score %.1f' % avg_score,
          'epsilon %.2f' % agent.epsilon)
