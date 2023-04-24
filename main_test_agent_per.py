import gymnasium
import numpy as np
import torch as T

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from reward_system import RewardBuffer
from risk_manager import RiskManager
from test_agent_per.agent import Agent

ticker = 'EURUSD'
device = 'cuda' if T.cuda.is_available() else 'cpu'

reward_buffer = RewardBuffer()

risk_manager = RiskManager(ticker=ticker,
                           initial_balance=10000,
                           atr_stop_loss_ratios=[2],
                           risk_reward_ratios=[1.5, 2, 3],
                           position_closing=True,
                           portfolio_risk=0.02,
                           reward_buffer=reward_buffer)

data_pipeline = DataPipeline(ticker=ticker,
                             intervals=[15, 60, 120],
                             return_window=1,
                             chart_window=100)

env = TradeGym(data_pipeline=data_pipeline,
               risk_manager=risk_manager,
               reward_scaling=1)

# env = gymnasium.make('CartPole-v1')

seed = 42
np.random.seed(seed=seed)
T.manual_seed(seed=seed)
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    num_games = 100
    agent = Agent(input_shape=env.observation_space.shape,
                  n_actions=env.action_space.n,
                  lr=0.003,
                  gamma=0.99,
                  epsilon=1.0,
                  epsilon_min=0.01,
                  epsilon_dec=0.0005,
                  mem_size=20000,
                  batch_size=128,
                  l1_dims=128,
                  l2_dims=64,
                  alpha=0.6,
                  beta=0.4,
                  beta_increment=0.001,
                  replace=200)

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
            agent.store_transition(state=observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            score += reward
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])

        print('     Episode: ', i,
              'score: %.1f ' % score,
              'average score: %.1f' % avg_score,
              'wallet balance:', env.risk_manager.wallet.total_balance
              )
