import copy
import gymnasium as gym
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from risk_manager import RiskManager

ticker = 'EURUSD'

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

trade_gym = TradeGym(data_pipeline=data_pipeline,
                     risk_manager=risk_manager,
                     reward_scaling=0.99)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # environments
    # env = copy.deepcopy(trade_gym)

    # environments
    env = gym.make('CartPole-v1')
    train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(1)])

    # model & optimizer
    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
    actor = Actor(net, *env.action_space.shape, device=device).to(device)
    # net = Net(env.observation_space.shape, hidden_sizes=[16, 16], device=device)
    # actor = Actor(net, env.action_space.shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    # PPO policy
    dist = torch.distributions.Categorical
    policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)

    # collector
    train_collector = Collector(policy,
                                train_envs,
                                PrioritizedVectorReplayBuffer(total_size=20000,
                                                              buffer_num=len(train_envs),
                                                              alpha=0.6,
                                                              beta=0.4)
                                )
    test_collector = Collector(policy,
                               test_envs)

    # trainer

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 195,
    )
    print(result)
