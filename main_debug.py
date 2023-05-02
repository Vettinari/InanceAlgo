import numpy as np
import pandas as pd
import torch

from old_DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from risk_manager import RiskManager, ActionValidator

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_validator = ActionValidator(position_reversing=False,
                                   position_closing=True,
                                   action_penalty=-0.001)

risk_manager = RiskManager(ticker=ticker,
                           initial_balance=10000,
                           atr_stop_loss_ratios=[3],
                           risk_reward_ratios=[1.5],
                           portfolio_risk=0.02)

data_pipeline = DataPipeline(ticker=ticker,
                             intervals=[15, 60, 240],
                             return_window=1,
                             chart_window=100,
                             test=True)

env = TradeGym(data_pipeline=data_pipeline,
               risk_manager=risk_manager,
               action_validator=action_validator,
               reward_scaling=0.99,
               verbose=250,
               wandb_logger=False,
               test=True)

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
        env.step(agent_action=action)
        print(env.risk_manager.wallet)
