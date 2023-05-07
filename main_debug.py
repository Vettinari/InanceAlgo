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

seed = 42
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
    path='data_streams/15_60_240_data30000')

env = TradeGym(data_stream=data_stream,
               risk_manager=risk_manager,
               reward_buffer=reward_buffer,
               reward_scaling=0.99,
               verbose=1,
               wandb_logger=False,
               test=True,
               full_control=False)

if __name__ == '__main__':
    # risk_manager.info()
    print()
    done = False

    while not done:
        action = int(input("Choose action:"))
        if action == 9:
            break
        env.step(agent_action=action)
        risk_manager.wallet.info(short=True)
