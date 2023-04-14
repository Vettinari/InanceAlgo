import pandas as pd

import Utils
from DataProcessing.data_processor import DataProcessor
from env import TradeGym
from reward_system import TransactionReward
from risk_manager import RiskManager

if __name__ == '__main__':
    data_manager = DataGenerator()
    data_processor = DataProcessor(ticker='EURUSD', intervals=[15, 60, 240])
    risk_manager = RiskManager(ticker='EURUSD', initial_balance=10000, portfolio_risk=0.01,
                               risk_reward_ratios=[1.5, 2], atr_stop_loss_ratios=[1.5],
                               manual_position_closing=True)
    print(risk_manager)
