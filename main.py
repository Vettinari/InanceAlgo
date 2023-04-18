import pandas as pd
from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    ticker = 'EURUSD'
    data_pipeline = DataPipeline(ticker=ticker,
                                 intervals=[15, 60, 240],
                                 data_window=1,
                                 chart_window=100)
    risk_manager = RiskManager(ticker=ticker,
                               initial_balance=10000,
                               atr_stop_loss_ratios=[2],
                               risk_reward_ratios=[1.5, 2, 3],
                               manual_position_closing=True,
                               portfolio_risk=0.01)

    trade_gym = TradeGym(data_pipeline=data_pipeline,
                         risk_manager=risk_manager,
                         reward_scaling=0.99)

    run = True
    while run:
        action = int(input("Choose action:"))
        trade_gym.step(action=action)
        if action == -1:
            run = False
            print('End of the game!')
