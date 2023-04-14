import pandas as pd

import Utils
from DataProcessing.data_generator import DataPipeline
from DataProcessing.data_processor import DataProcessor, ChartProcessor
from env import TradeGym
from reward_system import TransactionReward
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # Utils.start_logging('log.txt')
    data_pipeline = DataPipeline(ticker='EURUSD',
                                 intervals=[15, 60, 240],
                                 window=60,
                                 processor_type=ChartProcessor)
    data_pipeline.render_all_charts()
    # Rendering took 0:02:09.620930 seconds

    # print("-------------------------------------------")
    # print(data_pipeline)
    # data_pipeline.get_current_data()
    # print("-------------------------------------------")
    # print(data_pipeline)
    # data_pipeline.get_current_data()
    # print("-------------------------------------------")
    # print(data_pipeline)
    # data_pipeline.get_current_data()
    # print("-------------------------------------------")
    # print(data_pipeline)
    # data_pipeline.get_current_data()

    # data_pipeline.get_current_data()
    # data_processor = DataProcessor(ticker='EURUSD', intervals=[15, 60, 240])
    # risk_manager = RiskManager(ticker='EURUSD', initial_balance=10000, portfolio_risk=0.01,
    #                            risk_reward_ratios=[1.5, 2], atr_stop_loss_ratios=[1.5],
    #                            manual_position_closing=True)
    # print(risk_manager)
