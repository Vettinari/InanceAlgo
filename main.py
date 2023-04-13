import pandas as pd

import Utils
from DataProcessing.data_processor import DataProcessor
from env import TradeGym
from reward_system import TransactionReward
from risk_manager import RiskManager

if __name__ == '__main__':
    # Utils.start_logging('log.txt')
    # RiskManager class gives all the details for opening a position to wallet
    dp = DataProcessor(ticker='EURUSD', intervals=[15, 60, 240])
    rm = RiskManager(ticker='EURUSD', initial_balance=10000, portfolio_risk=0.02,
                     risk_reward_ratios=[1.5, 2], atr_stop_loss_ratios=[1.5],
                     manual_position_closing=True)
    rm.wallet_reset(ohlct=dp.ohlct(0))
    rm.execute_action(action_index=0, current_atr=0.001)  # LONG
    rm.wallet.info()
    rm.rewards_info()
    print('\nNEXT STEP\n')
    rm.wallet_step(ohlct=dp.ohlct(1))
    rm.execute_action(action_index=4, current_atr=0.001)  # LONG
    rm.wallet.info()
    rm.rewards_info()

    # Repeating actions - OK
    # LONG/SHORT TRANSACTION REWARD - OK
    # LONG/SHORT INTERMEDIATE REWARD -

    # rm.wallet.info()

    #
    # {0: Action(long, sl: 1.5, rr: 1.5),
    # 1: Action(long, sl: 1.5, rr: 2),
    # 2: Action(short, sl: 1.5, rr: 1.5),
    # 3: Action(short, sl: 1.5, rr: 2),
    # 4: Action(hold, sl: None, rr: None),
    # 5: Action(close, sl: None, rr: None)}

    # Wallet class opens the position according to RiskManager specs
    # Environment updates the wallet so it can check the position state.

    # position.update_position({'open': 1.0,
    #                           'high': 1.1,
    #                           'low': 0.95,
    #                           'close': 0.9,
    #                           'time': 1})
    # position.info()

    # ATR will give me the stop_level
    # risk will give me the position_volume
