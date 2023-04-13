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
    # Generate state
    rm.execute_action(action_index=2, current_atr=0.001)  # LONG
    # rm.rewards_info()
    # rm.wallet.info()

    print('\nNEXT STEP\n')
    rm.wallet_step(ohlct=dp.ohlct(2))
    # Generate state
    rm.execute_action(action_index=0, current_atr=0.001)  # LONG
    # rm.rewards_info()
    # rm.wallet.info()
    space = """"""
    # SHORTS
    # - (2)short intermediate_rew(4), long - OK
    # - (2)short intermediate_rew(4), short - OK
    # - (2)short intermediate_rew(4), hold - OK
    # - (2)short transaction_rew(4), close - OK
    # - (10)short stop_profit_rew(12), long - OK
    # - (10)short stop_profit_rew(12), short - OK
    # - (10)short stop_profit_rew(12), hold - OK
    # - (10)short stop_profit_rew(12), close - OK
    # - (0)short stop_loss_rew(2), long -
    # - (0)short stop_loss_rew(2), short -
    # - (0)short stop_loss_rew(2), hold -
    # close on empty wallet
    # hold on empty wallet

    # Repeating actions - OK
    # LONGS
    # - (0)long intermediate_rew(1), long - OK
    # - (0)long intermediate_rew(1), short - OK
    # - (0)long intermediate_rew(1), hold - OK
    # - (0)long transaction_rew(1), close - OK
    # - (0)long stop_profit_rew(2), long - OK
    # - (0)long stop_profit_rew(2), short - OK
    # - (0)long stop_profit_rew(2), hold - OK
    # - (0)long stop_profit_rew(2), close - OK
    # - (2)long stop_loss_rew(4), long - OK
    # - (2)long stop_loss_rew, short - OK
    # - (2)long stop_loss_rew, hold - OK

    # rm.wallet.info()

    #
    # {0: Action(long, sl: 1.5, rr: 1.5),
    # 1: Action(long, sl: 1.5, rr: 2),
    # 2: Action(short, sl: 1.5, rr: 1.5),
    # 3: Action(short, sl: 1.5, rr: 2),
    # 4: Action(hold, sl: None, rr: None),
    # 5: Action(close, sl: None, rr: None)}

    # Wallet class opens the position according to RiskManager specs
    # Environment updates the wallet, so it can check the position state.

    # position.update_position({'open': 1.0,
    #                           'high': 1.1,
    #                           'low': 0.95,
    #                           'close': 0.9,
    #                           'time': 1})
    # position.info()

    # ATR will give me the stop_level
    # risk will give me the position_volume
