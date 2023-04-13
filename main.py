from positions import Long, Short
from risk_manager import RiskManager

if __name__ == '__main__':
    # RiskManager class gives all the details for opening a position to wallet
    ohlct = {'open': 0.95, 'high': 1.1, 'low': 0.92, 'close': 1.0, 'time': 0}
    rm = RiskManager(ticker='EURUSD', initial_balance=10000, portfolio_risk=0.02,
                     risk_reward_ratios=[1.5, 2, 3], atr_stop_loss_ratios=[1.5, 2.5],
                     manual_position_closing=True)
    rm.info()

    # rm.wallet_reset(current_ohlct=ohlct)
    #
    # rm.execute_action(action_index=0,
    #                   open_price=ohlct['close'],
    #                   current_atr=0.005)
    #
    # print('\nNEXT STEP\n')
    # # NEXT STEP
    # ohlct = {'open': 1.0, 'high': 1.011, 'low': 0.992, 'close': 1.01, 'time': 0}
    # rm.wallet_step(current_ohlct=ohlct)
    # rm.wallet.info()

    # {0: {'action': 'long', 'rr': 1.5, 'sl': 1.5},
    #  1: {'action': 'long', 'rr': 2, 'sl': 1.5},
    #  2: {'action': 'long', 'rr': 3, 'sl': 1.5},
    #  3: {'action': 'short', 'rr': 1.5, 'sl': 1.5},
    #  4: {'action': 'short', 'rr': 2, 'sl': 1.5},
    #  5: {'action': 'short', 'rr': 3, 'sl': 1.5},
    #  6: {'action': 'hold', 'rr': None, 'sl': None},
    #  7: {'action': 'close', 'rr': None, 'sl': None}}

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
