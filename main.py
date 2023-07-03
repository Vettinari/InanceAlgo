from pprint import pprint

from positions.continuous import ContinuousPosition
from positions.discrete import DiscretePosition

if __name__ == '__main__':
    position = DiscretePosition(ticker='EURUSD',
                                scaler=0.000001,
                                stop_loss_pips=80,
                                stop_profit_pips=20,
                                risk=146.87)
    position.open_position(open_price=1.1, position_type='long')
    position.info()

    ohlc = {'close': 1.1,
            'open': 1.1,
            'high': 1.2,
            'low': 1.}

    out = position.check_stops(ohlc_dict=ohlc)
    print(out)
