from pprint import pprint

from positions.continuous import ContinuousPosition
from positions.discrete import DiscretePosition

if __name__ == '__main__':
    position = ContinuousPosition(ticker='EURUSD',
                                  scaler=0.000001)
    position.modify_position(current_price=1.12345, volume=-0.1)
    position.modify_position(current_price=1.12345, volume=0.2)
    position.modify_position(current_price=1.12345, volume=-0.1)
    position.modify_position(current_price=1.12345, volume=-0.1)