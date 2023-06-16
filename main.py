import pandas as pd
import torch
import numpy as np
from position import Position

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
np.random.seed(42)

ohlc_1 = {'open': 1.00, 'high': 1.008, 'low': 0.999, 'close': 1.001}

if __name__ == '__main__':
    position = Position(order_type="short", ticker='EURUSD', open_time="2020-01-01",
                        open_price=1.00, stop_loss_pips=80, stop_profit_pips=20,
                        leverage=30, one_pip=0.0001, position_margin=1000)
    position.info()
    print(position)
    position.check_stops(ohlc_dict=ohlc_1)
    print(position)
