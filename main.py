import pandas as pd
import torch
import numpy as np
from position import DiscretePosition

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
np.random.seed(42)

ohlc_1 = {'open': 1.00, 'high': 1.008, 'low': 0.999, 'close': 1.001}

if __name__ == '__main__':
    pass
