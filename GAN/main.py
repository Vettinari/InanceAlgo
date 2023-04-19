import numpy as np
import pandas as pd
from gym.vector.utils import spaces

from DataProcessing.data_pipeline import DataPipeline
from env import TradeGym
from positions import Position, Long
from risk_manager import RiskManager

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ticker = 'EURUSD'

if __name__ == '__main__':
    pass
