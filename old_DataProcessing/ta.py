from typing import Optional, Dict, List

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


class TechnicalIndicator:

    def __init__(self, indicator, params):
        self.indicator = indicator
        self.params = params

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.params.update({'close': data.close,
                            'high': data.high,
                            'low': data.low,
                            'open': data.open,
                            'volume': data.volume})
        tech_data = getattr(ta, self.indicator)(**self.params)

        if type(tech_data) == pd.Series:
            tech_data.name = str(tech_data.name).lower()
        else:
            tech_data.columns = [column.lower() for column in tech_data.columns]

        return tech_data
