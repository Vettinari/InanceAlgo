import collections
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from datetime import timedelta
from Utils import concurrent_execution, printline


class AbstractProcessor:
    def __init__(self,
                 ticker,
                 test,
                 processor_config,
                 base_dataframe=None,
                 start_date=None,
                 xtb_mode=False):
        self.test = test
        self.processor_config = processor_config
        self.live_mode = xtb_mode

        # Base info
        self.ticker = ticker
        self.start_date = '1990-01-01' if start_date is None else start_date
        self.processor_type = 'abstract'
        self.technical_indicators = self.processor_config['technical_indicators']

        # Intraday dataframes
        self.base_dataframe = base_dataframe
        self.interval_dataframes = {}
        self.interval_tech = {}
        self.time_dataframe = {}
        self.intervals = self.processor_config['intervals']

        # Step_dict_config
        self.step_size = self.processor_config['step_size']

    def run(self):
        printline(text=f"{self.ticker} processor start", title=True, test=self.test)
        if self.base_dataframe is None:
            self.__load_data()  # Load data from folders - OK
        self.base_dataframe = self.base_dataframe.loc[self.start_date:]
        if self.live_mode is False:
            printline('Marking forex days', line_char='-', test=self.test)
            self.__tailor_data_to_xtb()
        printline('Grouping intraday frames', line_char='-', test=self.test)
        self.__group_dataframes()  # Create time interval groups - OK
        printline('Intraday indicators', line_char='-', test=self.test)
        self.__generate_tech_dict()
        printline('Market opening', line_char='-', test=self.test)
        self.__apply_market_opening()  # Apply market opening times
        printline('Adding time', line_char='-', test=self.test)
        self.__create_time_df()
        printline('Cleaning', line_char='-', test=self.test)
        self.__sort_keys()
        self.__mark_column_names_with_intervals()  # Change columns names in time intervals
        self.__clip_dataframes()
        self.__delete_incomplete_rows()

        printline(f'{self.ticker} processor finished', test=self.test)

    def __delete_incomplete_rows(self):
        for key, data in self.interval_dataframes.items():
            # If last available index datetime with addon of its key is higher than now then loose last row
            if (data.index[-1] + timedelta(minutes=key)) > datetime.now():
                self.interval_dataframes[key] = self.interval_dataframes[key].iloc[:-1]

        for key, tech in self.interval_tech.items():
            # If last available index datetime with addon of its key is higher than now then loose last row
            if (tech.index[-1] + timedelta(minutes=key)) > datetime.now():
                self.interval_tech[key] = self.interval_tech[key].iloc[:-1]

    def update_run(self, new_dataframe):
        self.base_dataframe = new_dataframe
        self.__group_dataframes()
        self.__generate_tech_dict()
        self.__apply_market_opening()
        self.__create_sin_cos_linear_time_from_first_day()
        self.__sort_keys()
        self.__mark_column_names_with_intervals()
        self.__clip_dataframes()

    # INTRADAY FUNCTIONS
    def __load_data(self):
        self.base_dataframe = pd.read_csv(
            f'{DATASET_PATH}/{self.processor_type}/intraday/{self.ticker}.csv',
            parse_dates=['Datetime'], index_col='Datetime')
        if self.start_date:
            self.base_dataframe = self.base_dataframe.loc[self.start_date:]
        self.print(f'Intraday dataframe loaded! Shape: {self.base_dataframe.shape}')

    def __group_dataframes(self):
        def __group_by_interval(interval):
            self.interval_dataframes[interval] = self.base_dataframe.groupby(
                pd.Grouper(freq=f'{interval}T')).agg({'open': 'first',
                                                      'close': 'last',
                                                      'low': 'min',
                                                      'high': 'max',
                                                      'volume': 'sum'})
            # ADDED TO REMOVE NAN
            self.interval_dataframes[interval].dropna(inplace=True)
            self.print(f'    {interval} grouping finished! - Shape: {self.interval_dataframes[interval].shape}')

        concurrent_execution(__group_by_interval, self.intervals)

    def __sort_keys(self):
        sorted_dict = collections.OrderedDict(sorted(self.interval_dataframes.items()))
        self.interval_dataframes = dict(sorted_dict)

        sorted_dict = collections.OrderedDict(sorted(self.interval_tech.items()))
        self.interval_tech = dict(sorted_dict)

    def __generate_tech_dict(self):
        for key in list(self.interval_dataframes.keys()):
            key_out = []
            for indicator in self.processor_config['technical_indicators']:

                if indicator == 'sma':
                    temp_ta = self.interval_dataframes[key].ta.sma(length=50, close="close")
                    key_out.append(temp_ta)

                if indicator == 'atr':
                    key_out.append(self.interval_dataframes[key].ta.atr(close='close'))

                if indicator == 'rsi':
                    key_out.append(self.interval_dataframes[key].ta.rsi(close='close'))

                if indicator == 'bbands':
                    temp_ta = self.interval_dataframes[key].ta.bbands(close='close')
                    temp_ta['bbands_delta_up'] = temp_ta["BBU_5_2.0"] - self.interval_dataframes[key].close
                    temp_ta['bbands_delta_bottom'] = self.interval_dataframes[key].close - temp_ta["BBL_5_2.0"]
                    temp_ta.drop(["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "BBB_5_2.0", "BBP_5_2.0"], axis=1,
                                 inplace=True)
                    key_out.append(temp_ta)

                if indicator == 'macd':
                    key_out.append(self.interval_dataframes[key].ta.macd(close='close', fast=5, slow=34, signal=5))

                if indicator == 'stochastic':
                    key_out.append(self.interval_dataframes[key].ta.stoch(high='high', low='low', close='close',
                                                                          fast_k=14, slow_k=3, slow_d=3))
                if indicator == 'cci':
                    key_out.append(self.interval_dataframes[key].ta.cci(high='high', low='low', close='close'))

                if indicator == 'adx':
                    key_out.append(self.interval_dataframes[key].ta.adx(high='high', low='low', close='close'))

                if indicator == 'williams':
                    key_out.append(self.interval_dataframes[key].ta.willr(high='high', low='low', close='close'))

                if indicator == 'momentum':
                    key_out.append(self.interval_dataframes[key].ta.mom(close='close'))

                if indicator == 'directional_index':
                    key_out.append(self.interval_dataframes[key].ta.dm(high='high', low='low'))

                if indicator == 'candlesticks':
                    pass

            temp = pd.concat(key_out, axis=1)
            temp.columns = [f"{column.lower()}_{key}min" for column in temp.columns]
            temp.fillna(method="bfill", inplace=True)
            self.print(f"    {key} indicator finished! - Shape: {temp.shape}")
            self.interval_tech[key] = temp

    def __mark_column_names_with_intervals(self):
        def intraday_func(key):
            self.interval_dataframes[key].columns = [f'{column.lower()}_{key}min' for column in
                                                     self.interval_dataframes[key]]

        concurrent_execution(intraday_func, self.intervals)
        self.print(f"   Column example:\n"
                   f"   {list(self.interval_dataframes[self.intervals[0]].columns)}\n"
                   f"   {list(self.interval_dataframes[self.intervals[-1]].columns)}")

    def __create_sin_cos_linear_time_from_first_day(self):
        df = self.time_dataframe

        generated_idx = pd.date_range(df.index[0], df.index[-1], freq=f"{self.step_size}T")
        df = df.reindex(generated_idx, fill_value=np.nan)

        one_day_margin = int(24 * 60 / self.step_size)

        first_week_start_loc = None
        first_week_end_loc = None

        for idx in list(df.index):
            if idx.dayofweek == 0:
                first_week_start_loc = idx
                break

        for idx in list(df.loc[first_week_start_loc:].index)[one_day_margin:]:
            if idx.dayofweek == 0:
                first_week_end_loc = idx
                break

        # Calculate steps in a week
        first_week_start_iloc, first_week_end_iloc = df.index.get_indexer([first_week_start_loc, first_week_end_loc])
        steps_in_week = first_week_end_iloc - first_week_start_iloc

        # Trim start of DF to first monday
        df = df.iloc[first_week_start_iloc:]

        # Create x values for linear, sin and cos
        x_linear = np.arange(0, 1, 1 / steps_in_week)
        x_sin_cos = np.arange(0, 2 * np.pi, (2 * np.pi) / steps_in_week)

        # How many times tile the x values and sin/cos functions
        # Trim functions to the size of dataframe
        weeks = int(df.shape[0] / steps_in_week) + 1
        linear_weekly = np.tile(x_linear, weeks)[:df.shape[0]]
        sin_weekly = np.tile(np.sin(x_sin_cos), weeks)[:df.shape[0]]
        cos_weekly = np.tile(np.cos(x_sin_cos), weeks)[:df.shape[0]]

        # Add sin, cos and linear to df
        df['sin_weekly'] = sin_weekly
        df['cos_weekly'] = cos_weekly
        df['linear_weekly'] = linear_weekly

        def create_daily_time():
            steps_in_day = 24 * 60 / self.step_size

            x_linear_daily = np.arange(0, 1, 1 / steps_in_day)
            x_sin_cos_daily = np.arange(0, 2 * np.pi, (2 * np.pi) / steps_in_day)

            days = int(df.shape[0] / steps_in_day) + 1
            linear_daily = np.tile(x_linear_daily, days)[:df.shape[0]]
            sin_daily = np.tile(np.sin(x_sin_cos_daily), days)[:df.shape[0]]
            cos_daily = np.tile(np.cos(x_sin_cos_daily), days)[:df.shape[0]]

            # Add sin, cos and linear to df
            df['sin_daily'] = sin_daily
            df['cos_daily'] = cos_daily
            df['linear_daily'] = linear_daily

        create_daily_time()

        # Clean extra indexes that are np.nan
        df.dropna(inplace=True, axis=0)

        # Overwrite dataframe
        self.time_dataframe = df

    def __apply_market_opening(self):
        temp_df = self.interval_dataframes[self.step_size].copy(deep=True)
        temp_df['time'] = self.interval_dataframes[self.step_size].copy(deep=True).index

        def mark_australia(row):
            # FROM 23:00 to 8:00
            current_time = row.time.time()

            if self.is_forex_day(row=row):
                if current_time.hour == 23 or 0 <= current_time.hour < 8:
                    return 1
                else:
                    return 0
            else:
                return 0

        def mark_tokyo(row):
            # FROM 01:00 to 10:00
            current_time = row.time.time()

            if self.is_forex_day(row=row):
                if 1 <= current_time.hour < 10:
                    return 1
                else:
                    return 0
            else:
                return 0

        def mark_europe(row):
            # FROM 08:00 to 17:00
            current_time = row.time.time()

            if self.is_forex_day(row=row):
                if 8 <= current_time.hour < 17:
                    return 1
                else:
                    return 0
            else:
                return 0

        def mark_london(row):
            # FROM 09:00 to 18:00
            current_time = row.time.time()

            if self.is_forex_day(row=row):
                if 9 <= current_time.hour < 18:
                    return 1
                else:
                    return 0
            else:
                return 0

        def mark_ny(row):
            # FROM 14:00 to 23:00
            current_time = row.time.time()

            if self.is_forex_day(row=row):
                if 14 <= current_time.hour < 23:
                    return 1
                else:
                    return 0
            else:
                return 0

        temp_df['australia_market'] = temp_df.apply(mark_australia, axis=1)
        self.print('    Australia - marked')
        temp_df['tokyo_market'] = temp_df.apply(mark_tokyo, axis=1)
        self.print('    Tokyo - marked')
        temp_df['europe_market'] = temp_df.apply(mark_europe, axis=1)
        self.print('    Europe - marked')
        temp_df['london_market'] = temp_df.apply(mark_london, axis=1)
        self.print('    London - marked')
        temp_df['ny_market'] = temp_df.apply(mark_ny, axis=1)
        self.print('    New York - marked')

        temp_df.pop('time')
        temp_df = temp_df[['australia_market', 'tokyo_market', 'europe_market', 'london_market', 'ny_market']]
        self.time_dataframe = temp_df
        self.print(f"Time dataframe finished! - Shape: {self.time_dataframe.shape}")

    def __clip_dataframes(self):
        start_data = str(self.time_dataframe.index[0])
        for key in self.intervals:
            self.interval_dataframes[key] = self.interval_dataframes[key].loc[start_data:]
            self.interval_tech[key] = self.interval_tech[key].loc[start_data:]

    def print(self, text):
        if self.test:
            print(text)
        else:
            pass

    def __tailor_data_to_xtb(self, hours=1, minutes=0):
        interval = 5
        if self.processor_config['step_size'] == 5:
            interval = 5
        elif self.processor_config['step_size'] == 15:
            interval = 5
        elif self.processor_config['step_size'] == 30:
            interval = 15
        elif self.processor_config['step_size'] == 60:
            interval = 30

        self.base_dataframe = self.base_dataframe.groupby(
            pd.Grouper(freq=f'{interval}T')).agg({'open': 'first',
                                                  'close': 'last',
                                                  'low': 'min',
                                                  'high': 'max',
                                                  'volume': 'sum'})

        self.base_dataframe.index = self.base_dataframe.index + timedelta(hours=hours, minutes=minutes)
        self.base_dataframe['forex_open'] = self.base_dataframe.apply(self.is_forex_day, axis=1)
        self.apply_nan_to_closed_market()

    @staticmethod
    def is_forex_day(row):
        start_time = 23
        end_time = 22

        # MARK WORKING DAYS
        if row.name.dayofweek == 0:
            return True
        elif row.name.dayofweek == 1:
            return True
        elif row.name.dayofweek == 2:
            return True
        elif row.name.dayofweek == 3:
            return True
        elif row.name.dayofweek == 4 and row.name.time().hour < end_time:
            return True
        elif row.name.dayofweek == 6 and row.name.time().hour >= start_time:
            return True
        else:
            return False

    def apply_nan_to_closed_market(self):
        self.base_dataframe[self.base_dataframe['forex_open'] == False] = np.NaN

    def __create_time_df(self):
        df = self.time_dataframe

        df['hour'] = df.index.hour / 24
        df['minute'] = df.index.minute / 60
        df['day_of_week'] = df.index.dayofweek / 6

        # Overwrite dataframe
        self.time_dataframe = df
