import Utils
import pandas as pd

from DataProcessing.data_processor import OHLCT
from positions import Position, Long, Short
from reward_system import TransactionReward, IntermediateReward


class Wallet:
    def __init__(self,
                 ticker,
                 initial_balance):

        self.closed_positions = []

        # Market specific values
        self.ticker = ticker.upper()
        self.initial_balance = initial_balance
        self.margin_balance = {"free": self.initial_balance, "margin": 0}

        self.position: Position = None
        self.current_ohlct: OHLCT = None
        self.closed_positions_dataframe = pd.DataFrame({"open_time": [],
                                                        "close_time": [],
                                                        "volume": [],
                                                        "open_price": [],
                                                        "close_price": [],
                                                        "stop_loss": [],
                                                        "stop_profit": [],
                                                        "contract_value": [],
                                                        "profit": [],
                                                        "margin": [],
                                                        "is_closed": [],
                                                        "is_stop_loss": [],
                                                        "is_profit": []}, index=[])
        self.game_over = False
        self.transaction_reward: TransactionReward = None
        self.intermediate_reward: IntermediateReward = None

    def reserve_margin(self, amount: float):
        if amount > self.margin_balance['free']:
            self.game_over = True
            return
        else:
            self.margin_balance['free'] -= amount
            self.margin_balance['margin'] = amount
            return

    def free_margin(self):
        self.margin_balance['free'] += self.margin_balance['margin']
        self.margin_balance['margin'] = 0

    def open_long(self, stop_loss_delta: float, risk_reward_ratio: float, position_risk: float):
        transaction_reward = self._prepare_to_open_position(cancel_position_type='short')  # Generates wallet reward
        self.position = Long(ticker=self.ticker,
                             open_time=self.current_ohlct.time,
                             open_price=self.current_ohlct.close,
                             stop_loss=self.current_ohlct.close - stop_loss_delta,
                             risk_reward_ratio=risk_reward_ratio,
                             position_risk=position_risk)
        self.reserve_margin(amount=self.position.margin)
        return transaction_reward

    def open_short(self, stop_loss_delta: float, risk_reward_ratio: float, position_risk: float):
        transaction_reward = self._prepare_to_open_position(cancel_position_type='long')  # Generates wallet reward
        self.position = Short(ticker=self.ticker,
                              open_time=self.current_ohlct.time,
                              open_price=self.current_ohlct.close,
                              stop_loss=self.current_ohlct.close + stop_loss_delta,
                              risk_reward_ratio=risk_reward_ratio,
                              position_risk=position_risk)
        self.reserve_margin(amount=self.position.margin)
        return transaction_reward

    def position_close(self):
        self.position.is_closed = True
        self.position.close_price = self.current_ohlct.close
        self.position.close_time = self.current_ohlct.time
        self.margin_balance['free'] += self.position.profit
        self.free_margin()
        self.__update_dataframe(position=self.position)
        transaction_reward = TransactionReward(position=self.position)
        self.position = None

        return transaction_reward

    def update_wallet(self, ohlct: OHLCT):
        # Update ohlct
        self.current_ohlct = ohlct
        self.update_position()
        self.check_and_close_position()

    def update_position(self):
        if self.position is not None:
            self.position.update_position(ohlct=self.current_ohlct)
        self.update_rewards()

    def check_and_close_position(self):
        if self.position is not None and self.position.is_closed:
            self.margin_balance['free'] += self.position.profit
            self.free_margin()
            self.__update_dataframe(position=self.position)
            self.position = None

    def update_rewards(self):
        self.intermediate_reward = IntermediateReward(position=self.position)
        self.transaction_reward = TransactionReward(position=self.position)

    def get_current_state(self):
        current_position_type = [int(self.position.type == 'long'), int(self.position.type == 'short')]
        position_state = [0, 0, 0] if self.position is None else self.position.state
        state = [self.total_balance / (self.initial_balance * 10),
                 current_position_type,
                 position_state]

        return state

    @property
    def cash(self) -> float:
        return self.margin_balance['free']

    @property
    def total_balance(self) -> float:
        return self.margin_balance['free'] + self.margin_balance['margin']

    def info(self) -> None:
        if not self.game_over:
            Utils.printline(text='Wallet info', title=False, line_char=":", size=60)
            print(f"Free margin: {self.margin_balance['free']}$ | Margin: {self.margin_balance['margin']}$")
            if self.position:
                self.position.info()
                Utils.printline(text='', title=False, line_char=":", size=60, blank=True)
            else:
                Utils.printline(text='No opened positions', title=False, line_char=":", size=60)
        else:
            Utils.printline(text='GAME OVER', title=True, line_char="=")

    def save_wallet_history(self, path: str, filename: str):
        path = Utils.make_path(path)
        self.closed_positions_dataframe.to_csv(f'{path}/{filename}')

    def eval_strategy(self, only_win_rate: bool = False, prefix=False) -> dict:
        profits_count = self.closed_positions_dataframe[self.closed_positions_dataframe.profit > 0].shape[0]
        losses_count = self.closed_positions_dataframe[self.closed_positions_dataframe.profit <= 0].shape[0]
        accuracy = round((profits_count / (profits_count + losses_count)) * 100, 3)

        profit_df = self.closed_positions_dataframe[self.closed_positions_dataframe.profit > 0].profit
        profit_df_length = 1 if len(profit_df) == 0 else len(profit_df)
        mean_profit_value = profit_df.sum() / profit_df_length

        loss_df = self.closed_positions_dataframe[self.closed_positions_dataframe.profit <= 0].profit
        loss_df_length = 1 if len(loss_df) == 0 else len(loss_df)
        mean_loss_value = loss_df.sum() / loss_df_length

        if prefix is False:
            if only_win_rate:
                return {"Win_rate": round(accuracy, 3)}
            else:
                return {
                    "Total_wallet_value": round(self.total_balance(), 3),
                    "Trades_counter": self.total_trades,
                    "Win_rate": round(accuracy, 3),
                    "Mean_profit": round(mean_profit_value, 3),
                    "Mean_loss": round(mean_loss_value, 3),
                    "Combo_score": round(0.01 * accuracy * self.total_balance(), 4)
                }
        else:
            if only_win_rate:
                return {f"{prefix} Win_rate": round(accuracy, 3)}
            else:
                return {
                    f"{prefix}_total_wallet_value": round(self.total_balance(), 3),
                    f"{prefix}_trades_counter": self.total_trades,
                    f"{prefix}_win_rate": round(accuracy, 3),
                    f"{prefix}_mean_profit": round(mean_profit_value, 3),
                    f"{prefix}_mean_loss": round(mean_loss_value, 3),
                    f"{prefix}_combo_score": round(0.01 * accuracy * self.total_balance(), 4)
                }

    def reset(self, ohlct: OHLCT):
        self.position = None
        self.game_over = False
        self.margin_balance = {"free": self.initial_balance, "margin": 0}
        self.transaction_reward: TransactionReward = TransactionReward()
        self.intermediate_reward: IntermediateReward = IntermediateReward(position=self.position, scaling_factor=0)
        self.closed_positions_dataframe = pd.DataFrame({"open_time": [],
                                                        "close_time": [],
                                                        "volume": [],
                                                        "open_price": [],
                                                        "close_price": [],
                                                        "stop_loss": [],
                                                        "stop_profit": [],
                                                        "contract_value": [],
                                                        "profit": [],
                                                        "margin": [],
                                                        "is_closed": [],
                                                        "is_stop_loss": [],
                                                        "is_profit": []}, index=[])
        self.update_wallet(ohlct=ohlct)

    def cancel_position(self):
        self.margin_balance['free'] += self.position.margin
        self.position = None

    def _prepare_to_open_position(self, cancel_position_type):
        if self.position is not None and self.position.type == cancel_position_type:
            return self.position_close()
        return TransactionReward()

    def __update_dataframe(self, position: Position):
        self.closed_positions.append(position)
        self.closed_positions_dataframe.loc[-1] = {"open_time": position.open_time,
                                                   "close_time": position.close_time,
                                                   "volume": position.volume,
                                                   "open_price": position.open_price,
                                                   "close_price": position.close_price,
                                                   "stop_loss": position.stop_loss,
                                                   "stop_profit": position.stop_profit,
                                                   "contract_value": position.contract_value,
                                                   "profit": position.profit,
                                                   "margin": position.margin,
                                                   "is_closed": position.is_closed,
                                                   "is_stop_loss": position.is_stop_loss,
                                                   "is_profit": position.is_stop_profit}

    def __repr__(self):
        return f"<Wallet: ticker={self.ticker}, initial_balance={self.initial_balance}, margin_balance={self.margin_balance}>"
