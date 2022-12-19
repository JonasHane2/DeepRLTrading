import pandas as pd
import numpy as np


class Bars():
    def __init__(self, bar_type="tick", avg_bars_per_day=100) -> None:
        self.avg_bars_per_day = avg_bars_per_day
        self.theta = 0
        self.bar_types = ["tick", "volume", "dollar"]
        self.bar_type = bar_type
        if self.bar_type not in self.bar_types:
            raise ValueError("Invalid imbalance type %s. Expected one of: %s" % (bar_type, self.bar_types))

    def get_threshold(self, trades: pd.DataFrame, avg_bars_per_day=100) -> float:
        """
        Returns an estimate for threshold to get a target average number of samples per day.

        Args: 
            trade (series): information about a tick
            avg_bars_per_day (integer): the target number of samples per day. 
        Returns: 
            (float): the threshold to achieve the desired bar sampling frequency. 
        """
        total_num_trades = len(trades)
        total_num_of_days = (trades.index[-1]-trades.index[0]).days
        avg_vol_trades = np.average(trades['Volume'])
        avg_price_trades = np.average(trades['Price'])
        if self.bar_type == "tick":
            return total_num_trades / ((avg_bars_per_day * total_num_of_days) + float(np.finfo(np.float32).eps))
        elif self.bar_type == "volume":
            return total_num_trades * avg_vol_trades / ((avg_bars_per_day * total_num_of_days) + float(np.finfo(np.float32).eps))
        else:
            return total_num_trades * avg_vol_trades * avg_price_trades / ((avg_bars_per_day * total_num_of_days) + float(np.finfo(np.float32).eps))

    def get_inc(self, trade: pd.Series) -> float:
        """
        Args: 
            trade (pd.Series): information about a single tick
        Returns: 
            (float): the multiplication factor depending on what bar type we use.
        """
        if self.bar_type == "volume":
            return trade['Volume']
        elif self.bar_type == "dollar":
            return trade['Volume']*trade['Price'] 
        return 1.0

    def get_all_bar_ids(self, trades: pd.DataFrame) -> list:
        """
        Finds the index of all trades that pushes the theta over the selected threshold.

        Args: 
            trades (DataFrame): list of all trades
        Returns: 
            idx (list): indices of when the threshold is reached
        """
        threshold = self.get_threshold(trades, self.avg_bars_per_day)
        idx =[]
        for i, row in trades.iterrows():
            self.theta += self.get_inc(row)
            if self.theta >= threshold:
                idx.append(i)
                self.theta = 0
        return idx


class ImbalanceBars(Bars):
    def __init__(self, bar_type="tick", avg_bars_per_day=100) -> None:
        super().__init__(bar_type, avg_bars_per_day)
        self.avg_bars_per_day = avg_bars_per_day
        self.threshold = 0
        self.b = [0] 
        self.prev_price = 0
        self.days_between_samples = []

    def tick_rule(self, curr_price: float) -> float:
        """
        Returns the sign of the change in closing price from time i-1 to i. 
        If there is no change, it returns the previous sign. Since b[0] is 
        initialized to 0 this value always exists. 
        Args: 
            prev_price (number): previous price
            cur_price (number): current price
        Returns: 
            (float): sign of price change (+1./-1.), returns previous sign if no 
                    change
        """
        delta = curr_price - self.prev_price 
        return np.sign(delta) if delta != 0 else self.b[-1]

    def register_new_tick(self, trade: pd.Series) -> bool:
        """
        Registers a new tick by updating all relevant variables and checks 
        if threshold is broken. If it is then the new threshold is calculated
        and the theta is reset. 
        Args: 
            trade (series): information about a tick
        Retruns:
            (bool): True if threshold is broken, False otherwise
        """
        price = trade['Price']
        self.b.append(self.tick_rule(price))
        imbalance = self.b[-1] * self.get_inc(trade)
        self.theta += imbalance 
        self.prev_price = price
        if abs(self.theta) >= self.threshold:
            self.theta = 0
            return True 
        return False 

    def get_imbalance_threshold_estimates(self, trades: pd.DataFrame, avg_bars_per_day=100) -> float:
        """
        (bad) estimate for what the threshold should be to generate a bar 
        every n ticks
        """
        return np.log(self.get_threshold(trades, avg_bars_per_day))

    def update_threshold(self, idx: list) -> None:
        """ threshold = threshold / (avg(historical_sample_freq)*target_samples_per_day) """
        avg_lookback_window = 60
        self.days_between_samples.append((idx[-1]-idx[-2]).days)
        if len(self.days_between_samples) >= avg_lookback_window:
            avg_sample_freq = max(np.average(self.days_between_samples[-avg_lookback_window:]), 1/avg_lookback_window)
            diff = avg_sample_freq * self.avg_bars_per_day 
            self.threshold /= (diff + float(np.finfo(np.float32).eps))

    def get_all_imbalance_ids(self, trades: pd.DataFrame) -> list:
        """
        Returns a list of all the times when the imalance threshold
        is broken in the list of trades. 
        There is a possibility that the first tick is added twice, 
        so duplicates are removed from the list
        Args: 
            trades (DataFrame): list of all trades
        Returns: 
            (list): indexes of when the threshold is reached
        """
        self.threshold = self.get_imbalance_threshold_estimates(trades, self.avg_bars_per_day)
        idx = [trades.index[0]]
        for i, row in trades.iterrows():
            if self.register_new_tick(row):
                idx.append(i)
                self.update_threshold(idx)
        return list(dict.fromkeys(idx))
