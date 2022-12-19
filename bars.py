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
        """ Returns the sign of the price change, or the previous sign if the price is unchanged. """
        delta = curr_price - self.prev_price 
        return np.sign(delta) if delta != 0 else self.b[-1]

    def register_new_tick(self, trade: pd.Series, prev_observation_date: pd.Timestamp) -> bool:
        """
        Registers a new tick by updating all relevant variables and checks if threshold 
        is broken. If it is then the theta is reset and True is returned. If the time 
        since a tick was last registered exceeds the target average the threshold is lowered. 

        Args: 
            trade (pd.Series): a single trade. 
        Retruns:
            (bool): True if threshold is broken, False otherwise.
        """
        price = trade['Price']
        self.b.append(self.tick_rule(price))
        imbalance = self.b[-1] * self.get_inc(trade)
        self.theta += imbalance 
        self.prev_price = price
        if abs(self.theta) >= self.threshold:
            self.theta = 0
            return True 
        current_num_days = (trade.name - prev_observation_date).days
        if current_num_days > (1/self.avg_bars_per_day):
            self.threshold *= 0.9
        return False 

    def get_imbalance_threshold_estimates(self, trades: pd.DataFrame, avg_bars_per_day=100) -> float:
        """ Estimate for what the threshold should be to generate 'avg_bars_per_day' imbalance bars per day. """
        return np.log(self.get_threshold(trades, avg_bars_per_day))

    def update_threshold(self, idx: list) -> None:
        """ Adjusts the threshold by comparing the previous sample frequency to the target. """
        avg_lookback_window = 60
        self.days_between_samples.append((idx[-1]-idx[-2]).days)
        if len(self.days_between_samples) >= avg_lookback_window:
            avg_sample_freq = max(np.average(self.days_between_samples[-avg_lookback_window:]), 1/avg_lookback_window)
            diff = avg_sample_freq * self.avg_bars_per_day 
            self.threshold /= (diff + float(np.finfo(np.float32).eps))

    def get_all_imbalance_ids(self, trades: pd.DataFrame) -> list:
        """
        Args: 
            trades (DataFrame): list of all trades indexed by datetime and sorted. 
        Returns: 
            (list): datetimes of when the threshold is reached. 
        """
        self.threshold = self.get_imbalance_threshold_estimates(trades, self.avg_bars_per_day)
        idx = [trades.index[0]]
        for i, row in trades.iterrows():
            if self.register_new_tick(row, idx[-1]):
                idx.append(i)
                self.update_threshold(idx)
        return list(dict.fromkeys(idx))
