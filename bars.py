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
        Returns an initial estimate for threshold to get a target average number of samples per day
        based on the first month of trades.
        Args: 
            trade (series): information about a tick
            avg_bars_per_day (integer): the target number of samples per day. 
        Returns: 
            (float): the threshold to achieve the desired bar sampling frequency. 
        """
        if self.bar_type == "tick":
            threshold = trades['Price'].resample('1D').count()
        elif self.bar_type == "volume":
            threshold = trades['Volume'].resample('1D').sum()
        else:
            threshold = (trades['Volume']*trades['Price']).resample('1D').sum()
        threshold = threshold.rolling('90D').mean().bfill()
        threshold /= avg_bars_per_day
        return threshold

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
        idx =[]
        threshold = self.get_threshold(trades, self.avg_bars_per_day)
        curr_threshold=0
        for i, row in trades.iterrows():
            self.theta += self.get_inc(row)
            if self.theta >= curr_threshold:
                idx.append(i)
                curr_threshold = threshold.iloc[threshold.index.get_loc(i, method='ffill')]
                self.theta = 0
        return idx


class ImbalanceBars(Bars):
    def __init__(self, bar_type="tick", avg_bars_per_day=100) -> None:
        super().__init__(bar_type, avg_bars_per_day)
        self.avg_bars_per_day = avg_bars_per_day
        self.curr_threshold = 0
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
        if abs(self.theta) >= self.curr_threshold:
            self.theta = 0
            return True 
        current_num_days = (trade.name - prev_observation_date).days
        if current_num_days > (1/self.avg_bars_per_day):
            self.curr_threshold *= 0.9
        return False 

    def get_all_imbalance_ids(self, trades: pd.DataFrame) -> list:
        """
        Args: 
            trades (DataFrame): list of all trades indexed by datetime and sorted. 
        Returns: 
            (list): datetimes of when the threshold is reached. 
        """
        threshold = self.get_threshold(trades, self.avg_bars_per_day)
        self.curr_threshold=0
        idx = [trades.index[0]]
        for i, row in trades.iterrows():
            if self.register_new_tick(row, idx[-1]):
                idx.append(i)
                self.curr_threshold = np.log(threshold.iloc[threshold.index.get_loc(i, method='ffill')])
        return list(dict.fromkeys(idx))
