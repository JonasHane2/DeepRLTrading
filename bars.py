import numpy as np

class Bars():
    def __init__(self, bar_type="tick", avg_bars_per_day=100):
        self.avg_bars_per_day = avg_bars_per_day
        self.theta = 0

        self.bar_types = ["tick", "volume", "dollar"]
        self.bar_type = bar_type
        if self.bar_type not in self.bar_types:
            raise ValueError("Invalid imbalance type %s. Expected one of: %s" % (bar_type, self.bar_types))

    def get_threshold(self, trades, avg_bars_per_day=100):
        """
        Returns an estimate for threshold to get a target average number of samples per day.

        Args: 
            trade (series): information about a tick
            avg_bars_per_day (integer): the target number of samples per day. 
        Returns: 
            (number): the threshold to achieve the desired bar sampling 
                      frequency. 
        """
        total_num_trades = len(trades)
        total_num_of_days = (trades.index[-1]-trades.index[0]).days
        avg_vol_trades = np.average(trades['Volume'])
        avg_price_trades = np.average(trades['Price'])

        if self.bar_type == "tick":
            return total_num_trades / (avg_bars_per_day * total_num_of_days)
        elif self.bar_type == "volume":
            return total_num_trades * avg_vol_trades / (avg_bars_per_day * total_num_of_days)
        else:
            return total_num_trades * avg_vol_trades * avg_price_trades / (avg_bars_per_day * total_num_of_days)

    def get_inc(self, trade):
        """
        Args: 
            trade (series): information about a single tick
        Returns: 
            (number): the multiplication factor depending on what bar type we use.
        """
        if self.bar_type == "volume":
            return trade['Volume']
        elif self.bar_type == "dollar":
            return trade['Volume']*trade['Price'] 
        return 1 

    def get_all_bar_ids(self, trades):
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
