import numpy as np
import pandas as pd
from scipy.special import softmax


def long_only_portfolio(prices: pd.DataFrame, scaled=False) -> pd.DataFrame: 
    """ Returns a long only weights dataframe """
    if scaled: 
        return pd.DataFrame(softmax(np.ones((prices.shape[0]-1, prices.shape[1])), axis=1), index=prices.index[:-1]).set_axis(list(prices.columns), axis=1, inplace=False)
    return pd.DataFrame(np.ones((prices.shape[0]-1, prices.shape[1])), index=prices.index[:-1]).set_axis(list(prices.columns), axis=1, inplace=False)


def percentage_returns(prices: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """ Returns the % returns for every intsrument in the portfolio.
        The number of prices should always be one greater than the number of positions.
        It is assumed the portfolio always start with position 0. 
        The % change at price zero is also set to 0, that way the portfolios initial return is zero. """
    # TODO this assumes that prices are always > 0, which is not the case  
    price_changes = prices.pct_change(1)
    pos = np.insert(positions.values, 0, 0, axis=0) #insert 0 as the initial position of the portfolio
    portfolio_returns = price_changes * pos
    return portfolio_returns


def create_pyfolio_compatible_returns_history(prices: pd.DataFrame, positions: pd.DataFrame, transaction_fraction=0.00002) -> pd.Series:
    """ Takes the prices of all instruments and their corresponding positions as argument.
        Calculates their noncumulative returns individually and takes the sum. 
        Subtracts transaction costs. Averages over daily returns. The returned series is compatible with pyfolio. """
    # Gross % returns in one series
    portfolio_returns = percentage_returns(prices, positions)
    portfolio_returns = portfolio_returns.sum(axis=1)
    portfolio_returns = portfolio_returns.fillna(0)
    portfolio_returns = portfolio_returns.divide(100)

    # Transaction costs as a % 
    tc = positions.diff(1).abs().sum(axis=1).multiply(transaction_fraction)
    tc.iloc[0] = sum(abs(positions.iloc[0])) * transaction_fraction
    tc = tc.divide(100)

    # Net % returns
    portfolio_returns = portfolio_returns.subtract(np.append(tc.values, 0))
    portfolio_returns = portfolio_returns.resample("1D").sum().fillna(0)
    if portfolio_returns.index.tzinfo is None or portfolio_returns.index.tzinfo.utcoffset(portfolio_returns.index) is None:
        portfolio_returns.index = portfolio_returns.index.tz_localize('utc')
    return portfolio_returns


def create_pyfolio_compatible_positions_history(prices: pd.DataFrame, positions: pd.DataFrame, labels: list, transaction_fraction=0.00002, init_balance=1000) -> pd.DataFrame:
    """ Dollar amount invested in each position. Non-working capital labelled cash. """
    portfolio_returns = create_pyfolio_compatible_returns_history(prices, positions, transaction_fraction)
    cumul_ret = portfolio_returns.add(1).cumprod()
    total_balance_history = cumul_ret.multiply(init_balance)    
    positions_hist = positions.resample("1D").last().fillna(method='ffill')
    positions_hist = positions_hist.multiply(total_balance_history.values, axis=0)
    if positions_hist.index.tzinfo is None or positions_hist.index.tzinfo.utcoffset(positions_hist.index) is None:
        positions_hist.index = positions_hist.index.tz_localize('utc')
    return positions_hist.set_axis(labels, axis=1)


def positions_numpy_to_dataframe(positions: np.ndarray, prices: pd.DataFrame, labels: list) -> pd.DataFrame:
    """ Converts our output positions numpy array to dataframe compatible with pyfolio """
    print(prices.shape, positions.shape)
    return pd.DataFrame(positions, index=prices.index[:-1]).set_axis(labels, axis=1, inplace=False)


def get_baseline_returns_history(baseline_strat, prices: pd.DataFrame, scaled=False) -> pd.DataFrame:
    """ Gets the returns for a baseline strategy that is compatible with pyfolio """
    baseline_long_only_portfolio = baseline_strat(prices, scaled=scaled) 
    baseline_returns_history = create_pyfolio_compatible_returns_history(prices, baseline_long_only_portfolio)
    return baseline_returns_history


def get_pyfolio_history(prices: pd.DataFrame, positions: np.ndarray): 
    """ Returns returns and position history for pyfolio """
    labels = list(prices.columns)
    positions = positions_numpy_to_dataframe(positions, prices, labels)
    returns_history = create_pyfolio_compatible_returns_history(prices, positions)
    #position_history = create_pyfolio_compatible_positions_history(prices, positions, labels) # not really needed, maybe remove
    return returns_history#, position_history
