import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_coordinates(positions: np.ndarray) -> np.ndarray:
    return np.array((np.sin(positions), np.cos(positions)))


def get_year_time_cycle_coordinates(timestamps) -> np.ndarray:
    year_pos = np.array([2*np.pi * (((stamp.dayofyear+1)/366) if stamp.is_leap_year else ((stamp.dayofyear+1)/365)) for stamp in timestamps])
    return get_coordinates(year_pos)


def get_week_time_cycle_coordinates(timestamps) -> np.ndarray:
    week_pos = np.array([2*np.pi * ((stamp.dayofweek+1)/7) for stamp in timestamps])
    return get_coordinates(week_pos)


def get_day_time_cycle_coordinates(timestamps) -> np.ndarray:
    day_pos = np.array([2*np.pi * ((stamp.hour+1)/24) for stamp in timestamps])
    return get_coordinates(day_pos)


def get_price_dataframe_2(df: pd.DataFrame, freq='1H') -> pd.Series:
    """ Returns the quoted price at some freq, forward fills missing values 
        it only gets the last observed price before that freq period,
        otherwise the agent would get information that is not yet available """
    price = df['Price'].resample(freq).last() # maybe mean()
    price = price.fillna(method='ffill')
    price.index = price.index.shift(1) # not sure if this is a good idea
    return price


def get_log_return_dataframe(price: pd.Series, period=1) -> pd.DataFrame:
    """ Returns the log returns at a specified period. 
        TODO doesn't work when the price is zero, which has happened. """
    returns = price.pct_change(period)
    returns = returns.fillna(0)
    returns = returns+1
    returns = np.log(returns) 
    return returns


def moving_avg_conv_div(price: pd.Series) -> np.ndarray:
    """ Standard MACD(12,26,9) normalized to [-1,1] """
    exp1 = price.ewm(span=12, adjust=False).mean()
    exp2 = price.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    h = macd_line - signal_line
    norm_h = (normalize(h.values) * 2) - 1
    return norm_h


def get_relative_strength_index(price: pd.Series, periods=14) -> pd.Series:
    """ Relative strength index from [0,100] to [-1, 1] """
    change = price.diff().fillna(0)
    gain = change.clip(lower=0)
    loss = -1 * change.clip(upper=0)
    ma_gain = gain.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_loss = loss.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    rs = ma_gain / ma_loss
    rsi = 100 - (100/(1+rs))
    rsi = rsi.fillna(50)
    rsi = ((rsi/100) * 2) - 1
    return rsi


def normalize(series: np.ndarray) -> np.ndarray: 
    """ Return the normalized  series """
    series = series.reshape((len(series), 1))
    scaler = MinMaxScaler().fit(series)
    series = scaler.transform(series)
    series = series.flatten()
    return series


def get_volume_dataframe(df: pd.DataFrame, index: list, freq="1H") -> pd.Series:
    """ Returns the total trade volume for some instrument at some frequency """
    volume = df['Volume'].resample(freq).sum()
    volume.index = volume.index.shift(1)
    volume = volume.loc[volume.index.intersection(index)] 
    return volume


def get_instrument_features(price: pd.Series, volume: pd.Series, returns_lag=[1,3,12]): 
    """ 
    Args: 
        prices (pd.Series): the price series for one instrument.
        volume (pd.Series): the volume series for one instruments, indexed idetically to the first argument.
        returns_lag (list): a list of integer values specifying the lag of the returns interval.
    Returns:
        norm_price (np.ndarray): an array of the normalized price series.
        norm_volume (np.ndarray): an array of the normalized volume series. 
        *log_returns (np.ndarray): arrays of log returns at the specified intervals given in the arguments.
                                   the number of arrays depends on the number of intervals provided. 
        rsi (np.ndarray): an array of the relative strength index for this price series. 
        macd (np.ndarray): an array of the moving average convergenece divergence for this price series. 
    """
    norm_price = normalize(price.values)
    norm_volume = normalize(volume.values) 
    log_returns = [get_log_return_dataframe(price, lag).values for lag in returns_lag] 
    rsi = get_relative_strength_index(price).values 
    macd = moving_avg_conv_div(price) 
    return norm_price, norm_volume, *log_returns, rsi, macd


def get_state_features(prices: pd.DataFrame, volumes: pd.DataFrame, returns_lag=[1,3,12]) -> np.ndarray:
    """ 
    Args: 
        prices (pd.DataFrame): the price series for one or more instruments. 
        volume (pd.DataFrame): the volume series for one or more instruments, indexed idetically to the first argument.
        returns_lag (list): a list of integer values specifying the lag of the returns interval.
    Returns:
        state (np.ndarray): the state features represented in an array that for n instruments and m observations 
                            has the shape n * num features X m. The features are, in order, the normalized prices, 
                            normalized volumes, normalized log returns at specified intervals, relative strength 
                            index and moving average convergence divergence. I.e., the first column is the normalized 
                            price series for the first instrument, the second column is the normalized price series 
                            for the second instrument, and so on. The time cycle coordinates at the end are just added
                            once for all instruments as they have the same datetime index. 
    """
    returns = [get_instrument_features(p[1], v[1], returns_lag=returns_lag) for p, v in zip(prices.iteritems(), volumes.iteritems())]
    state = np.array(returns, dtype=np.float64)
    state = state.T.reshape(state.shape[2], state.shape[0]*state.shape[1])
    year = get_year_time_cycle_coordinates(prices.index)
    week = get_week_time_cycle_coordinates(prices.index)
    day = get_day_time_cycle_coordinates(prices.index)
    state = np.concatenate((state, year.T, week.T, day.T), axis=1)
    return state


def get_state_array_2(dfs: list, labels: list, freq="1H", returns_lag=[1,3,12], riskfree_asset=False) -> np.ndarray: 
    """     
    Args:
        dfs (list): a list of dataframes that consists of trade information of different instruments.
                    Each dataframe must have a column 'DateTime', 'Price', and 'Volume'. 
        labels (list): a list of labels for each instrument. 
        freq (str): a string specifying the frequency at which to resample the observations.
        riskfree_asset (bool): if true, a riskfree asset is added as the first asset in the state array.
    Returns:
        state (np.ndarray): the state vector designed to be compatible with a deep RL agent. 
                            See get_state_features() for a more in-depth explanation. 
        non_norm_prices (pd.DataFrame): a dataframe consisting of the non normalized prices of every
                                        instrument in the state array (including the riskfree asset if used). 
    """
    dfs = [d.set_index('DateTime') for d in dfs]
    prices = [get_price_dataframe_2(d, freq) for d in dfs]
    indices = [d.index for d in prices]
    index_intersection = list(set(indices[0]).intersection(*indices))
    index_intersection.sort() 
    prices_intersection = [p.loc[p.index.intersection(index_intersection)] for p in prices]
    volumes = [get_volume_dataframe(d, index_intersection, freq) for d in dfs]

    if riskfree_asset:
        labels.insert(0, 'cash')
        riskfree_asset_price = prices_intersection[0].copy()
        riskfree_asset_price.values[:] = 1
        prices_intersection.insert(0, riskfree_asset_price)
        riskfree_asset_volume = volumes[0].copy()
        riskfree_asset_volume.values[:] = 0 
        volumes.insert(0, riskfree_asset_volume)

    non_norm_prices = pd.DataFrame(list(map(list, zip(*prices_intersection))), index=index_intersection).set_axis(labels, axis=1, inplace=False)
    non_norm_prices.index = non_norm_prices.index.tz_localize('utc')
    non_norm_volumes = pd.DataFrame(list(map(list, zip(*volumes))), index=index_intersection).set_axis(labels, axis=1, inplace=False)
    state = get_state_features(non_norm_prices, non_norm_volumes, returns_lag)
    return state, non_norm_prices
