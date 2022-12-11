import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_price_dataframe_1(df: pd.DataFrame) -> pd.Series:
    price = df['Price']
    price = price.dropna()
    return price


def get_price_dataframe_2(df: pd.DataFrame, freq='1min') -> pd.Series:
    """ Returns the quoted price at some freq, 
        forward fills missing values 
        it only gets the last observed price before that freq period,
        otherwise the agent would get information that is not yet available """
    price = df['Price'].resample(freq).last() 
    price = price.fillna(method='ffill')
    price.index = price.index.shift(1)
    return price


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


# This doesn't work when the price can be zero or negative
"""
def get_log_return_dataframe(price: pd.Series, period=1) -> np.ndarray:
    returns = price.pct_change(period)
    returns = returns.fillna(0)
    returns = returns+1
    returns = np.log(returns) 
    returns = returns.values
    return returns
def get_log_returns(price: np.ndarray, lag: int) -> np.ndarray:
    if lag >= len(price) or lag < 1:
        return np.zeros(price.shape)
    returns = price[lag:]/price[:-lag]
    returns = np.log(returns)
    returns = np.insert(returns, [0], np.zeros(lag), axis=0)
    return returns
"""


def get_rolling_return(price: pd.Series, period="30D") -> np.ndarray:
    """ Returns the rolling return normalized to [-1, 1] """
    returns = price.rolling(period).mean()
    returns = (normalize_prices(returns.values) * 2) - 1
    return returns


def moving_avg_conv_div(price: pd.Series) -> np.ndarray:
    """ Standard MACD(12,26,9) normalized to [-1,1] """
    exp1 = price.ewm(span=12, adjust=False).mean()
    exp2 = price.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    h = macd_line - signal_line
    norm_h = (normalize_prices(h.values) * 2) - 1
    return norm_h


def get_relative_strength_index(price: pd.Series, periods=14) -> np.ndarray:
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


def normalize_prices(price: np.ndarray) -> np.ndarray: 
    """ Return the normalized price series """
    price = price.reshape((len(price), 1))
    scaler = MinMaxScaler().fit(price)
    price = scaler.transform(price)
    price = price.flatten()
    return price


def get_instrument_info(df: pd.DataFrame):# -> tuple[np.ndarray, pd.Series]:
    non_norm_price = get_price_dataframe_1(df)
    lag = ["1D", "7D", "30D"]
    returns = [get_rolling_return(non_norm_price, l) for l in lag]

    state = np.concatenate((np.array([
        normalize_prices(non_norm_price.values),
        moving_avg_conv_div(non_norm_price),
        get_relative_strength_index(non_norm_price)
    ]), returns))

    return state, non_norm_price


def get_state_array(df: pd.DataFrame, riskfree_asset=False) -> np.ndarray: 
    """ Returns a state array that can be used by the RL agent """
    # TODO make compatible with multiple instruments
    df = df.set_index('DateTime').sort_values(by='DateTime')
    timestamps = df.index 
    inst_state, non_norm_price = get_instrument_info(df)
    year = get_year_time_cycle_coordinates(timestamps)
    week = get_week_time_cycle_coordinates(timestamps)
    day = get_day_time_cycle_coordinates(timestamps)

    if riskfree_asset:
        rf_df = df.copy()
        rf_df['Price'].values[:] = 0
        rf_info, _ = get_instrument_info(rf_df)
        inst_state = np.concatenate((rf_info, inst_state))

    states = inst_state
    for arr in [year, week, day]:
        states = np.concatenate((states, arr))
    states = states.T
    return states, non_norm_price
