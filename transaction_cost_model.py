import numpy as np


def transaction_cost_model(position_new, position_old, price_new, price_old, transaction_fraction=0.0002) -> np.ndarray:
    """ Calculates the transaction costs for a single trade, or a list of trades. """
    if len(position_new.shape) == 1:
        position_new = np.array([position_new])
    if len(position_old.shape) == 1:
        position_old = np.array([position_old])
    if len(price_new.shape) == 1:
        price_new = np.array([price_new])
    if len(price_old.shape) == 1:
        price_old = np.array([price_old])

    volatility = np.std((price_new / price_old), axis=0)
    trade_size = size_of_trade(position_new, position_old, price_new, price_old)
    c_n_f = comissions_and_fees(trade_size=trade_size, transaction_fraction=transaction_fraction)
    slip = slippage(trade_size=trade_size, volatility=volatility)
    m_imp = market_impact(trade_size=trade_size, volatility=volatility)
    return (c_n_f + slip + m_imp)


def size_of_trade(position_new, position_old, price_new, price_old) -> np.ndarray:
    """ 
    When the prices of the individual instruments in the portfolio changes, their respective weighting also changes. 
    Therefore, the agent will need to rebalance the portfolio to the desired new weights from the old (changed) weights. 
    The agent will need to pay transaction costs on these changes.
    This function calculates the percentage of changes for each individual instrument in the portfolio. 

    Args:
        position_new (list): the portfolio weights at time t.
        position_old (list): the portfolio weights at time t-1.
        price_new (list): the corresponding price for each instrument at time t.
        price_old (list): the corresponding price for each instrument at time t-1.
    Returns:
        change_as_perc_of_port_val (float): the change in portfolio from time t-1 to t for each instrument. 
    """    
    current_portolio_weights = position_old * (price_new/(price_old + float(np.finfo(np.float32).eps)))
    portfolio_val_increase = np.sum((price_new/price_old), axis=1).reshape(-1, 1) / price_new.shape[-1]
    required_rebalancing = position_new - current_portolio_weights/portfolio_val_increase
    required_rebalancing = np.nan_to_num(abs(required_rebalancing))
    return required_rebalancing


def comissions_and_fees(trade_size, transaction_fraction=0.0002) -> np.ndarray:
    """ Comissions and fees are usually linear w.r.t the trade size. """
    tc = transaction_fraction * trade_size
    tc = np.nan_to_num(tc)
    return tc


def slippage(trade_size, volatility=0.01, mean_spread=0.001) -> np.ndarray:
    """ Change in bid/ask spread from alpha modeling to execution.
        Usually quadratic w.r.t trade size """
    slp = trade_size**2 * volatility * np.random.normal(0, mean_spread, trade_size.shape)
    slp = np.nan_to_num(slp)
    return slp


def market_impact(trade_size, mean_spread=(0.001/200), alpha=1, volatility=0, daily_volume=1e10) -> np.ndarray:
    """ The square-root formula for market impact described by Grinold and Kahn. """
    m_impact = mean_spread + alpha * volatility * np.sqrt(trade_size/daily_volume)
    m_impact *= (trade_size != 0)
    m_impact = np.nan_to_num(m_impact)
    return m_impact
