import numpy as np
#from transaction_cost_model import transaction_cost_model


class PortfolioEnvironment():

    def __init__(self, states, prices, num_instruments=1, transaction_fraction=0.002, num_prev_observations=10, flatten_prev_observations=True, transpose_prev_obs=True, transaction_cost=False, add_prev_position=False, std_lookback=60, std_factor=0, downside_deviation=False) -> None:
        self.prices = prices
        self.transaction_fraction = max(float(transaction_fraction), 0.0)
        self.transaction_cost = transaction_cost        
        self.add_prev_position = add_prev_position
        self.flatten_prev_observations = flatten_prev_observations
        self.transpose_prev_obs = transpose_prev_obs
        self.num_instruments = num_instruments
        self.states = states
        self.current_index = 0
        self.position = np.zeros((num_instruments,))
        num_prev_observations = max(int(num_prev_observations), 1)
        self.returns = np.zeros((std_lookback,))
        self.std_factor = std_factor
        self.downside_deviation = downside_deviation
        self.observations = np.array([self.newest_observation() for _ in range(num_prev_observations)]) 

    def reset(self) -> np.ndarray:
        """ Reset environment to start and return initial state """
        self.current_index = 0 
        self.position.fill(0)
        self.observations = np.array([self.newest_observation() for _ in range(len(self.observations))]) 
        return self.state()

    def state(self) -> np.ndarray:
        """ Returns the state vector """
        if self.flatten_prev_observations:
            return self.observations.flatten()
        else: 
            if self.transpose_prev_obs:
                return self.observations.T
            else: 
                return self.observations

    def newest_observation(self) -> np.ndarray:
        """ Returns the current position inserted into the current state array, if specified. 
            Else returns the current state array. """
        if self.add_prev_position: 
            return np.insert(self.states[self.current_index], len(self.states[self.current_index]), self.position)
        else:
            return self.states[self.current_index]

    def reward_function(self, action) -> float: 
        """ Returns reward signal based on the environments chosen reward function. """
        ret = asset_return(position_new=action, 
                        position_old=self.position, 
                        #price_new=self.states[self.current_index][:self.num_instruments], 
                        #price_old=self.states[self.current_index-1][:self.num_instruments], 
                        price_new=self.prices[self.current_index],
                        price_old=self.prices[self.current_index-1],
                        transaction_fraction=self.transaction_fraction,
                        transaction_cost=self.transaction_cost)
        
        # Add returns to array of old returns
        self.returns = np.roll(self.returns,-1)
        self.returns[-1] = np.sum(ret)

        if self.downside_deviation: 
            risk_punishment = np.nan_to_num(np.std(self.returns.clip(max=0)))
        else: 
            risk_punishment = np.std(self.returns)

        return np.sum(ret) - (self.std_factor * risk_punishment)

    def step(self, action):
        """
        Args:
            action (np.ndarray):        the actors action. 
        Returns:
            new state (np.ndarray):     the new observed state. 
            reward (float):             reward signal. 
            termination status (bool):  flag indicating if the state is terminal. 
            info:                       -
        """
        self.current_index += 1 
        assert all(action <= 1.) and all(action >= -1.)
        if self.current_index >= len(self.states): 
            return self.state(), 0, True, {} 
        reward = self.reward_function(action)
        self.position = action
        self.observations = np.concatenate((self.observations[1:], [self.newest_observation()]), axis=0) # FIFO state vector update
        return self.state(), reward, False, {}


def asset_return(position_new, position_old, price_new, price_old, transaction_fraction=0.0002, transaction_cost=True) -> np.ndarray:
    """ R_t = a_{t-1} * log(p_t / p_{t-1}) - transaction costs """
    rtn = position_new * np.log((price_new + float(np.finfo(np.float32).eps))/(price_old + float(np.finfo(np.float32).eps)))
    if transaction_cost: 
        #rtn -= transaction_fraction * np.abs(position_new - position_old)
        #rtn -= transaction_cost_model(position_new, position_old, price_new, price_old, transaction_fraction)[0]
        rtn -= transaction_costs(position_new, position_old, price_new, price_old, transaction_fraction)
    return rtn

def transaction_costs(position_new, position_old, price_new, price_old, transaction_fraction=0.0002):
    """ Transaction costs for single instrument trading """
    price_changes = ((price_new + float(np.finfo(np.float32).eps)) /(price_old + float(np.finfo(np.float32).eps))) 
    weight_as_result_of_price_change = (position_old * price_changes) / ((position_old * (price_changes-1)) + 1)
    required_rebalance = np.abs(position_new - weight_as_result_of_price_change)
    t_costs = transaction_fraction * required_rebalance
    return t_costs
