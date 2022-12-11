import numpy as np


#TODO how to calculate transaction costs with normalized prices
def asset_return(position_new, position_old, price_new, price_old, transaction_fraction=0.002) -> float:
    """ R_t = A_{t-1} * (p_t - p_{t-1}) - p_{t-1} * c * |A_{t-1} - A_{t-2}| """ 
    return position_new * (price_new - price_old) - (price_old * transaction_fraction * abs(position_new - position_old))


class PortfolioEnvironment():
    def __init__(self, states, num_instruments=1, transaction_fraction=0.002, num_prev_observations=10) -> None:
        self.transaction_fraction = float(transaction_fraction)
        num_prev_observations = int(num_prev_observations)
        if num_prev_observations < 1:
            raise ValueError("Argument num_prev_observations must be integer >= 1, and not {}".format(num_prev_observations))
        
        self.num_instruments = num_instruments
        self.states = states
        self.current_index = 0 
        self.position = np.zeros((num_instruments,))
        self.observations = np.array([self.newest_observation() for _ in range(num_prev_observations)]) # State vector of previous n observations

    def reset(self) -> np.ndarray:
        """ Reset environment to start and return initial state """
        self.current_index = 0 
        self.position.fill(0)
        self.observations = np.array([self.newest_observation() for _ in range(len(self.observations))]) 
        return self.state()

    def state(self) -> np.ndarray:
        """ Returns the state vector in a flattened format """
        return self.observations.flatten()

    def newest_observation(self) -> np.ndarray:
        """
        Returns:
            state + current position
        """
        return np.insert(self.states[self.current_index], len(self.states[self.current_index]), self.position)

    def reward_function(self, action) -> float: 
        """ Calculate return of all assets from t-1 to t, and then return sharpe.
            If only one insturment, just return the return """
        a_return = asset_return(action, self.position, self.states[self.current_index][:self.num_instruments], self.states[self.current_index-1][:self.num_instruments])
        if self.num_instruments  == 1: 
            return np.mean(a_return) 
        return np.mean(a_return)/(np.std(a_return) + float(np.finfo(np.float32).eps))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            action
        Returns:
            new state
            reward
            termination status
            info
        """
        self.current_index += 1 
        assert all(action <= 1.) and all(action >= -1.)
        if self.current_index >= len(self.states): 
            return self.state(), 0, True, {} # The end has been reached
        reward = self.reward_function(action)
        self.position = action
        self.observations = np.concatenate((self.observations[1:], [self.newest_observation()]), axis=0) # FIFO state vector update
        return self.state(), reward, False, {}
