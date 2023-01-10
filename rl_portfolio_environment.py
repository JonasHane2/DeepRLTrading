import numpy as np
from transaction_cost_model import transaction_cost_model


class PortfolioEnvironment():

    def __init__(self, states, num_instruments=1, transaction_fraction=0.002, num_prev_observations=10, reward_function_type='return') -> None:
        self.transaction_fraction = float(transaction_fraction)
        num_prev_observations = int(num_prev_observations)
        reward_functions = ['return', 'Sharpe', 'Sortino']
        if num_prev_observations < 1:
            raise ValueError("Argument num_prev_observations must be integer >= 1, and not {}".format(num_prev_observations))
        if reward_function_type not in reward_functions:
            raise ValueError("Argument reward_function must be one of {}".format(reward_functions))
        self.reward_function_type = reward_function_type
        self.num_instruments = num_instruments
        self.states = states
        self.current_index = 0
        self.position = np.zeros((num_instruments,))
        self.observations = np.array([self.newest_observation() for _ in range(num_prev_observations)]) 

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
        """ Returns the current position inserted into the current state array. """
        return np.insert(self.states[self.current_index], len(self.states[self.current_index]), self.position)

    def reward_function(self, action) -> float: 
        """ Returns reward signal based on the environments chosen reward function. """
        if self.reward_function_type == 'return':
            ret = asset_return(action, self.position, self.states[self.current_index][:self.num_instruments], self.states[self.current_index-1][:self.num_instruments])
            return np.mean(ret)
        elif self.reward_function_type == 'Sharpe':
            ret = asset_return(action, self.position, self.states[self.current_index][:self.num_instruments], self.states[self.current_index-1][:self.num_instruments])
            if self.num_instruments  == 1: 
                return np.mean(ret) 
            return np.mean(ret)/(np.std(ret) + float(np.finfo(np.float32).eps))
        elif self.reward_function_type == 'Sortino':
            ret = asset_return(action, self.position, self.states[self.current_index][:self.num_instruments], self.states[self.current_index-1][:self.num_instruments])
            if self.num_instruments  == 1: 
                return np.mean(ret) 
            return np.mean(ret)/(np.std(ret[ret<0]) + float(np.finfo(np.float32).eps))            

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


def asset_return(position_new, position_old, price_new, price_old, transaction_fraction=0.0002) -> np.ndarray:
    """ R_t = A_{t-1} * log(p_t / p_{t-1}) - transaction costs """
    return position_new * np.log((price_new + float(np.finfo(np.float32).eps))/(price_old + float(np.finfo(np.float32).eps))) - transaction_cost_model(position_new, position_old, price_new, price_old, transaction_fraction)[0]
