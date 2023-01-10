import numpy as np
from transaction_cost_model import transaction_cost_model


class PortfolioEnvironment():

    def __init__(self, states, num_instruments=1, transaction_fraction=0.002, num_prev_observations=10, reward_function_type='return', transaction_cost=False) -> None:
        self.transaction_fraction = float(transaction_fraction)
        self.transaction_cost = transaction_cost
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
        ret = asset_return(position_new=action, 
                        position_old=self.position, 
                        price_new=self.states[self.current_index][:self.num_instruments], 
                        price_old=self.states[self.current_index-1][:self.num_instruments], 
                        transaction_cost=self.transaction_cost)
        if self.reward_function_type == 'return':
            return np.mean(ret)
        elif self.reward_function_type == 'Sharpe':
            std = np.nan_to_num(np.std(ret))
            if self.num_instruments  == 1 or std == 0: 
                return np.mean(ret) 
            return np.mean(ret)/(std + float(np.finfo(np.float32).eps))
        elif self.reward_function_type == 'Sortino':
            std = np.nan_to_num(np.std(ret[ret<0]))
            if self.num_instruments  == 1 or std == 0: 
                return np.mean(ret) 
            return np.mean(ret)/(std + float(np.finfo(np.float32).eps))            

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
    """ R_t = A_{t-1} * log(p_t / p_{t-1}) - transaction costs """
    rtn = position_new * np.log((price_new + float(np.finfo(np.float32).eps))/(price_old + float(np.finfo(np.float32).eps)))
    return rtn if not transaction_cost else (rtn - transaction_cost_model(position_new, position_old, price_new, price_old, transaction_fraction)[0])
