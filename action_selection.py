import random
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_action_pobs(net: nn.Module, state: np.ndarray, hx=None, recurrent=False) -> tuple[torch.Tensor, tuple]:
    """ Returns action probabilities and hidden state (if recurrent) from network """
    if recurrent: 
        probs, hx = net.forward(state, hx)
    else: 
        probs = net.forward(state).cpu()
    return probs, hx


def add_noise(action: torch.Tensor, epsilon=0, training=True) -> torch.Tensor:
    """ Adds random noise to the actions """
    noise = ((np.random.rand((action.shape[1])) * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    noise = noise * training * max(epsilon,0) 
    action = action.add(torch.FloatTensor(noise))
    return action 


def action_transform(action: torch.Tensor) -> torch.Tensor:
    """ Returns the tanh of the action """
    action = torch.tanh(action)
    return action 


def action_softmax_transform(action: torch.Tensor) -> torch.Tensor:
    """ Returns the softmax of the action_transform """
    action = action_transform(action)
    action = F.softmax(action, dim=1)
    return action


# All functions are compatible with both recurrent and non-recurrent nets
# ------------------ One instrument 
## ----------------- Stochastic Sampling
### ---------------- Discrete Action Space (Multinoulli Dist)
# TODO maybe add random exploration to stochastic action selection
def act_stochastic_discrete(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, tuple]:
    """ Returns a sampled action on {-1, 0, 1}, the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, hx, recurrent)
    probs = F.softmax(probs, dim=1)
    m = Categorical(probs) 
    action = m.sample() 
    return action.add(-1).numpy(), m.log_prob(action), hx


### ---------------- Continuous Action Space (Gaussian Dist)
def act_stochastic_continuous_2(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, torch.Tensor]:
    """ Returns a sampled action on [-1, 1], the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    mean, hx = get_action_pobs(net, state, hx, recurrent)
    mean = torch.tanh(mean)
    std = max(epsilon, 1e-8)
    dist = Normal(mean, std) 
    action = dist.sample() 
    return torch.clamp(action, -1, 1).numpy().flatten(), dist.log_prob(action), hx


## ----------------- Deterministic Sampling 
### ---------------- Discrete Action Space (DQN)
def act_DQN(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[int, torch.Tensor]:
    """ Returns the highest value action of a discrete set of actions {-1, 0, 1}, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():    
        action_vals, hx = get_action_pobs(net, state, hx, recurrent)
    if random.random() < epsilon:
        action = np.array([np.random.randint(-1, 2)]) #random int on interval [-1, 2)
    else:
        action = action_vals.argmax().add(-1).numpy().flatten()
    return action, hx


# ------------------ Portfolio
## ----------------- Stochastic Sampling
### ---------------- Long & Short 
def act_stochastic_portfolio(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, tuple]:
    """ Returns a sampled action the interval [-1, 1] for N instruments, the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, hx, recurrent)
    probs = add_noise(probs, epsilon)
    m = MultivariateNormal(probs, torch.eye(probs.size(1))) # second arg is indentity matrix of size num instruments X num instruments
    action = m.sample()
    logprob = m.log_prob(action)
    action = action_transform(action)
    action = action.numpy().flatten()
    return action, logprob, hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_stochastic_portfolio_long(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, tuple]:
    """ Returns a sampled action the interval [0, 1] for N instruments where all weights sum to 1, 
        the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, hx, recurrent) 
    probs = add_noise(probs, epsilon)
    m = MultivariateNormal(probs, torch.eye(probs.size(1))) 
    action = m.sample()
    logprob = m.log_prob(action)
    action = F.softmax(action, dim=1)
    action = action.numpy().flatten()
    return action, logprob, hx


## ----------------- Deterministic Sampling (DDPG)
### ---------------- Long & Short 
# This works for both portfolio and single instrument trading
def act_DDPG_portfolio(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0, training=True) -> tuple[np.ndarray, torch.Tensor]:
    """ Returns the highest value action on a continous interval [-1, 1] for N instruments, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action, hx = get_action_pobs(net, state, hx, recurrent)
    action = add_noise(action, epsilon, training)
    action = action_transform(action)
    action = action.numpy().flatten()
    return action, hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_DDPG_portfolio_long(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0, training=True) -> tuple[np.ndarray, torch.Tensor]:
    """ Returns the highest value action on a continous interval [0, 1] for N instruments where all weights sum to 1, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action, hx = get_action_pobs(net, state, hx, recurrent)
    action = add_noise(action, epsilon, training)
    action = action_softmax_transform(action)
    action = action.numpy().flatten() 
    return action, hx
