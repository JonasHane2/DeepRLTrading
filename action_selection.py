import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_action_pobs(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False):
    """ Returns action probabilities and hidden state (if recurrent) from network """
    if recurrent: 
        probs, hx = net.forward(state, prev_action, hx)
    else: 
        probs = net.forward(state, prev_action)
    return probs.to(device), hx


def add_noise(action: torch.Tensor, epsilon=0, training=True) -> torch.Tensor:
    """ Adds random noise to the actions """
    noise = ((np.random.rand((action.shape[1])) * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    noise *= training * max(epsilon,0) 
    action = action.add(torch.FloatTensor(noise).to(device)).to(device)
    return action


def action_transform(action: torch.Tensor) -> torch.Tensor:
    """ Returns the tanh of the action """
    action = torch.tanh(action).to(device)
    return action


def action_softmax_transform(action: torch.Tensor) -> torch.Tensor:
    """ Returns the softmax of the action_transform """
    action = action_transform(action)
    action = F.softmax(action, dim=1).to(device)
    return action


# All functions are compatible with both recurrent and non-recurrent nets
# ------------------ One instrument 
## ----------------- Stochastic Sampling
### ---------------- Discrete Action Space (Multinoulli Dist)
def act_stochastic_discrete(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0):
    """ Returns a sampled action on {-1, 0, 1}, the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    probs = F.softmax(probs, dim=1)
    m = Categorical(probs) 
    action = m.sample() 
    return action.add(-1).cpu().numpy(), m.log_prob(action).to(device), hx


### ---------------- Continuous Action Space (Gaussian Dist)
def act_stochastic_continuous_2(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0):
    """ Returns a sampled action on [-1, 1], the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    mean, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    mean = torch.tanh(mean)
    std = max(epsilon, 1e-8)
    dist = Normal(mean, std) 
    action = dist.sample() 
    return torch.clamp(action, -1, 1).cpu().numpy().flatten(), dist.log_prob(action).to(device), hx


## ----------------- Deterministic Sampling 
### ---------------- Discrete Action Space (DQN)
def act_DQN(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0):
    """ Returns the highest value action of a discrete set of actions {-1, 0, 1}, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():    
        action_vals, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    if random.random() < epsilon:
        action = np.array([np.random.randint(-1, 2)]) #random int on interval [-1, 2)
    else:
        action = action_vals.argmax().add(-1).cpu().numpy().flatten()
    return action, hx


# ------------------ Portfolio
## ----------------- Stochastic Sampling
### ---------------- Long & Short 
def act_stochastic_portfolio(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0):
    """ Returns a sampled action the interval [-1, 1] for N instruments, the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    cov_matrix = max(epsilon, 1e-8)*torch.eye(probs.size(1))
    m = MultivariateNormal(probs.to(device), cov_matrix.to(device))
    action = m.sample().to(device)
    logprob = m.log_prob(action)
    action = action_transform(action)
    action = action.cpu().numpy().flatten()
    return action, logprob.to(device), hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_stochastic_portfolio_long(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0):
    """ Returns a sampled action the interval [0, 1] for N instruments where all weights sum to 1, 
        the log probability of choosing that action, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, hx = get_action_pobs(net, state, prev_action, hx, recurrent) 
    cov_matrix = max(epsilon, 1e-8)*torch.eye(probs.size(1))
    m = MultivariateNormal(probs.to(device), cov_matrix.to(device))
    action = m.sample().to(device)
    logprob = m.log_prob(action)
    action = F.softmax(action, dim=1)
    action = action.cpu().numpy().flatten()
    return action, logprob.to(device), hx


## ----------------- Deterministic Sampling (DDPG)
### ---------------- Long & Short 
# This works for both portfolio and single instrument trading
def act_DDPG_portfolio(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0, training=True):
    """ Returns the highest value action on a continous interval [-1, 1] for N instruments, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    action = add_noise(action, epsilon, training)
    action = action_transform(action)
    action = action.cpu().numpy().flatten()
    return action, hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_DDPG_portfolio_long(net: nn.Module, state: np.ndarray, prev_action=None, hx=None, recurrent=False, epsilon=0, training=True):
    """ Returns the highest value action on a continous interval [0, 1] for N instruments where all weights sum to 1, and the hidden state """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action, hx = get_action_pobs(net, state, prev_action, hx, recurrent)
    action = add_noise(action, epsilon, training)
    action = action_softmax_transform(action)
    action = action.cpu().numpy().flatten()
    return action, hx
