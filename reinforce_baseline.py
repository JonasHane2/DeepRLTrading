import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) # for relative imports
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
from reinforce import optimize
#from DeepRLTrading.reinforce import reinforce
criterion = torch.nn.MSELoss()

def get_policy_and_value_loss(value_function, state_batch, reward_batch, log_probs):# -> tuple[torch.Tensor, torch.Tensor]:
    state_value = value_function(state_batch).squeeze().to(device)
    delta = reward_batch.to(device) - state_value.detach()
    delta = (delta - delta.mean()) / (delta.std() + float(np.finfo(np.float32).eps))
    policy_loss = (-log_probs.to(device) * delta.to(device)).mean().to(device)
    vf_loss = criterion(state_value, reward_batch.to(device))
    return policy_loss, vf_loss


def reinforce_baseline(policy_network: torch.nn.Module, value_function: torch.nn.Module, env, act, alpha_policy=1e-3, alpha_vf=1e-5, weight_decay=1e-5, exploration_rate=1, exploration_decay=(1-1e-4), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False, early_stopping=False, early_stopping_freq=100, val_env=None):# -> tuple[np.ndarray, np.ndarray]: 
    """
    REINFORCE with baseline

    Args: 
        policy_network (nn.Module): the policy network.
        value_function (nn.Module): the value function that is used as a baseline.
        env: the reinforcement learning environment that takes the action from the network and performs 
             a step where it returns the new state and the reward in addition to a flag that signals if 
             the environment has reached a terminal state.
        act: a function that uses the policy network to generate some output based on the state, and 
             then transforms that output to a problem-dependent action.
        alpha_policy (float): the learning rate on the interval [0,1] for the policy network.
        alpha_vf (float): the learning rate on the interval [0,1] for the value function.
        weight_decay (float): regularization parameter for the policy network and value function.
        exploration_rate (number): the intial exploration rate.
        exploration_decay (number): the rate of which the exploration rate decays over time.
        exploration_min (number): the minimum exploration rate. 
        num_episodes (int): the number of episodes to be performed. Not necessarily completed episodes 
                            depending on the next parameter max_episode_length.
        max_episodes_length (int): the maximal length of a single episode. 
        train (bool): wheteher the policy network is in train or evaluation mode. 
        print_res (bool): whether to print results after some number of episodes. 
        print_freq (int): the frequency of which the results after an episode are printed. 
        recurrent (bool): whether the policy network is recurrent or not. 
        early_stopping (bool): whether or not to use early stopping.
        early_stopping_freq (int): the frequency at which to test the validation set.
        val_env: the validation environment. 
    Returns:
        reward_history (np.ndarray): the sum of rewards for all completed episodes. 
        action_history (np.ndarray): the array of all actions of all completed episodes.
    """
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=alpha_policy, weight_decay=weight_decay)
    optimizer_vf = optim.Adam(value_function.parameters(), lr=alpha_vf, weight_decay=weight_decay)
    reward_history = []
    action_history = []
    total_rewards = []
    total_actions = []
    validation_rewards = []
    completed_episodes_counter = 0
    done = False
    state = env.reset() #S_0

    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        policy_network.eval()
        value_function.eval()
    else:
        policy_network.train()
        value_function.train()

    for n in range(num_episodes):
        rewards = [] 
        actions = [] 
        log_probs = []  
        states = []
        hx = None

        for _ in range(max_episode_length):
            action, log_prob, hx = act(policy_network, state, hx, recurrent, exploration_rate) #A_{t-1}
            state, reward, done, _ = env.step(action)#.to(device) # S_t, R_t 
            
            if done:
                break

            actions.append(action)
            rewards.append(reward) 
            log_probs.append(log_prob)
            states.append(torch.from_numpy(state).float().unsqueeze(0).to(device))
            exploration_rate = max(exploration_rate*exploration_decay, exploration_min)

        if train:
            reward_batch = torch.FloatTensor(rewards)
            state_batch = torch.cat(states)
            log_probs = torch.stack(log_probs).squeeze()
            policy_loss, vf_loss = get_policy_and_value_loss(value_function, state_batch, reward_batch, log_probs)
            optimize(optimizer_policy, policy_loss)
            optimize(optimizer_vf, vf_loss)

        total_rewards.extend(rewards)
        total_actions.extend(actions)

        if done: 
            reward_history.append(sum(total_rewards))
            action_history.append(np.array(total_actions))
            state = env.reset() #S_0
            total_rewards = []
            total_actions = []
            completed_episodes_counter += 1

        if done and print_res and (completed_episodes_counter-1) % print_freq == 0:
            print("Completed episodes: ", completed_episodes_counter)                  
            print("Actions: ", action_history[-1])
            print("Sum rewards: ", reward_history[-1])
            print("-"*20)
            print()
        
        if done and early_stopping and completed_episodes_counter % early_stopping_freq == 0:
            val_reward, _ = reinforce_baseline(policy_network, value_function, val_env, act, train=False, num_episodes=1, print_res=False, recurrent=recurrent, exploration_rate=0, exploration_min=0)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                return np.array(reward_history), np.array(action_history)
            validation_rewards.append(val_reward)

    return np.array(reward_history), np.array(action_history)
