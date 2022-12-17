import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) # for relative imports
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from batch_learning import ReplayMemory, Transition, get_batch
from reinforce import optimize
from action_selection import get_action_pobs
#from DeepRLTrading.batch_learning import ReplayMemory, Transition, get_batch
#from DeepRLTrading.reinforce import optimize
#from DeepRLTrading.action_selection import get_action_pobs


def update(replay_buffer: ReplayMemory, batch_size: int, critic: torch.nn.Module, actor: torch.nn.Module, optimizer_critic: torch.optim, optimizer_actor: torch.optim, processing, recurrent=False) -> None: 
    """ Get batch, get loss and optimize critic, freeze q net, get loss and optimize actor, unfreeze q net """
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    c_loss = compute_critic_loss(critic, batch)
    optimize(optimizer_critic, c_loss)
    for p in critic.parameters(): # Freeze Q-net
        p.requires_grad = False
    a_loss = compute_actor_loss(actor, critic, batch[0], processing, recurrent)
    optimize(optimizer_actor, a_loss)
    for p in critic.parameters(): # Unfreeze Q-net
        p.requires_grad = True


def compute_actor_loss(actor, critic, state, processing, recurrent=False) -> torch.Tensor: 
    """ Returns policy loss -Q(s, mu(s)) """
    action, _ = get_action_pobs(net=actor, state=state, recurrent=recurrent)
    action = processing(action).to(device)
    q_sa = critic(state, action).to(device)
    loss = -1*torch.mean(q_sa) 
    return loss


def compute_critic_loss(critic, batch) -> torch.Tensor: 
    """ Returns error Q(s_t, a) - R_t+1 """
    state, action, reward, _ = batch
    reward = ((reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps))).to(device) # does this actually improve performance here?
    q_sa = critic(state, action.view(action.shape[0], -1)).squeeze().to(device)
    loss = torch.nn.MSELoss()(q_sa, reward)
    return loss


def deep_determinstic_policy_gradient(actor_net: nn.Module, critic_net: nn.Module, env, act, processing, alpha_actor=1e-3, alpha_critic=1e-3, weight_decay=1e-4, batch_size=30, update_freq=1, exploration_rate=1, exploration_decay=(1-1e-3), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False, replay_buffer_size=1000, early_stopping=False, early_stopping_freq=100, val_env=None):# -> tuple[np.ndarray, np.ndarray]: 
    """
    Deep Deterministic Policy Gradient

    Args: 
        actor_net (nn.Module): the actor network.
        critic_net (nn.Module): the critic network. 
        env: the reinforcement learning environment that takes the action from the network and performs 
             a step where it returns the new state and the reward in addition to a flag that signals if 
             the environment has reached a terminal state.
        act: a function that uses the policy network to generate some output based on the state, and 
             then transforms that output to a problem-dependent action.
        processing: a function that is used to process actions when calculating actor loss. 
                    It is the same processing that is used in the act function. 
        alpha_actor (float): the learning rate on the interval [0,1] for the actor network.
        alpha_critic (float): the learning rate on the interval [0,1] for the critic network.
        weight_decay (float): regularization parameter for the actor and critic networks.
        batch_size (int): the size of the training batches. 
        update_freq (int): the frequency at which training occurs. 
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
        replay_buffer_size (int): the size of the replay buffer. 
        early_stopping (bool): whether or not to use early stopping.
        early_stopping_freq (int): the frequency at which to test the validation set.
        val_env: the validation environment. 
    Returns:
        reward_history (np.ndarray): the sum of rewards for all completed episodes. 
        action_history (np.ndarray): the array of all actions of all completed episodes.
    """
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(replay_buffer_size)
    reward_history = []
    action_history = []
    total_rewards = []
    total_actions = []
    validation_rewards = []
    completed_episodes_counter = 0
    done = False
    state = env.reset() #S_0
    hx = None 

    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        actor_net.eval()
        critic_net.eval()
    else:
        actor_net.train()
        critic_net.train()
        
    for n in range(num_episodes):
        rewards = []
        actions = []

        for i in range(max_episode_length):
            action, hx = act(actor_net, state, hx, recurrent, exploration_rate, train) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor(np.array([action])), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and len(replay_buffer) >= batch_size and (i+1) % update_freq == 0:
                update(replay_buffer, batch_size, critic_net, actor_net, optimizer_critic, optimizer_actor, processing, recurrent)    
            
            state = next_state
            exploration_rate = max(exploration_rate*exploration_decay, exploration_min)

        total_rewards.extend(rewards)
        total_actions.extend(actions)

        if done: 
            reward_history.append(sum(total_rewards))
            action_history.append(np.array(total_actions))
            state = env.reset()
            hx = None
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
            val_reward, _ = deep_determinstic_policy_gradient(actor_net, critic_net, val_env, act, processing, train=False, num_episodes=1, print_res=False, recurrent=recurrent, exploration_rate=0, exploration_min=0)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                return np.array(reward_history), np.array(action_history)
            validation_rewards.append(val_reward)

    return np.array(reward_history), np.array(action_history)
