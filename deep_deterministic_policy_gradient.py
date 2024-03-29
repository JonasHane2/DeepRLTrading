import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) # for relative imports
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from batch_learning import ReplayMemory, Transition, get_batch
from reinforce import optimize
from action_selection import get_action_pobs
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
#criterion = torch.nn.HuberLoss()


def soft_updates(net: torch.nn.Module, target_net: torch.nn.Module, tau: float):
    """ Ø' <- tØ' + (1-t)Ø """
    tau = min(1, max(0, tau))
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.mul_(tau)
            p_targ.data.add_((1 - tau) * p.data)


def update(replay_buffer: ReplayMemory, batch_size: int, 
           critic: torch.nn.Module, actor: torch.nn.Module, 
           critic_target: torch.nn.Module, actor_target: torch.nn.Module,
           optimizer_critic: torch.optim, optimizer_actor: torch.optim, 
           processing, discount_factor, recurrent=False, 
           normalize_rewards=True, normalize_critic=False,
           gradient_clipping=True) -> None: 
    """ Get batch, get loss and optimize critic, freeze q net, get loss and optimize actor, unfreeze q net """
    batch = get_batch(replay_buffer, batch_size, recurrent) 
    if batch is None:
        return
    if discount_factor <= 0: 
        c_loss = cumpute_critic_loss_alternative(critic, batch, normalize_rewards)
    else:
        c_loss = compute_critic_loss(critic, batch, actor_target, critic_target, processing, recurrent, discount_factor, normalize_rewards)
    optimize(optimizer_critic, c_loss, critic, gradient_clipping)
    for p in critic.parameters(): # Freeze Q-net
        p.requires_grad = False
    a_loss = compute_actor_loss(actor, critic, batch[0], batch[3], processing, recurrent, normalize_critic)
    optimize(optimizer_actor, a_loss, actor, gradient_clipping)
    for p in critic.parameters(): # Unfreeze Q-net
        p.requires_grad = True


def compute_actor_loss(actor, critic, state, prev_action, processing, recurrent=False, normalize=False) -> torch.Tensor: 
    """ Returns policy loss -Q(s, mu(s)) """
    action, _ = get_action_pobs(net=actor, state=state, recurrent=recurrent, prev_action=prev_action)
    action = processing(action).to(device)
    prev_action = torch.cat((action[0].unsqueeze(0), action[:-1]), dim=0).to(device)
    q_sa = critic(state.to(device), action.to(device), prev_action.to(device)).to(device)
    if len(q_sa) > 1 and normalize:
        q_sa = ((q_sa - q_sa.mean()) / (q_sa.std() + float(np.finfo(np.float32).eps))).to(device)
    loss = -1*torch.mean(q_sa) 
    return loss


def cumpute_critic_loss_alternative(critic, batch, normalize=True) -> torch.Tensor: 
    """ Returns error Q(s_t, a) - R_t+1 """
    state, action, reward, prev_action = batch
    q_sa = critic(state.to(device), action.view(action.shape[0], -1).to(device), prev_action.view(prev_action.shape[0], -1).to(device)).squeeze().to(device)
    #q_sa = critic(state.to(device), action.view(action.shape[0], -1).to(device)).squeeze().to(device)
    if len(reward) > 1 and normalize:
        reward = ((reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps))).to(device)
    loss = criterion(q_sa, reward.to(device))
    return loss


def compute_critic_loss(critic, batch, actor_target, critic_target, processing, recurrent, discount_factor=0, normalize=True) -> torch.Tensor: 
    """ Returns error Q(s_t, a) - (R_t+1 + Q(S_t+1, mu(s_t+1))) """
    # ! Doesnt work now that i have changed the replay memory and networks 
    state, action, reward, next_state = batch
    if len(reward) > 1 and normalize:
        reward = ((reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps))).to(device) # does this actually improve performance here?
    with torch.no_grad():
        if recurrent:
            a_hat, _ = actor_target(next_state)
        else: 
            a_hat = actor_target(next_state) 
        a_hat = processing(a_hat).to(device)
        q_sa_hat = critic_target(next_state, a_hat).squeeze()
    q_sa = critic(state.to(device), action.view(action.shape[0], -1).to(device)).squeeze().to(device)
    target = reward + discount_factor * q_sa_hat
    loss = criterion(q_sa, target.to(device))
    return loss


def deep_determinstic_policy_gradient(
        actor_net: nn.Module, critic_net: nn.Module, env, act, processing, 
        discount_factor=0, tau=0.2, normalize_rewards=True, normalize_critic=False,
        gradient_clipping=True,
        alpha_actor=1e-4, alpha_critic=1e-3, weight_decay=1e-4, batch_size=128, 
        update_freq=1, exploration_rate=1, exploration_decay=0.9, 
        exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, 
        train=True, print_res=True, print_freq=100, recurrent=False, 
        replay_buffer_size=1000, early_stopping=False, early_stopping_freq=100, val_env=None
    ):
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
        discount_factor (float): rate of discounting future rewards on interval [0,1]
        tau (float): learning rate for target networks on the interval [0,1]
        normalize_rewards (bool): normalizes the rewards in critic optimization.
        normalize_critic (bool): normalizes the critique in policy optimization.
        gradient_clipping (bool): clips gradients above a threshold in policy 
             optimization. 
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
    prev_action = None
    
    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        actor_net.eval()
        critic_net.eval()
    else:
        actor_net.train()
        critic_net.train()
    
    actor_target_net = deepcopy(actor_net)
    critic_target_net = deepcopy(critic_net)

    for n in range(num_episodes):
        rewards = []
        actions = []

        for i in range(max_episode_length):
            action, hx = act(actor_net, state, prev_action, hx, recurrent, exploration_rate, train) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                if recurrent: 
                    replay_buffer.push(None, None, None, None)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                               torch.FloatTensor(np.array([action])), 
                               torch.FloatTensor([reward]), 
                               torch.FloatTensor(np.zeros((1, action.shape[0]))) if prev_action is None else torch.FloatTensor(np.array(prev_action)))
                               #torch.from_numpy(next_state).float().unsqueeze(0).to(device))
            prev_action = torch.Tensor(action).unsqueeze(0)

            if train and len(replay_buffer) >= batch_size and (i+1) % update_freq == 0:
                update(replay_buffer=replay_buffer, 
                       batch_size=batch_size,
                       critic=critic_net,
                       actor=actor_net,
                       critic_target=critic_target_net,
                       actor_target=actor_target_net,
                       optimizer_critic=optimizer_critic,
                       optimizer_actor=optimizer_actor,
                       processing=processing,
                       discount_factor=discount_factor,
                       recurrent=recurrent, 
                       normalize_rewards=normalize_rewards,
                       normalize_critic=normalize_critic,
                       gradient_clipping=gradient_clipping)    
            
            if discount_factor > 0:
                soft_updates(critic_net, critic_target_net, tau)
                soft_updates(actor_net, actor_target_net, tau)
            state = next_state

        total_rewards.extend(rewards)
        total_actions.extend(actions)

        if done: 
            reward_history.append(sum(total_rewards))
            action_history.append(np.array(total_actions))
            state = env.reset()
            hx = None
            prev_action = None
            total_rewards = []
            total_actions = []
            exploration_rate = max(exploration_rate*exploration_decay, exploration_min)
            completed_episodes_counter += 1

        if done and print_res and (completed_episodes_counter-1) % print_freq == 0:
            print("Completed episodes: ", completed_episodes_counter)                  
            print("Actions: ", action_history[-1])
            print("Sum rewards: ", reward_history[-1])
            print("-"*20)
            print()
        
        if done and early_stopping and completed_episodes_counter % early_stopping_freq == 0:
            val_reward, _ = deep_determinstic_policy_gradient(actor_net, 
                                                              critic_net, 
                                                              val_env, 
                                                              act, 
                                                              processing, 
                                                              train=False, 
                                                              num_episodes=1, 
                                                              print_res=False, 
                                                              recurrent=recurrent, 
                                                              exploration_rate=0, 
                                                              exploration_min=0,
                                                              early_stopping=False)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                actor_net.load_state_dict(actor_net_copy.state_dict())
                critic_net.load_state_dict(critic_net_copy.state_dict())
                return np.array(reward_history), np.array(action_history)
            actor_net_copy = deepcopy(actor_net)
            critic_net_copy = deepcopy(critic_net)
            validation_rewards.append(val_reward[0])
            actor_net.train()
            critic_net.train()
    
    return np.array(reward_history), np.array(action_history)
