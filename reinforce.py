from itertools import accumulate
import numpy as np
import torch
import torch.optim as optim
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gradient_clip_threshold = 1


def optimize(optimizer: optim.Adam, loss: torch.Tensor, net=None, gradient_clipping=False) -> None: 
    """ Set gradients to zero, backpropagate loss, optimization step """
    optimizer.zero_grad()
    loss.backward()
    if gradient_clipping:
        torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip_threshold) 
    optimizer.step()


def get_policy_loss(rewards: list, log_probs: list, normalize=True) -> torch.Tensor:
    """ Return policy loss """
    r = torch.FloatTensor(rewards).to(device)
    if len(r) > 1 and normalize:
        r = (r - r.mean()) / (r.std() + float(np.finfo(np.float32).eps))
    log_probs = torch.stack(log_probs).squeeze().to(device)
    policy_loss = torch.mul(log_probs, r).mul(-1).mean().to(device)
    return policy_loss


def reinforce(policy_network: torch.nn.Module, env, act, alpha=1e-3, 
              discount_factor=0, normalize_rewards=True, gradient_clipping=False,
              weight_decay=1e-5, exploration_rate=1, exploration_decay=0.9, 
              exploration_min=0, num_episodes=1000, 
              max_episode_length=np.iinfo(np.int32).max, train=True, 
              print_res=True, print_freq=100, recurrent=False, 
              early_stopping=False, early_stopping_freq=100, val_env=None):
    """
    REINFORCE/Monte Carlo policy gradient algorithm

    Args: 
        policy_network (nn.Module): the policy network. 
        env: the reinforcement learning environment which takes an action and 
             returns the new state, reward, and termination flag. 
        act: a function that uses the policy network to generate some output 
             based on the state, and then transforms that output to an action.
        alpha (float): the learning rate on [0,1] for the policy network. 
        discount_factor (float): number on [0,1] discounting future rewards.
        normalize_rewards (bool): normalizes the rewards in policy optimization.
        gradient_clipping (bool): clips gradients above a threshold in policy 
             optimization. 
        weight_decay (float): regularization parameter for the policy network.
        exploration_rate (number): the intial exploration rate.
        exploration_decay (number): the exploration decay rate.
        exploration_min (number): the minimum exploration rate. 
        num_episodes (int): the number of episodes to be performed. 
        max_episodes_length (int): the maximal length of a single episode. 
        train (bool): wheteher the policy network is in train or evaluation mode. 
        print_res (bool): whether to print results after some number of episodes. 
        print_freq (int): the frequency the results after an episode are printed. 
        recurrent (bool): whether the policy network is recurrent or not. 
        early_stopping (bool): whether or not to use early stopping.
        early_stopping_freq (int): the frequency at which to test the validation set.
        val_env: the validation environment. 
    Returns:
        reward_history (np.ndarray): the sum of rewards for all completed episodes.
        action_history (np.ndarray): the array of actions of all completed episodes.
    """
    optimizer = optim.Adam(policy_network.parameters(), 
                           lr=alpha, 
                           weight_decay=weight_decay)
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
    else:
        policy_network.train()

    done = False
    state = env.reset() #S_0
    total_rewards = []
    total_actions = []
    completed_episodes_counter = 0
    validation_rewards = []

    for n in range(num_episodes):
        rewards = [] 
        actions = [] 
        log_probs = []  
        hx = None

        for _ in range(max_episode_length):
            action, log_prob, hx = act(policy_network, state, hx, recurrent, exploration_rate) #A_{t-1}
            state, reward, done, _ = env.step(action) # S_t, R_t 

            if done:
                break

            actions.append(action)
            rewards.append(reward) 
            log_probs.append(log_prob)

        if train and rewards != []:
            weighted_rewards = list(accumulate(reversed(rewards), lambda x,y: x*discount_factor + y))[::-1]
            policy_loss = get_policy_loss(weighted_rewards, log_probs, normalize_rewards)
            optimize(optimizer, policy_loss, policy_network, gradient_clipping)

        total_rewards.extend(rewards)
        total_actions.extend(actions)

        if done: 
            reward_history.append(sum(total_rewards))
            action_history.append(np.array(total_actions))
            state = env.reset() #S_0
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
            val_reward, _ = reinforce(policy_network, 
                                      val_env, 
                                      act, 
                                      train=False, 
                                      num_episodes=1, 
                                      print_res=False, 
                                      recurrent=recurrent, 
                                      exploration_rate=0, 
                                      exploration_min=0)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                return np.array(reward_history), np.array(action_history)
            validation_rewards.append(val_reward)

    return np.array(reward_history), np.array(action_history)
