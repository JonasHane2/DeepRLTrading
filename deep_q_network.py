import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) # for relative imports
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from batch_learning import ReplayMemory, Transition, get_batch
from action_selection import get_action_pobs
from reinforce import optimize
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
#criterion = torch.nn.HuberLoss()


def compute_loss_dqn(batch: tuple, net: torch.nn.Module, recurrent=False, normalize=True) -> torch.Tensor: 
    """ Return critic loss 1/N * (Q(s_t, a_t) - R)^2 for t = 0,1,...,N """
    state_batch, action_batch, reward_batch, prev_action = batch
    if len(reward_batch) > 1 and normalize:
        reward_batch = ((reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))).to(device)
    state_vals, _ = get_action_pobs(net=net, state=state_batch, recurrent=recurrent, prev_action=prev_action)
    action_batch = action_batch.flatten().long().add(1) #add 1 because -1 actions before
    state_action_vals = state_vals[range(action_batch.size(0)), action_batch]
    return criterion(state_action_vals, reward_batch.to(device)).to(device)


def update(replay_buffer: ReplayMemory, batch_size: int, net: torch.nn.Module, optimizer: torch.optim, 
           recurrent=False, normalize_rewards=True, gradient_clipping=True) -> None:
    """ Get loss and perform optimization step """
    batch = get_batch(replay_buffer, batch_size, recurrent)
    if batch is None:
        return
    loss = compute_loss_dqn(batch, net, recurrent, normalize_rewards)
    optimize(optimizer, loss, net, gradient_clipping)


def deep_q_network(q_net, env, act, alpha=1e-4, weight_decay=1e-5, batch_size=64, 
                   update_freq=1, normalize_rewards=True, gradient_clipping=True,
                   exploration_rate=1, exploration_decay=0.9, exploration_min=0, 
                   num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, 
                   train=True, print_res=True, print_freq=100, recurrent=False, 
                   replay_buffer_size=1000, early_stopping=False, 
                   early_stopping_freq=100, val_env=None):
    """
    Deep Q Network 

    Args: 
        q_net (nn.Module): the q network. 
        env: the reinforcement learning environment that takes the action from the network and performs 
             a step where it returns the new state and the reward in addition to a flag that signals if 
             the environment has reached a terminal state.
        act: a function that uses the policy network to generate some output based on the state, and 
             then transforms that output to a problem-dependent action.
        alpha (float): the learning rate on the interval [0,1] for the q network. 
        weight_decay (float): regularization parameter for the q network.
        batch_size (int): the size of the training batches. 
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
    optimizer = optim.Adam(q_net.parameters(), 
                           lr=alpha, 
                           weight_decay=weight_decay)
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
        q_net.eval()
    else:
        q_net.train()
    
    
    for n in range(num_episodes):
        rewards = []
        actions = []

        for i in range(max_episode_length): 
            action, hx = act(q_net, state, prev_action, hx, recurrent, exploration_rate) 
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
                       net=q_net,
                       optimizer=optimizer,
                       recurrent=recurrent, 
                       normalize_rewards=normalize_rewards,
                       gradient_clipping=gradient_clipping)

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
            val_reward, _ = deep_q_network(q_net, 
                                           val_env, 
                                           act, 
                                           train=False, 
                                           num_episodes=1, 
                                           print_res=False, 
                                           recurrent=recurrent, 
                                           exploration_rate=0, 
                                           exploration_min=0,
                                           early_stopping=False)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                q_net.load_state_dict(q_net_copy.state_dict())
                return np.array(reward_history), np.array(action_history)
            q_net_copy = deepcopy(q_net)
            validation_rewards.append(val_reward[0])
            q_net.train()

    return np.array(reward_history), np.array(action_history)
