import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) # for relative imports
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
from batch_learning import ReplayMemory, Transition, get_batch
from action_selection import get_action_pobs
from reinforce import optimize
#from DeepRLTrading.batch_learning import ReplayMemory, Transition, get_batch
#from DeepRLTrading.action_selection import get_action_pobs
#from DeepRLTrading.reinforce import optimize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()


def compute_loss_dqn(batch: tuple, net: torch.nn.Module, recurrent=False) -> torch.Tensor: 
    """ Return critic loss 1/N * (Q(s_t, a_t) - R)^2 for t = 0,1,...,N """
    state_batch, action_batch, reward_batch, _ = batch
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))
    action_batch = action_batch.flatten().long().add(1) #add 1 because -1 actions before
    state_vals, _ = get_action_pobs(net=net, state=state_batch, recurrent=recurrent)
    state_action_vals = state_vals[range(action_batch.size(0)), action_batch]
    return criterion(state_action_vals, reward_batch)


def update(replay_buffer: ReplayMemory, batch_size: int, net: torch.nn.Module, optimizer: torch.optim, recurrent=False) -> None:
    """ Get loss and perform optimization step """
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    loss = compute_loss_dqn(batch, net, recurrent)
    optimize(optimizer, loss)


def deep_q_network(q_net, env, act, alpha=1e-4, weight_decay=1e-5, batch_size=10, exploration_rate=1, exploration_decay=(1-1e-3), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False, replay_buffer_size=1000, early_stopping=False, early_stopping_freq=100, val_env=None):# -> tuple[np.ndarray, np.ndarray]: 
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
    optimizer = optim.Adam(q_net.parameters(), lr=alpha, weight_decay=weight_decay)
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
        q_net.eval()
    else:
        q_net.train()
    
    
    for n in range(num_episodes):
        rewards = []
        actions = []

        for i in range(max_episode_length): 
            action, hx = act(q_net, state, hx, recurrent, exploration_rate) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor(np.array([action])), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and len(replay_buffer) >= batch_size:
                update(replay_buffer, batch_size, q_net, optimizer, recurrent)

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
            val_reward, _ = deep_q_network(q_net, val_env, act, train=False, num_episodes=1, print_res=False, recurrent=recurrent, exploration_rate=0, exploration_min=0)
            if len(validation_rewards) > 0 and val_reward[0] < validation_rewards[-1]:
                return np.array(reward_history), np.array(action_history)
            validation_rewards.append(val_reward)

    return np.array(reward_history), np.array(action_history)
