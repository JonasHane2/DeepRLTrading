import numpy as np
from batch_learning import ReplayMemory, Transition, get_batch
import torch
torch.manual_seed(0)
import torch.optim as optim
from action_selection import get_action_pobs
from reinforce import optimize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()


def compute_loss_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, recurrent=False) -> torch.Tensor: 
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


def deep_q_network(q_net, env, act, alpha=1e-4, weight_decay=1e-5, batch_size=10, exploration_rate=1, exploration_decay=(1-1e-3), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DQN

    Args: 
        q_net (): network that parameterizes the Q-learning function
        env: environment that the rl agent interacts with
        act: function that chooses action. depends on problem.
        learning_rate (float): 
        weight_decay (float): regularization 
        target_update_freq (int): 
        batch_size (int): 
        exploration_rate (float): 
        exploration_decay (float): 
        exploration_min (float): 
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent        
    """
    optimizer = optim.Adam(q_net.parameters(), lr=alpha, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(1000) # what capacity makes sense?
    reward_history = []
    action_history = []

    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        q_net.eval()
    else:
        q_net.train()
    
    
    for n in range(num_episodes):
        rewards = []
        actions = []
        state = env.reset() 
        hx = None

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
        
        if print_res:
            if n % print_freq == 0:
                print("Episode ", n)
                print("Actions: ", np.array(actions))
                print("Sum rewards: ", sum(rewards))
                print("-"*20)
                print()
        
        reward_history.append(sum(rewards))
        action_history.append(np.array(actions))

    return np.array(reward_history), np.array(action_history)
