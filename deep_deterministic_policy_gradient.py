import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from batch_learning import ReplayMemory, Transition, get_batch
from reinforce import optimize
from action_selection import get_action_pobs


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
    action = processing(action)
    q_sa = critic(state, action)
    loss = -1*torch.mean(q_sa) 
    return loss


def compute_critic_loss(critic, batch) -> torch.Tensor: 
    """ Returns error Q(s_t, a) - R_t+1 """
    state, action, reward, _ = batch
    reward = (reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps)) # does this actually improve performance here?
    q_sa = critic(state, action.view(action.shape[0], -1)).squeeze()
    loss = torch.nn.MSELoss()(q_sa, reward)
    return loss


def deep_determinstic_policy_gradient(actor_net, critic_net, env, act, processing, alpha_actor=1e-3, alpha_critic=1e-3, weight_decay=1e-4, batch_size=30, update_freq=1, exploration_rate=1, exploration_decay=(1-1e-3), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DDPG

    Args: 
        actor_net (): network that parameterizes the policy
        critic_net (): network that parameterizes the Q-learning function
        env: environment that the rl agent interacts with
        act: function that chooses action. depends on problem.
        leraning_rate_q (float): 
        leraning_rate_policy (float): 
        weight_decay (float): regularization 
        target_learning_rate (float): 
        batch_size (int): 
        exploration_rate (float): 
        exploration_decay (float): 
        exploration_min (float): 
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent
    """
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(1000)
    reward_history = []
    action_history = []

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
        state = env.reset() 
        hx = None

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
