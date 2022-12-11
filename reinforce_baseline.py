import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from reinforce import optimize
criterion = torch.nn.MSELoss()


def get_policy_and_value_loss(value_function, state_batch, reward_batch, log_probs) -> tuple[torch.Tensor, torch.Tensor]:
    state_value = value_function(state_batch).squeeze()
    delta = reward_batch - state_value.detach()
    delta = (delta - delta.mean()) / (delta.std() + float(np.finfo(np.float32).eps))
    policy_loss = (-log_probs * delta).mean() 
    vf_loss = criterion(state_value, reward_batch)
    return policy_loss, vf_loss


def reinforce_baseline(policy_network: torch.nn.Module, value_function: torch.nn.Module, env, act, alpha_policy=1e-3, alpha_vf=1e-5, weight_decay=1e-5, exploration_rate=1, exploration_decay=(1-1e-4), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=alpha_policy, weight_decay=weight_decay)
    optimizer_vf = optim.Adam(value_function.parameters(), lr=alpha_vf, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        policy_network.eval()
        value_function.eval()
    else:
        policy_network.train()
        value_function.train()

    for n in range(num_episodes):
        state = env.reset() #S_0
        rewards = [] 
        actions = [] 
        log_probs = []  
        states = []
        hx = None

        for _ in range(max_episode_length):
            action, log_prob, hx = act(policy_network, state, hx, recurrent, exploration_rate) #A_{t-1}
            state, reward, done, _ = env.step(action) # S_t, R_t 
            
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
