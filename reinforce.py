#Pay attention to possible exploding gradient for certain hyperparameters
#torch.nn.utils.clip_grad_norm_(net.parameters(), 1) 
#Maybe implement a safety mechanism that clips gradients 
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim


def optimize(optimizer: optim.Adam, loss: torch.Tensor) -> None: 
    """ Set gradients to zero, backpropagate loss, and take optimization step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_policy_loss(rewards: list, log_probs: list) -> torch.Tensor:
    """ Return policy loss """
    r = torch.FloatTensor(rewards)
    r = (r - r.mean()) / (r.std() + float(np.finfo(np.float32).eps))
    log_probs = torch.stack(log_probs).squeeze()
    policy_loss = torch.mul(log_probs, r).mul(-1).sum()
    return policy_loss


def reinforce(policy_network: torch.nn.Module, env, act, alpha=1e-3, weight_decay=1e-5, exploration_rate=1, exploration_decay=(1-1e-4), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
    optimizer = optim.Adam(policy_network.parameters(), lr=alpha, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        policy_network.eval()
    else:
        policy_network.train()

    for n in range(num_episodes):
        state = env.reset() #S_0
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
            exploration_rate = max(exploration_rate*exploration_decay, exploration_min)

        if train:
            policy_loss = get_policy_loss(rewards, log_probs)
            optimize(optimizer, policy_loss)

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
