import random
from collections import namedtuple, deque
import itertools
import torch
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


def get_batch(replay_buffer: ReplayMemory, batch_size: int, recurrent=False, env_copy=False):
    """ Return a batch of concatenated S, A, R, S' values in random order """
    if len(replay_buffer) < batch_size:
        return
    if batch_size < 2:
        raise ValueError("Argument batch_size must be integer >= 2")
    if recurrent:
        batch = get_sequential_batch(replay_buffer, batch_size)
    else:
        batch = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*batch))
    if env_copy:
        return torch.cat(batch.state), batch.action, torch.cat(batch.reward), torch.cat(batch.next_state)
    else:
        return torch.cat(batch.state), torch.cat(batch.action) , torch.cat(batch.reward), torch.cat(batch.next_state)


def get_sequential_batch(replay_buffer: ReplayMemory, batch_size: int):
    """ Returns a batch with random starting point, but in order
        so that it can be used to train recurrent neural networks. """
    latest_start = len(replay_buffer) - batch_size
    random_start = random.randint(0, latest_start) # [0, latest_start]
    batch = list(itertools.islice(replay_buffer.memory, random_start, (random_start+batch_size)))
    none_index = [i for i,v in enumerate(batch) if v[0] == None]
    if none_index != []:
        if none_index[0] > 1:
            batch = batch[:none_index[0]]
        else: 
            batch = batch[(none_index[0]+1):] 
    return batch
