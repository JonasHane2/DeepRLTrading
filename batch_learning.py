import random
from collections import namedtuple, deque
import torch
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[object]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


def get_batch(replay_buffer: ReplayMemory, batch_size: int) -> tuple[torch.Tensor]:
    """ Return a batch of concatenated S, A, R, S' values """
    if len(replay_buffer) < batch_size:
        return
    if batch_size < 2:
        raise ValueError("Argument batch_size must be integer >= 2")
    
    batch = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*batch))
    return torch.cat(batch.state), torch.cat(batch.action) , torch.cat(batch.reward), torch.cat(batch.next_state)
