import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random


Transition = namedtuple('Transition', ('state', 'next_state', 'action','reward', 'mask'))

class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
        self.memory[self.position] = Transition(state, next_state, action, reward, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)




# nn: take state from env as input
class DQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, input_feature, outputs_feature):
        nn.Module.__init__(self)
        self.f1 = nn.Linear(input_feature, 128)
        self.f2 = nn.Linear(128, outputs_feature)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.f1(x))
        return self.f2(x)



