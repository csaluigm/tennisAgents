
import torch
import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.stack_by_field("state",experiences).float().to(self.device)
        actions = self.stack_by_field("action",experiences).float().to(self.device)
        rewards = self.stack_by_field("reward",experiences).float().to(self.device)
        next_states = self.stack_by_field("next_state",experiences).float().to(self.device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def stack_by_field(self,field_name,experiences):
        return torch.from_numpy(np.vstack([getattr(e, field_name) for e in experiences if e is not None]))
    
    def __len__(self):
        return len(self.memory)