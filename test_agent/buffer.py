import numpy as np
import torch as T
from torch import nn


class TorchReplayBuffer(object):
    def __init__(self, memory_size: int, input_shape, network: nn.Module):
        self.memory_size = memory_size
        self.mem_cntr = 0
        self.state_memory = T.zeros((self.memory_size, *input_shape), dtype=T.float).to(network.device)
        self.new_state_memory = T.zeros((self.memory_size, *input_shape), dtype=T.float).to(network.device)
        self.action_memory = T.zeros(self.memory_size, dtype=T.int32).to(network.device)
        self.reward_memory = T.zeros(self.memory_size, dtype=T.float).to(network.device)
        self.terminal_memory = T.zeros(self.memory_size, dtype=T.bool).to(network.device)
        self.network = network

    def store_transition(self, state, state_, action, reward, done) -> None:
        index = self.mem_cntr % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = T.tensor(state_, dtype=T.float)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> (T.tensor, T.tensor, T.tensor, T.tensor, T.tensor):
        max_mem = min(self.mem_cntr, self.memory_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Return memory batches
        states = self.state_memory[batch]
        actions = self.action_memory[batch].cpu().numpy()
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
