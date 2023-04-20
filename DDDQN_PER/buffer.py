import numpy as np
import random
from sumtree import SumTree


class PERBuffer:
    def __init__(self, buffer_size, batch_size, seed, alpha=0.6, beta=0.4):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def add(self, experience):
        max_priority = max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1
        self.tree.add(max_priority, experience)

    def sample(self):
        experiences = []
        idxs = []
        weights = []
        total_priority = self.tree.total()
        segment_priority = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = segment_priority * i
            b = segment_priority * (i + 1)
            value = random.uniform(a, b)
            idx, priority, experience = self.tree.get_leaf(value)
            weight = (1 / (self.tree.capacity * priority)) ** self.beta
            idxs.append(idx)
            weights.append(weight)
            experiences.append(experience)

        weights = np.array(weights) / np.max(weights)
        return idxs, experiences, weights

    def update_priorities(self, idxs, td_errors):
        priorities = np.abs(td_errors) + self.epsilon
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.tree.n_entries
