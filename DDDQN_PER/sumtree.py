import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.pointer + self.capacity - 1
        self.update(idx, priority)
        self.data[self.pointer] = data
        self.pointer = (self.pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def get_leaf(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, value):
        left_child_idx = 2 * idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):
            return idx

        if value <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, value)
        else:
            return self._retrieve(right_child_idx, value - self.tree[left_child_idx])

    def total(self):
        return self.tree[0]
