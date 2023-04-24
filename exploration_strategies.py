from abc import ABC, abstractmethod
import numpy as np


class ExplorationStrategy(ABC):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    @abstractmethod
    def choose_action(self, q_values: np.array) -> int:
        pass


class ThompsonSampling(ExplorationStrategy):
    def __init__(self, n_actions: int, beta: float = 1.0):
        super().__init__(n_actions)
        self.beta = beta

    def choose_action(self, q_values: np.array) -> int:
        samples = np.random.beta(q_values + 1, self.beta)
        return np.argmax(samples)


class UCB(ExplorationStrategy):
    def __init__(self, n_actions: int, c: float = 1.0, n: np.array = None):
        super().__init__(n_actions)
        self.c = c
        self.n = n if n is not None else np.ones(n_actions)

    def choose_action(self, q_values: np.array) -> int:
        upper_bounds = q_values + self.c * np.sqrt(np.log(np.sum(self.n)) / self.n)
        return np.argmax(upper_bounds)


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, n_actions: int, epsilon: float = 0.01):
        super().__init__(n_actions)
        self.epsilon = epsilon

    def choose_action(self, q_values: np.array) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(q_values)
