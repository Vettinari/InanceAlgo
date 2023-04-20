import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNet(nn.Module):
    def __init__(self,
                 learning_rate: float,
                 n_actions: int,
                 input_shape: tuple,
                 l1_dims: int,
                 l2_dims: int):
        super().__init__()
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.n_actions = n_actions

        self.Q_fc_1 = nn.Linear(*self.input_shape, self.l1_dims)
        self.Q_fc_2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.Q_V = nn.Linear(self.l2_dims, 1)
        self.Q_A = nn.Linear(self.l2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.scheduler = None
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.tensor) -> (T.tensor, T.tensor):
        l1 = F.relu(self.Q_fc_1(state))
        l2 = F.relu(self.Q_fc_2(l1))
        V = self.Q_V(l2)
        A = self.Q_A(l2)
        return V, A

    def reduce_learning_rate(self, factor):
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: factor ** epoch)

        self.optimizer.step()
        self.scheduler.step()

    def __repr__(self) -> str:
        return "network"
