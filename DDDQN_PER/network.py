import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layer_sizes=(64, 64)):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Define the common feature extractor
        layers = []
        input_size = state_size
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, h_size))
            layers.append(nn.ReLU())
            input_size = h_size
        self.feature_extractor = nn.Sequential(*layers)

        # Define the advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], action_size)
        )

        # Define the value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], 1)
        )

    def forward(self, state):
        x = self.feature_extractor(state)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
