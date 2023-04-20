import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

from DDDQN_PER.buffer import PERBuffer
from DDDQN_PER.network import DuelingQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, hidden_layer_sizes=(64, 64),
                 gamma=0.99, tau=1e-3, learning_rate=5e-4, update_every=4, alpha=0.6, beta=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.current_step = 0

        # Initialize the local and target Q-networks
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, hidden_layer_sizes).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, hidden_layer_sizes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Initialize the prioritized experience replay buffer
        self.memory = PERBuffer(buffer_size, batch_size, seed, alpha, beta)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        self.current_step = (self.current_step + 1) % self.update_every
        if self.current_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences, idxs, weights = self.memory.sample()
                self.learn(experiences, idxs, weights, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxs, weights, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        # Compute the local Q-values
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute the target Q-values using the double DQN method
        next_local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_local_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute the TD errors and update the priorities
        td_errors = (Q_targets - Q_expected).detach().squeeze().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Update the local Q-network
        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_targets, reduction='none') * weights
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # Update the target Q-network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
