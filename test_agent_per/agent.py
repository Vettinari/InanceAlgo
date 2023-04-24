import numpy as np
import torch as T
import torch
from test_agent.network import DuelingDeepQNet
from test_agent_per.per_buffer import PERBuffer


class Agent:
    def __init__(self, input_shape, n_actions, lr, gamma, epsilon, epsilon_min, epsilon_dec, mem_size, batch_size,
                 l1_dims, l2_dims, alpha, beta, beta_increment, replace):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step_counter = 0

        self.memory = PERBuffer(mem_size, alpha, beta, beta_increment)

        self.Q_eval = DuelingDeepQNet(learning_rate=lr, n_actions=n_actions, input_shape=input_shape,
                                      l1_dims=l1_dims, l2_dims=l2_dims)
        self.Q_next = DuelingDeepQNet(learning_rate=lr, n_actions=n_actions, input_shape=input_shape,
                                      l1_dims=l1_dims, l2_dims=l2_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            if type(observation) != torch.Tensor:
                observation = T.tensor(observation, dtype=T.float).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(observation)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def store_transition(self, state, action, reward, state_, done):
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        state_ = np.array(state_, dtype=np.float32)
        done = np.array(done, dtype=np.bool)
        self.memory.add((state, action, reward, state_, done), 1)

    def sample_memory(self):
        idxs, samples, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, states_, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(states_), np.array(dones), idxs, weights

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        if self.memory.tree.write < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones, idxs, weights = self.sample_memory()

        # print("States: ", type(states))

        states = T.tensor(states).to(self.Q_eval.device)
        actions = T.tensor(actions).to(self.Q_eval.device)
        rewards = T.tensor(rewards).to(self.Q_eval.device)
        states_ = T.tensor(states_).to(self.Q_eval.device)
        dones = T.tensor(dones).to(self.Q_eval.device)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_next.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[np.arange(self.batch_size), actions.long()]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next.max(dim=1).values

        errors = (q_target - q_pred).cpu().detach().numpy()
        self.memory.update_priorities(idxs, errors)

        weights = T.tensor(weights, dtype=T.float).to(self.Q_eval.device)
        loss = T.mean(T.mul((q_target - q_pred).pow(2), weights))

        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
