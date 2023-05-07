import numpy as np
import torch as T

from test_agent.buffer import TorchReplayBuffer
from test_agent.network import DuelingDeepQNet


class Agent:
    def __init__(self,
                 learning_rate: float,
                 n_actions: int,
                 input_shape: tuple,
                 l1_dims: int,
                 l2_dims: int,
                 memory_size: int,
                 batch_size: int,
                 epsilon: float,
                 eps_min: float,
                 eps_dec: float,
                 replace_target_counter: int,
                 gamma: float):
        self.learning_rate = learning_rate
        self.input_shape = input_shape  # ok
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.gamma = gamma  # ok
        self.epsilon = epsilon  # ok
        self.eps_min = eps_min  # ok
        self.eps_dec = eps_dec  # ok
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_counter = replace_target_counter

        # OBJECTS
        self.Q_eval = DuelingDeepQNet(learning_rate=self.learning_rate,
                                      n_actions=self.n_actions,
                                      input_shape=self.input_shape,
                                      l1_dims=self.l1_dims,
                                      l2_dims=self.l2_dims)

        self.Q_next = DuelingDeepQNet(learning_rate=self.learning_rate,
                                      n_actions=self.n_actions,
                                      input_shape=self.input_shape,
                                      l1_dims=self.l1_dims,
                                      l2_dims=self.l2_dims)

        self.memory = TorchReplayBuffer(memory_size=self.memory_size,
                                        input_shape=self.input_shape,
                                        network=self.Q_eval)

    def choose_action(self, observation: np.array) -> np.array:
        # observation = T.tensor(observation, dtype=T.float)
        if np.random.random() > self.epsilon:
            state = observation.to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self) -> None:
        if self.learn_step_counter % self.replace_target_counter == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self) -> None:
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state: np.array, action: np.array, reward: np.array,
              state_: np.array, done: np.array) -> None:
        self.memory.store_transition(state=state,
                                     action=action,
                                     reward=reward,
                                     state_=state_,
                                     done=done)

        if self.memory.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, terminal = self.memory.sample_buffer(batch_size=self.batch_size)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_next.forward(states_)

        V_s_eval, A_s_eval = self.Q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[batch_index, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[terminal] = 0.0
        q_target = rewards + self.gamma * q_next[batch_index, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
