from torch import optim, nn
import torch as T


class NetworkBuilderDDDQN(nn.Module):
    def __init__(self, input_shape: tuple, hidden_dims: list, activation: str,
                 weight_init: bool, learning_rate: float, n_actions: int,
                 optimizer: str, loss: str, batch_norm: bool, dropout: float):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.weight_init = weight_init

        # NETWORK
        self.network = self.build_network()
        layer_idx = 2

        if self.dropout:
            layer_idx += 1
        if self.batch_norm:
            layer_idx += 1

        self.Q_V = nn.Linear(self.network[-layer_idx].out_features, 1)
        self.Q_A = nn.Linear(self.network[-layer_idx].out_features, self.n_actions)

        # OPTIMIZER
        self.optimizer_name = optimizer
        self.optimizer = self.__get_optimizer()

        self.scheduler = None

        # # LOSS
        self.loss_name = loss
        self.loss = self.__get_loss()

        # UTILS
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def build_network(self):
        # linear -> batch_norm -> activation -> dropout
        out = []
        # Linear layer
        layer = nn.Linear(*self.input_shape, self.hidden_dims[0])
        if self.weight_init:
            self.__init_weights(layer_weight=layer.weight)
        out.append(layer)
        # Batch norm
        if self.batch_norm:
            out.append(nn.LayerNorm(normalized_shape=self.hidden_dims[0]))
        # Activation
        out.append(self.__get_activation())
        # Dropout
        if self.dropout:
            out.append(nn.Dropout(self.dropout))

        previous_dims = self.hidden_dims[0]
        for dims in self.hidden_dims[1:-1]:
            # Linear layer
            layer = nn.Linear(previous_dims, dims)
            # Weight initializer
            if self.weight_init:
                self.__init_weights(layer_weight=layer.weight)
            out.append(layer)
            # Batch norm
            if self.batch_norm:
                # out.append(F.batch_norm(input=dims))
                out.append(nn.LayerNorm(normalized_shape=dims))
            # Activation
            out.append(self.__get_activation())
            # Dropout
            if self.dropout:
                out.append(nn.Dropout(self.dropout))
            previous_dims = dims

        return nn.Sequential(*out)

    def forward(self, state: T.tensor):
        state_1 = self.network(state)
        V = self.Q_V(state_1)
        A = self.Q_A(state_1)
        return V, A

    def __get_activation(self):
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'tanh':
            return nn.Tanh()

    def __get_loss(self):
        if self.loss_name.lower() == 'mse':
            return nn.MSELoss()

    def __get_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            return T.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return T.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return T.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'amsgrad':
            return T.optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=True)

    def __init_weights(self, layer_weight):
        if self.activation == 'relu' or self.activation == 'leaky_relu':
            nn.init.kaiming_uniform_(layer_weight,
                                     mode='fan_in',
                                     nonlinearity=self.activation)
        elif self.activation == 'tanh':
            nn.init.xavier_uniform_(layer_weight,
                                    gain=1.0)

    def reduce_learning_rate(self, factor):
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: factor ** epoch)

        self.optimizer.step()
        self.scheduler.step()

    def __repr__(self) -> str:
        return "network"
