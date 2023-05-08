1. TIPS AND TRICKS

    - When using ReLU or leaky RELU, use He initialization also called Kaiming initialization.
    - When using SELU or ELU, use LeCun initialization.
    - When using softmax or tanh, use Glorot initialization also called Xavier initialization.

2. Dropout and Batch Normalization

    - Dropout is a regularization technique that “drops out” or “deactivates” a few neurons in the neural network
      randomly, in order to avoid the problem of over fitting. During training some neurons in the layer after which the
      dropout is applied are “turned off”. An ensemble of neural networks with fewer parameters (simpler model) reduces
      over fitting. Dropout simulates this phenomenon, contrary to snapshot ensembles of networks, without additional
      computational expense of training and maintaining multiple models. It introduces noise into a neural network to
      force it to learn to generalize well enough to deal with noise.
    - Batch Normalization is a technique to improve optimization. It’s a good practice to normalize the input data
      before training on it which prevents the learning algorithm from oscillating. We can say that the output of one
      layer is the input to the next layer. If this output can be normalized before being used as the input the learning
      process can be stabilized. This dramatically reduces the number of training epochs required to train deep
      networks. Batch Normalization makes normalization a part of the model architecture and is performed on
      mini-batches while training. Batch Normalization also allows the use of much higher learning rates and for us to
      be less careful about initialization.
    - ORDER:
        -      CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC


!pip install --upgrade pip pandas_ta wandb gym protobuf==3.20.* --upgrade pandas numexpr==2.8.4 
conda install -c conda-forge ta-lib

Disable gradients for the network.
Set your input tensor as a parameter requiring grad.
Initialize an optimizer wrapping the input tensor.
Backprop with some loss function and a goal tensor
...
Profit!
import torch

f = torch.nn.Linear(10, 5)
f.requires_grad_(False)
x = torch.nn.Parameter(torch.rand(10), requires_grad=True)
optim = torch.optim.SGD([x], lr=1e-1)
mse = torch.nn.MSELoss()
y = torch.ones(5)  # the desired network response

num_steps = 5  # how many optim steps to take
for _ in range(num_steps):
   loss = mse(f(x), y)
   loss.backward()
   optim.step()
   optim.zero_grad()
