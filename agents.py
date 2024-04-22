import gym
import numpy as np
from torch import nn, optim


class LinearNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        action_shape,
        hidden_shape=[128, 128],
        learning_rate=0.00025,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.action_space = action_shape
        self.hidden_shape = hidden_shape

        layers = []
        for i in range(len(hidden_shape)):
            if i == 0:  # input layer
                layers.append(
                    (f"linear{i}", nn.linear(self.input_shape, self.hidden_shape[0]))
                )
                layers.append((f"relu{i}", nn.ReLU))
            elif i == len(hidden_shape) - 1:  # output layer
                layers.append(
                    (f"linear{i}", nn.linear(self.hidden_shape[-1], self.action_space))
                )
            else:  # hidden layer
                layers.append(
                    (
                        f"linear{i}",
                        nn.linear(self.hidden_shape[i - 1], self.hidden_shape[i]),
                    )
                )
                layers.append((f"relu{i}", nn.ReLU))

        # build an MLP with 2 hidden layers
        self.layers = nn.Sequential(nn.OrderedDict(layers))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)
