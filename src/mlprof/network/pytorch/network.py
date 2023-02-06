"""
network.py
"""
from __future__ import absolute_import, annotations, division, print_function
import torch

import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.models import resnet50

from mlprof.configs import ConvNetworkConfig, SpatialTransformerConfig


ACTIVATION_FNS = {
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'swish': nn.SiLU(),
    'leaky_relu': nn.LeakyReLU()
}


class SpatialTransformer(nn.Module):
    """Example taken from:
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()  # type:ignore
        self.fc_loc[2].bias.data.copy_(     # type:ignore
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(
            self,
            config: dict | ConvNetworkConfig | DictConfig,
    ):
        super(Net, self).__init__()
        self._with_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self._with_cuda else 'cpu'
        if isinstance(config, (dict, DictConfig)):
            config = instantiate(config)

        assert isinstance(config, ConvNetworkConfig)
        self.config = config

        self.activation_fn = ACTIVATION_FNS.get(
            self.config.activation_fn.lower(),
            None
        )
        assert callable(self.activation_fn)
        self.layers = nn.ModuleList([
            nn.LazyConv2d(self.config.filters1, 3),
            nn.LazyConv2d(self.config.filters2, 3),
            nn.MaxPool2d(2),
        ])
        self.layers.append(self.activation_fn)
        self.layers.extend([
            nn.Flatten(),
            nn.Dropout(self.config.drop1),
            nn.LazyLinear(self.config.hidden_size),
            nn.Dropout(self.config.drop2),
            nn.LazyLinear(10)
        ])

        if torch.cuda.is_available():
            self.cuda()
            self.layers.cuda()

    def get_config(self):
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)

        return F.log_softmax(x, dim=1)


def get_network(config: ConvNetworkConfig) -> nn.Module:
    return Net(config)
