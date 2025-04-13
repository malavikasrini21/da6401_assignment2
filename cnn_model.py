import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, num_layers, filters, kernel_size, activation_fn, dense_neurons):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = input_channels

        for _ in range(num_layers):
            conv = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size//2)
            bn = nn.BatchNorm2d(filters)
            act = activation_fn()
            pool = nn.MaxPool2d(2)
            layers.extend([conv, bn, act, pool])
            in_channels = filters

        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(filters * (224 // (2 ** num_layers)) ** 2, dense_neurons),
            activation_fn(),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
