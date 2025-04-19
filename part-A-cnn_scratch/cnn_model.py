import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, batch_norm):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvNeuralNet(nn.Module):
    def __init__(self,
                 in_dims,  # (C, H, W)
                 out_dims,
                 conv_activation: str,
                 dense_activation: str,
                 dense_size: int,
                 filter_size: list,
                 n_filters: int,
                 filter_org: str,
                 batch_norm: bool,
                 dropout: float):

        super(ConvNeuralNet, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.filter_org = filter_org
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout)

        self.conv_activation_func = self._get_activation(conv_activation)
        self.dense_activation_func = self._get_activation(dense_activation)

        self.filter_counts = self._init_filter_counts()

        self.convs = nn.ModuleList()
        in_channels = self.in_dims[0]
        for i in range(5):
            self.convs.append(ConvBlock(in_channels,
                                        self.filter_counts[i],
                                        self.filter_size[i],
                                        activation=self.conv_activation_func,
                                        batch_norm=self.batch_norm))
            in_channels = self.filter_counts[i]

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # compute flatten size after convs+pool
        sample_input = torch.zeros(1, *self.in_dims)
        with torch.no_grad():
            x = sample_input
            for conv in self.convs:
                x = self.pool(conv(x))
            flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, dense_size)
        self.fc2 = nn.Linear(dense_size, out_dims)
        self.softmax = nn.Softmax(dim=1)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _init_filter_counts(self):
        num_layers = 5
        if self.n_filters < 16:
            raise Exception("Minimum filters should be >= 16")
        filters = [self.n_filters]
        for _ in range(num_layers - 1):
            if self.filter_org == "same":
                filters.append(filters[-1])
            elif self.filter_org == "double":
                filters.append(filters[-1] * 2)
            elif self.filter_org == "halve":
                filters.append(max(filters[-1] // 2, 16))
            else:
                raise ValueError("filter_org must be one of: same, double, halve")
        return filters

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(conv(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.dense_activation_func(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x
