import torch
from torch import nn
import torch.nn.functional as F


class split_tanh(nn.Module):
    """
    An activation function that smoothes the initial activation function and is of form
    x \mapsto x + c * tanh(\lamb * (x - a))
    """

    def __init__(self):
        super(split_tanh, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.rand(1))
        self.a = torch.nn.parameter.Parameter(torch.randn(1))
        self.lamb = torch.nn.parameter.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return x + self.c * torch.tanh(self.lamb * (x - self.a))


class split_sign(nn.Module):
    """
    An activation function that splits the data in the naive way
    x \mapsto x + c * sign(x)
    """

    def __init__(self):
        super(split_sign, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.rand(1))
        self.c.requiresGrad = True

    def forward(self, x):
        return x + self.c * torch.sign(x)


class split_sincos(nn.Module):
    """
    An activation function that splits the data in the sin-cos way
    """

    def __init__(self):
        super(split_sincos, self).__init__()
        self.a = torch.nn.parameter.Parameter(torch.pi * torch.rand(1) / 2)
        self.b = torch.nn.parameter.Parameter(torch.rand(1))

    def forward(self, x):
        cosa = torch.cos(self.a)
        sina = torch.sin(self.a)
        tga = torch.tan(self.a)

        if x < -cosa:
            return self.b * x + self.b * cosa - sina
        if x > cosa:
            return x * tga
        return x + sina - cosa


class Classifier1L(nn.Module):
    def __init__(
        self,
        dim_of_in=2,
        num_of_hidden=2,
        dim_of_hidden=3,
        activation="split_tanh",
        batch_norm=False,
        initialize_weights=True,
        save_snapshots=True,
    ):
        """
        layers_signature = (dim_of_in, num_of_hidden, dim_of_hidden, is_with_split, split_constant)
        """
        super().__init__()

        self.activations = {
            "split_tanh": split_tanh(),
            "split_sign": split_sign(),
            "split_sincos": split_sincos(),
            "relu": F.relu,
        }

        self.fc_in = nn.Linear(dim_of_in, dim_of_hidden)
        self.norm = nn.BatchNorm1d(dim_of_hidden) if batch_norm else None
        self.hiddens = nn.ModuleList(
            [nn.Linear(dim_of_hidden, dim_of_hidden) for _ in range(num_of_hidden)]
        )
        self.fc_out = nn.Linear(dim_of_hidden, 2)

        self.activation_name = activation
        self.activation = self.activations[self.activation_name]

        if initialize_weights:
            self.apply(self.__initialize_weights)

        if save_snapshots:
            self.snapshots = []

    def forward(self, x, save=False):
        x = self.fc_in(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)

        for l in self.hiddens[:-1]:
            x = self.activation(l(x))

        x = F.relu(self.hiddens[-1](x))

        if save:
            self.snapshots.append(x.cpu().detach().numpy())

        x = self.fc_out(x)

        return x

    def __initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.activation_name != "relu":
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
