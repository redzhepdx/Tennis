import numpy as np
import torch
import torch.nn as nn


# Layer Initialization
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def weights_init(model):
    if isinstance(model, nn.Linear):
        print(model)
        model.weight.data.uniform_(*hidden_init(model))


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        if fc_units is None:
            fc_units = [128, 128]

        self.seed = torch.manual_seed(seed)
        self.state_norm = nn.BatchNorm1d(state_size)

        self.fc_units = fc_units
        self.fc_units.insert(0, state_size)
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU())
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.fc_out = nn.Linear(self.fc_units[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.state_norm(state)
        for layer_idx in range(len(self.fc_layers)):
            x = self.fc_layers[layer_idx](x)

        return torch.tanh(self.fc_out(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=0, fcs_units=None, fc_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        if fc_units is None:
            fc_units = [128]
        if fcs_units is None:
            fcs_units = [128]

        self.seed = torch.manual_seed(seed)

        self.state_norm = nn.BatchNorm1d(state_size)

        # Fully connected layers for state-only
        self.fcs_units = fcs_units
        self.fcs_units.insert(0, state_size)
        self.fcs_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(), nn.BatchNorm1d(fc_out))
                                         for fc_in, fc_out in zip(self.fcs_units[:-1], self.fcs_units[1:])])

        # Fully connected layers for state-action fused layers
        self.fc_units = fc_units
        self.fc_units.insert(0, self.fcs_units[-1] + action_size)
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU())
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.fc_out = nn.Linear(self.fc_units[-1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fcs_layers)):
            self.fcs_layers[layer_idx].apply(weights_init)

        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)

        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = self.state_norm(state)

        # Process state only
        for layer_idx in range(len(self.fcs_layers)):
            xs = self.fcs_layers[layer_idx](xs)

        # Fuse action and state
        x = torch.cat((xs, action), dim=1)

        # Process Fused
        for layer_idx in range(len(self.fc_layers)):
            x = self.fc_layers[layer_idx](x)
        return self.fc_out(x)
