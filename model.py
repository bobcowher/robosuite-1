import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 300)
        self.bn1 = nn.BatchNorm1d(300, momentum=0.5)
        self.layer_2 = nn.Linear(300, 400)
        self.bn2 = nn.BatchNorm1d(400, momentum=0.5)
        self.layer_3 = nn.Linear(400, 300)
        self.bn3 = nn.BatchNorm1d(300, momentum=0.5)
        self.output = nn.Linear(300, action_dim)
        self.dropout = nn.Dropout(0.2)
        self.max_action = max_action

    def forward(self, x):
        # Ensure that the input tensor has at least 2 dimensions
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.layer_1(x))
        if self.training and x.shape[0] > 1:
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.layer_2(x))
        if self.training and x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.layer_3(x))
        if self.training and x.shape[0] > 1:
            x = self.bn3(x)
        x = self.max_action * torch.tanh(self.output(x))

        return x

    def save_the_model(self, weights_filename='actor_latest.pt'):
        weights_filename = "models/" + weights_filename
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='actor_latest.pt'):
        weights_filename = "models/" + weights_filename

        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
            return True
        except:
            print(f"No weights file available at {weights_filename}")
            return False

    def print_model(self):
        for name, param in self.named_parameters():
            print(name, param.data)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Critic Network.
        print(f"Critic initialized with input of {state_dim + action_dim}")
        self.layer_1 = nn.Linear(state_dim + action_dim, 300)
        self.layer_2 = nn.Linear(300, 400)
        self.layer_3 = nn.Linear(400, 300)
        self.output_1 = nn.Linear(300, 1)

        # Second critic network
        self.layer_4 = nn.Linear(state_dim + action_dim, 300)
        self.layer_5 = nn.Linear(300, 400)
        self.layer_6 = nn.Linear(400, 300)
        self.output_2 = nn.Linear(300, 1)

        self.bn1 = nn.BatchNorm1d(300, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(400, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(300, momentum=0.5)

        self.bn4 = nn.BatchNorm1d(300, momentum=0.5)
        self.bn5 = nn.BatchNorm1d(400, momentum=0.5)
        self.bn6 = nn.BatchNorm1d(300, momentum=0.5)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, u):

        xu = torch.cat([x, u], 1)

        # First critic forward prop
        x1 = F.relu(self.layer_1(xu))
        if self.training:
            x1 = self.bn1(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.layer_2(x1))
        if self.training:
            x1 = self.bn2(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.layer_3(x1))
        if self.training:
            x1 = self.bn3(x1)
        x1 = self.output_1(x1)

        # Second critic forward prop
        x2 = F.relu(self.layer_4(xu))
        if self.training:
            x2 = self.bn4(x2)
        x2 = self.dropout(x2)

        x2 = F.relu(self.layer_5(x2))
        if self.training:
            x2 = self.bn5(x2)
        x2 = self.dropout(x2)

        x2 = F.relu(self.layer_6(x2))
        if self.training:
            x2 = self.bn6(x2)
        x2 = self.output_2(x2)

        return x1, x2

    def Q1(self, x, u):

        xu = torch.cat([x, u], 1)

        # First critic forward prop
        x1 = F.relu(self.layer_1(xu))
        if self.training:
            x1 = self.bn1(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.layer_2(x1))
        if self.training:
            x1 = self.bn2(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.layer_3(x1))
        if self.training:
            x1 = self.bn3(x1)
        x1 = self.output_1(x1)

        return x1

    def save_the_model(self, weights_filename='critic_latest.pt'):
        weights_filename = "models/" + weights_filename
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='critic_latest.pt'):
        weights_filename = "models/" + weights_filename
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
            return True
        except:
            print(f"No weights file available at {weights_filename}")
            return False

    def print_model(self):
        for name, param in self.named_parameters():
            print(name, param.data)

