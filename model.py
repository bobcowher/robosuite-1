import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.layer_1 = nn.Linear(1568, 400)
        self.ln1 = nn.LayerNorm(400)
        self.layer_2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.output = nn.Linear(300, action_dim)  # Adjusted output layer
        self.dropout = nn.Dropout(0.5)
        self.max_action = max_action

        self._init_weights()



    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # Process image through the convolution layers and flatten.
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        x = self.flatten(x)

        x = F.leaky_relu(self.layer_1(x))

        # if self.training and x.shape[0] > 1:
        #     x = self.ln1(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.layer_2(x))
        # if self.training and x.shape[0] > 1:
        #     x = self.ln2(x)
        x = self.dropout(x)

        x = self.max_action * torch.tanh(self.output(x))  # Adjusted output layer

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

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))

        # First Critic Network
        self.state_value1 = nn.Linear(1568, 400)
        # self.state_value2 = nn.Linear(300, 400)
        # self.state_value3 = nn.Linear(400, 1)  # Adjusted output layer

        self.action_value1 = nn.Linear(action_dim, 400)
        # self.action_value2 = nn.Linear(300, 400)
        # self.action_value3 = nn.Linear(400, 1)  # Adjusted output layer

        self.combined_layer = nn.Linear(800, 300)
        self.output_layer = nn.Linear(300, 1)  # Adjusted output layer

        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)


    def forward(self, x, u):

        x = torch.Tensor(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.flatten(x)

        state_value = F.leaky_relu(self.state_value1(x))
        state_value = self.dropout(state_value)

        action_value = F.leaky_relu(self.action_value1(u))
        action_value = self.dropout(action_value)

        combined_value = torch.cat([state_value, action_value], 1)

        combined_value = F.leaky_relu(self.combined_layer(combined_value))

        output = self.output_layer(combined_value)

        return output

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

