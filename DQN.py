import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

# import Utilities


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# meta controller network
class hDQN(nn.Module):
    def __init__(self, params):
        # utilities = Utilities.Utilities()
        self.params = params
        super(hDQN, self).__init__()
        env_layer_num = self.params.OBJECT_TYPE_NUM + 1  # +1 for agent layer

        kernel_size = 2
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=128,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=96,
                               kernel_size=kernel_size + 1)
        self.conv3 = nn.Conv2d(in_channels=96,
                               out_channels=96,
                               kernel_size=kernel_size + 2)

        self.fc1 = nn.Linear(in_features=96 * 4,
                             out_features=320)

        self.fc2 = nn.Linear(in_features=320 + self.params.OBJECT_TYPE_NUM + 4,  # +4: 2 for b matrix, and 2 other for u
                             out_features=256)  # Try dense architecture, or try multiplying the value for parameters by 320/6

        self.fc3 = nn.Linear(in_features=256,
                             out_features=192)

        self.fc4 = nn.Linear(in_features=192,
                             out_features=128)
        self.fc5 = nn.Linear(in_features=128,
                             out_features=64)

    def forward(self, env_map, mental_states, states_params):
        batch_size = env_map.shape[0]

        y = F.relu(self.conv1(env_map))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        # y = torch.concat([y, agent_need], dim=1)
        # y = self.batch_norm(y)
        y = F.relu(self.fc1(y))
        y = torch.concat([y, mental_states, states_params], dim=1)
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))
        y = self.fc5(y)

        y = y.reshape(batch_size,
                      self.params.HEIGHT,
                      self.params.WIDTH)
        return y
