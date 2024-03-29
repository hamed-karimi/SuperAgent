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

        kernel_size = 1
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=128,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=kernel_size + 1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=kernel_size + 2)
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=kernel_size + 2)
        each_channel_size = params.WIDTH - kernel_size - (kernel_size + 1) - (kernel_size + 2) - (kernel_size + 2) + 4
        self.fc1 = nn.Linear(in_features=64 * each_channel_size**2,
                             out_features=64)

        self.fc2 = nn.Linear(in_features=(each_channel_size**2+1)*64 + self.params.OBJECT_TYPE_NUM + 4,  # +4: 2 for b matrix, and 2 other for u
                             out_features=64)  # Try dense architecture, or try multiplying the value for parameters by 320/6

        self.fc3 = nn.Linear(in_features=(each_channel_size**2+2)*64 + (self.params.OBJECT_TYPE_NUM + 4),
                             out_features=64)

        self.fc4 = nn.Linear(in_features=(each_channel_size**2+3)*64 + (self.params.OBJECT_TYPE_NUM + 4),
                             out_features=64)
        self.fc5 = nn.Linear(in_features=(each_channel_size**2+4)*64 + (self.params.OBJECT_TYPE_NUM + 4),
                             out_features=64)

    def forward(self, env_map, mental_states, states_params):
        batch_size = env_map.shape[0]

        y = F.relu(self.conv1(env_map))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y0 = y.flatten(start_dim=1, end_dim=-1)
        # y = torch.concat([y, agent_need], dim=1)
        # y = self.batch_norm(y)
        y1 = F.relu(self.fc1(y0))
        y1 = torch.concat([y1, mental_states, states_params], dim=1)
        y2 = F.relu(self.fc2(torch.concat([y0, y1], dim=1)))  # or first relu then fc (try this for all layers)
        y3 = F.relu(self.fc3(torch.concat([y0, y1, y2], dim=1)))
        y4 = F.relu(self.fc4(torch.concat([y0, y1, y2, y3], dim=1)))
        y = self.fc5(torch.concat([y0, y1, y2, y3, y4], dim=1))

        y = y.reshape(batch_size,
                      self.params.HEIGHT,
                      self.params.WIDTH)
        return y
