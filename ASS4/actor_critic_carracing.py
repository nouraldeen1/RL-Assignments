import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from encoders import CarRacingEncoder


class Actor(nn.Module):
    """
    Continuous policy for CarRacing using the CNN encoder.
    Outputs mean and log_std for a Gaussian policy producing 3-dim actions.
    """
    def __init__(self, feature_dim=256, action_dim=3, encoder_channels=3, hidden_dim=256, min_log_std=-20, max_log_std=2):
        super().__init__()
        self.encoder = CarRacingEncoder(input_channels=encoder_channels, feature_dim=feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        # independent log std parameters (state-independent)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, img):
        # img expected shape: (B, C, H, W)
        feats = self.encoder(img)
        h = self.fc(feats)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std_param, self.min_log_std, self.max_log_std)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def sample(self, img):
        mean, log_std = self.forward(img)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # For CarRacing, action components have different ranges: steering [-1,1], gas [0,1], brake [0,1].
        # Here we return tanh output in (-1,1) for all 3 dims. Caller can post-process gas/brake if needed.
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z


class QNetwork(nn.Module):
    """
    Critic network for continuous actions: takes image and action, returns Q-value.
    """
    def __init__(self, feature_dim=256, action_dim=3, hidden_dim=256, encoder_channels=3):
        super().__init__()
        self.encoder = CarRacingEncoder(input_channels=encoder_channels, feature_dim=feature_dim)
        # After encoding, concatenate action
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img, action):
        feats = self.encoder(img)
        x = torch.cat([feats, action], dim=-1)
        q = self.q_net(x)
        return q


# Lightweight helper MLP for LunarLander example
class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
