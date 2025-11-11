import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class DiscreteActor(nn.Module):
    def __init__(self,hidden_dim ,obs_dim, act_dim):
        super().__init__()       
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(), 
            nn.Linear(hidden_dim,act_dim)
        )   
 

    def forward(self,obs):

        logits = self.net(obs)
        dist = Categorical(logits=logits)
        return dist

    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
       
class ContinuouActor(nn.Module):
    def __init__(self,hidden_dim ,obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh()
        )
        self.mu_layer = nn.Linear(hidden_dim,act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.act_limit = act_limit

    def forward(self,obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std)
        return Normal(mu , std)
    
    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        action_clipped = torch.tanh(action) * self.act_limit
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action_clipped, log_prob