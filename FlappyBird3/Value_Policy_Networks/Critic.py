import torch.nn as nn

class ContinuousCritic(nn.Module):
    def __init__(self,hidden_dim, obs_dim=0):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )      

    def forward(self,obs):
        return self.value(obs)    
    
class DiscreteCritic(nn.Module):
    def __init__(self,hidden_dim, obs_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,obs):
        return self.value(obs)        