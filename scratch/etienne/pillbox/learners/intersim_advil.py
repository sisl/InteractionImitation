import torch
import torch.nn as nn

def unnormalize(val, mean, std):
    val *= std or 1
    val += mean or 0
    return val

def normalize(val, mean, std):
    val -= mean or 0
    val /= std or 1
    return val

class IntersimPolicy(nn.Module):
    def __init__(self, env, mean=None, std=None):
        # assert "intersim" in env.unwrapped.spec.id
        super().__init__()

        self._ego_encoder = nn.Sequential(
            # in 5, out 5
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self._state_encoder = nn.Sequential(
            # in 5, out 5
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self._deepset = lambda e: e.sum(-2)
        self._action_decoder = nn.Sequential(
            # in 5 + 5, out 1
            nn.Linear(5 + 5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
    
    def forward(self, obs):
        # obs.shape = (batch=514, 1 + others=150, 5)
        # act.shape = (batch=514, 1)
        
        ego = obs[:, 0]#.detach().clone()
        rel = obs[:, 1:]#.detach().clone()
        nan = rel.isnan().any(-1, keepdim=True)
        rel = torch.where(nan, torch.zeros_like(rel), rel) # required because of https://github.com/pytorch/pytorch/issues/15506
        
        d = (rel[:, :, :2] ** 2).sum(-1).sqrt()
        front = torch.stack((torch.cos(ego[:, 3]), torch.sin(ego[:, 3])), -1)
        left = torch.stack((-torch.sin(ego[:, 3]), torch.cos(ego[:, 3])), -1)
        df = (rel[:, :, :2] * front.unsqueeze(1)).sum(-1)
        dl = (rel[:, :, :2] * left.unsqueeze(1)).sum(-1)
        alpha = torch.atan2(dl, df)
        
        rel[:, :, 0] = d
        rel[:, :, 1] = alpha

        e = self._ego_encoder(ego)
        x = self._state_encoder(rel)
        x = torch.where(nan, torch.zeros_like(x), x)
        x = self._deepset(x)
        a = self._action_decoder(torch.cat((e, x), 1))
        
        return 10 * a

    def predict(self, state, mask, deterministic):
        #action_distribution = self.forward(obs)
        #action = action_distribution.argmax()
        #return action
        return self.forward(obs)

class IntersimDiscriminator(nn.Module):
    def __init__(self, env):
        # assert "intersim" in env.unwrapped.spec.id
        super().__init__()
        
        self._ego_encoder = nn.Sequential(
            # in 5, out 5
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self._state_encoder = nn.Sequential(
            # in 5, out 5
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self._deepset = lambda e: e.sum(-2)
        self._discriminator = nn.Sequential(
            # in 5 + 5 + 1, out 1
            nn.Linear(5 + 5 + 1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
    
    def forward(self, obs, acts):
        # obs.shape = (batch=514, 1 + others=150, 5)
        # acts.shape = (batch=514, 1)
        # val.shape = (batch=514,)

        ego = obs[:, 0]
        rel = obs[:, 1:]
        nan = rel.isnan().any(-1, keepdim=True)
        rel = torch.where(nan, torch.zeros_like(rel), rel) # required because of https://github.com/pytorch/pytorch/issues/15506
        
        d = (rel[:, :, :2] ** 2).sum(-1).sqrt()
        front = torch.stack((torch.cos(ego[:, 3]), torch.sin(ego[:, 3])), -1)
        left = torch.stack((-torch.sin(ego[:, 3]), torch.cos(ego[:, 3])), -1)
        df = (rel[:, :, :2] * front.unsqueeze(1)).sum(-1)
        dl = (rel[:, :, :2] * left.unsqueeze(1)).sum(-1)
        alpha = torch.atan2(dl, df)
        
        rel[:, :, 0] = d
        rel[:, :, 1] = alpha
        
        e = self._ego_encoder(ego)
        x = self._state_encoder(rel)
        x = torch.where(nan, torch.zeros_like(x), x)
        x = self._deepset(x)
        v = self._discriminator(torch.cat((e, x, acts), 1))
        
        return v.squeeze(1)