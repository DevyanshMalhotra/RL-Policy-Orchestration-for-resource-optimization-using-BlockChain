import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self,s_dim,a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,128),
            nn.ReLU(),
            nn.Linear(128,a_dim),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+a_dim,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,s,a):
        return self.net(torch.cat([s,a],dim=1))

class RLAgent:

    def __init__(self,s_dim,a_dim,cfg):
        self.actor = Actor(s_dim,a_dim)
        self.critic = Critic(s_dim,a_dim)

        self.t_actor = Actor(s_dim,a_dim)
        self.t_critic = Critic(s_dim,a_dim)

        self.t_actor.load_state_dict(self.actor.state_dict())
        self.t_critic.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(),lr=cfg['lr_actor'])
        self.opt_c = optim.Adam(self.critic.parameters(),lr=cfg['lr_critic'])

        self.cfg = cfg
        self.step = 0

    def act(self, state):
        state = (state - np.mean(state)) / (np.std(state) + 1e-6)
        s = torch.from_numpy(state).float().unsqueeze(0)
        a = self.actor(s).detach().numpy().flatten()

        noise = np.random.normal(
            scale=self.cfg['noise_scale'] * (0.995 ** self.step),
            size=a.shape
        )
        self.step += 1

        return np.clip(a + noise, 0, 1)

    def learn(self, batch):

        s, a, r, s2, done = batch

        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        r = torch.from_numpy(r).unsqueeze(1).float()
        s2 = torch.from_numpy(s2).float()
        done = torch.from_numpy(done).unsqueeze(1).float()

        with torch.no_grad():
            next_a = self.t_actor(s2)
            target_q = self.t_critic(s2, next_a)
            y = r + self.cfg['gamma'] * (1 - done) * target_q
            y = torch.clamp(y, -10, 10)

        current_q = self.critic(s, a)

        loss_c = nn.MSELoss()(current_q, y)

        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        pred_a = self.actor(s)
        loss_a = -self.critic(s, pred_a).mean()

        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        tau = self.cfg['tau']

        for p, tp in zip(self.actor.parameters(), self.t_actor.parameters()):
            tp.data.copy_(tp.data*(1-tau) + p.data*tau)

        for p, tp in zip(self.critic.parameters(), self.t_critic.parameters()):
            tp.data.copy_(tp.data*(1-tau) + p.data*tau)

    def save(self, path="rl_agent.pt"):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path="rl_agent.pt"):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
