import parl
import torch
import numpy as np


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        super(MujocoAgent, self).__init__(algorithm)
        self.alg = algorithm

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        return self.alg.predict(obs).cpu().data.numpy().flatten()

    def learn(self, obs, act, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        self.alg.learn(obs, act, reward, next_obs, terminal)
