import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl.utils import utils


class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef / 60.0
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_amp_traj, lambda_=10):
        expert_amp_traj.requires_grad = True
        expert_amp_traj = expert_amp_traj.view(-1, self.input_dim)

        disc = self.amp_linear(self.trunk(expert_amp_traj))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_amp_traj,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_disc_logit_reg(self, disc_logit_reg=0.01):
        return disc_logit_reg * torch.sum(torch.square(self.amp_linear.weight))

    def predict_amp_reward(self, amp_traj, task_reward, normalizer=None):
        _amp_traj = amp_traj.clone()
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                _amp_traj = normalizer.normalize_torch(_amp_traj, self.device)

            _amp_traj = _amp_traj.view(-1, self.input_dim)

            d = self.amp_linear(self.trunk(_amp_traj))
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            reward = style_reward.clone()
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), 60 * style_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r