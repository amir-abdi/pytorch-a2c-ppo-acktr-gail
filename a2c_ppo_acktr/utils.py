import torch
import torch.nn as nn
import numpy as np
import sys

from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
        # print('*********self._bias ', self._bias)
        # print('*********self._bias.t() ', self._bias.t())

    def forward(self, x):

        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        # print('*****bias', bias)
        # print('*****x', x)
        # print('********************', bias.mean().detach().cpu().item())
        if torch.isnan(bias.mean()):
            print('******************** detected nan!, exiting')
            sys.exit()
            # bias = nn.Parameter((torch.ones(210) * -1).unsqueeze(1).cuda()).t().view(1, -1)
        #     self._bias = nn.Parameter((torch.ones(210) * -1).unsqueeze(1).cuda())

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
