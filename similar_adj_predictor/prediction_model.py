import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from config_params import *

class MLPNet(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super(MLPNet, self).__init__()
        torch.set_num_threads(THREADS)
        self.linear_input = nn.Linear(d_in, d_hidden)
        torch.nn.init.xavier_uniform(self.linear_input.weight)
        self.hidden_linears = nn.ModuleList([nn.Linear(d_hidden,d_hidden) for i in xrange(0,HIDDEN_LAYERS-1)])
        for hidden_lin in self.hidden_linears:
            torch.nn.init.xavier_uniform(hidden_lin.weight)
        self.out_layer = nn.Linear(d_hidden,d_out)
        torch.nn.init.xavier_uniform(self.out_layer.weight)
        #Todo: think about initialization (common initailization for tanh xavier, glorot)
        #try to icrease th hidden dimenssion

    def forward(self, x):
        x = F.tanh(self.linear_input(x))
        for i, linear in enumerate(self.hidden_linears):
            x = F.tanh(linear(x))
        x = self.out_layer(x)
        return x


class ShallowNet(nn.Module):
    def __init__(self, D_in, D_out):
        super(ShallowNet, self).__init__()
        self.linear_1 = nn.Linear(D_in,D_out,bias=True)
        weights = np.concatenate((np.identity(D_out), np.identity(D_out)),axis=1)
        self.linear_1.weight.data = torch.Tensor(weights)

    def forward(self, x):
        return self.linear_1(x)