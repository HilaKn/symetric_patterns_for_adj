import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from config_params import *

class MLPNet(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super(MLPNet, self).__init__()
        self.linear_input = nn.Linear(d_in, d_hidden)
        self.hidden_linears = nn.ModuleList([nn.Linear(d_hidden,d_hidden) for i in xrange(0,HIDDEN_LAYERS-1)])
        self.out_layer = nn.Linear(d_hidden,d_out)
        #Todo: think about initialization

    def forward(self, x):
        x = F.tanh(self.linear_input(x))
        for i, linear in enumerate(self.hidden_linears):
            x = F.tanh(linear(x))
        x = self.out_layer(x)
        return x
