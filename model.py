import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torchsummary import summary
dtype = torch.float
device = torch.device('cpu')

def build_input(num_points, D_in):
    v = torch.randn((num_points, D_in), device = device, dtype = dtype, requires_grad = True)
    return v

def build_network(D_in, D_out, hidden1, hidden2):
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden1, hidden2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden2, D_out))
    return model