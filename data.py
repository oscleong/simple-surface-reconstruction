#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch import distributions

def sample_plane(num_points, noise_var):
    X = torch.empty(num_points, 3).uniform_(0, 2)
    X[:,2] = 0
    noise = noise_var * torch.randn(num_points, 3)
    
    return X.double(), noise.double()

def sample_sphere(num_points, noise_var):
    X = np.random.randn(3, num_points)
    X /= np.linalg.norm(X, axis = 0)
    X = X.T
    X_torch = torch.from_numpy(X)
    noise = noise_var * torch.randn(num_points, 3)
    
    return X_torch.double(), noise.double()

