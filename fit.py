import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import ot # optimal transport solver
from helpers import *
from torchsummary import summary
dtype = torch.float
device = torch.device('cpu')

def fit_local_patch(v1, model1, v2, model2, loss_fn, maxiter, noise_level, lr, P1, P2, Y, X):
    # Step 1: fit local patches to the surface
    print("Fitting of local patches...")  
    error_train_net1 = []
    error_test_net1 = []
    error_train_net2 = []
    error_test_net2 = []
    optimizer_w1 = torch.optim.Adam(model1.parameters(), lr)
    optimizer_w2 = torch.optim.Adam(model2.parameters(), lr)
    iteration = 0
    while iteration < maxiter:

        # Prediction & pairwise distance matrices 
        Y1_pred = model1(v1)
        Y2_pred = model2(v2)
        C1 = pairwise_distance_matrix(Y1_pred, Y).double()
        C2 = pairwise_distance_matrix(Y2_pred, Y).double()
        C1x = pairwise_distance_matrix(Y1_pred, X).double()
        C2x = pairwise_distance_matrix(Y2_pred, X).double()
    
        # Losses
        # Train loss
        train_loss_net1 = torch.mul(P1, C1).sum().sum()
        train_loss_net2 = torch.mul(P2, C2).sum().sum()
        total_train_loss = train_loss_net1 + train_loss_net2
    
        # Test loss
        test_loss_net1 = torch.mul(P1, C1x).sum().sum() 
        test_loss_net2 = torch.mul(P2, C2x).sum().sum()   
    
        # Optimization
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        total_train_loss.backward()
        optimizer_w1.step()
        optimizer_w2.step()
        
        if iteration % 1000 == 0:
            print("Iteration:" + str(iteration) 
                      + ", Training loss (net 1):" + str(train_loss_net1.item()) 
                      + ", Test loss (net 1):" + str(test_loss_net1.item()))
            print("Iteration:" + str(iteration) 
                      + ", Training loss (net 2):" + str(train_loss_net2.item()) 
                      + ", Test loss (net 2):" + str(test_loss_net2.item()))
            print("Noise level:" + str(noise_level.item()))
    
        if iteration % 1 == 0:
            error_train_net1.append(train_loss_net1.item())
            error_train_net2.append(train_loss_net2.item())
            error_test_net1.append(test_loss_net1.item())
            error_test_net2.append(test_loss_net2.item())
            
        iteration += 1
    print('--------------------------------------------------------------')
    
    return error_train_net1, error_test_net1, error_train_net2, error_test_net2

def fit_consistency(v1, model1, v2, model2, maxiter, noise_level, lr, P1, P2, P3, Y, X):
    # Step 2: Ensure that the output of the networks is consistent with one another
    print("Consistency fitting...")
    error_train_net1 = []
    error_test_net1 = []
    error_train_net2 = []
    error_test_net2 = []
    optimizer_w1 = torch.optim.Adam(model1.parameters(), lr)
    optimizer_w2 = torch.optim.Adam(model2.parameters(), lr)
    iteration = 0
    while iteration < maxiter:
    
        # Prediction & pairwise distance matrices
        Y1_pred = model1(v1)
        Y2_pred = model2(v2)
        C1 = pairwise_distance_matrix(Y1_pred, Y).double()
        C2 = pairwise_distance_matrix(Y2_pred, Y).double()    
        C3 = pairwise_distance_matrix(Y1_pred, Y2_pred).double()
        C1x = pairwise_distance_matrix(Y1_pred, X).double()
        C2x = pairwise_distance_matrix(Y2_pred, X).double()   
                
        # Losses
        train_loss_net1 = torch.mul(P1, C1).sum().sum()
        train_loss_net2 = torch.mul(P2, C2).sum().sum()
        train_consistency = torch.mul(P3, C3).sum().sum()
        test_loss_net1 = torch.mul(P1, C1x).sum().sum() 
        test_loss_net2 = torch.mul(P2, C2x).sum().sum()  
        
        # Optimization
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        train_consistency.backward()
        optimizer_w1.step()
        optimizer_w2.step()
        
        if iteration % 500 == 0:
            print("Iteration:" + str(iteration) + ", Training loss (net 1):" + str(train_loss_net1.item())
                  + ", Test loss (net 1):" + str(test_loss_net1.item()))
            print("Iteration:" + str(iteration) + ", Training loss (net 2):" + str(train_loss_net2.item())
                  + ", Test loss (net 2):" + str(test_loss_net2.item()))
            print("Iteration:" + str(iteration) + ", Consistency loss:" + str(train_consistency.item()))    
            print("Noise level:" + str(noise_level.item()))
        
        error_train_net1.append(train_loss_net1.item())
        error_train_net2.append(train_loss_net2.item())    
        error_test_net1.append(test_loss_net1.item())
        error_test_net2.append(test_loss_net2.item())
    
        iteration += 1
    print('--------------------------------------------------------------')
    
    return error_train_net1, error_train_net2, error_test_net1, error_test_net2