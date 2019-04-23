#!/usr/bin/env python
# coding: utf-8

# # Orchestra wrapper for NN experiments
# This orchestra wrapper:
# * Takes model parameters as input
# * Takes a dataset as input (in this case, the MNIST dataset is default
<<<<<<< HEAD
# * Sends multiple instances of the regularization  
=======
>>>>>>> c769c2b03242f9415ba8d337db2d589252c2b1bc

# In[ ]:


<<<<<<< HEAD
import os,sys,re
from tqdm import tnrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


def input_params ():
    trainloader,testloader,BATCH_SIZE = load_mnist()
    infa = sys.argv[1:] # infasta
    
    # parameters 
    N_STEPS = 28
    N_INPUTS = 28
    N_HIDDEN = 112
    N_OUTPUTS = 10
    N_EPHOCS = 11
    N_REPS = 15
    N_PARAMS = nparam(N_INPUTS,N_HIDDEN,N_OUTPUTS)

    lambdas = np.arange(0,5.5,0.5,dtype=np.float)
    N_LAMBDA = len(lambdas)
    gidx = int(N_HIDDEN/2)


# In[ ]:


def load_mnist ():
    BATCH_SIZE = 1000

    # list all transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    # download and load training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    # download and load testing dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)
    
    return(trainloader,testloader,BATCH_SIZE)


# In[ ]:


def run_model ():
=======
# pseudocode for main
def main ():
>>>>>>> c769c2b03242f9415ba8d337db2d589252c2b1bc
    return()

