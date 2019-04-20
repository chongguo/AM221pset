#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook will test the performance of a 2-layer MLP (multi-layer perceptron) in performing a character-wise prediction task on 
# * MNIST
# * the play *Hamlet*. 

# In[1]:


# imports
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

# alphabet
import string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # MLP code

# ## Data preparation

# In[2]:


def onehotTensor(category,n_categories):
    tensor = torch.zeros(1, n_categories,dtype=torch.long)
    tensor[0][category] = 1
    return tensor
        
def get_accuracy(logit, target):
    batch_size = len(target)
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def nparam(ninputs,nhidden,noutputs):
    return ninputs*(nhidden+1) + nhidden*(nhidden+1)+nhidden*(noutputs+1)


# ## Model

# In[22]:


# a prototype MLP
class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_neurons, n_hidden_layers, n_output, dt, device):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs # set the number of neurons in the input layer
        self.n_hidden_neurons = n_hidden_neurons # how many neurons are in each hidden layer
        self.n_hidden_layers = n_hidden_layers # how many hidden layers are we using in our model
        self.n_output = n_output # set the number of neurons in the output layer
        self.dt = nn.Parameter(torch.Tensor([dt])) # set the change in parameter update
        self.a = nn.Parameter(torch.Tensor([1])) # set the bias
        self.sig = nn.Sigmoid() # set the activation function 
        self.n_hidden = n_hidden_neurons
        self.decoder = nn.Linear(n_hidden_neurons, n_output) # decode output
        self.encoder = nn.Linear(n_inputs, n_hidden_neurons) # encode input
        
    # the main operation for the MLP is to update each hidden layer with the state of the previous hidden layer
    # so, if you need to update the hidden layers, make sure you update each layer with state of the previous layer
    
    def update_hidden_layer(self):
        # update the neurons in the current hidden layer with the state of the inputted "previous" layer
        for i in range(1,self.n_hidden_layers):
            # update each node in the h1_current
            self.h1 = self.hidden
#             self.hidden = nn.Linear(self.h1,self.hidden) # update the stored hidden layer state as many times as there are hidden layers specified # BUG
            # the nn.Linear should take in the shape...
            self.hidden = nn.Linear(self.n_hidden_neurons,self.n_hidden_neurons) 
        self.h1 = self.hidden
        return(self.h1)
    
    def forward(self, x0):
        x0=x0.permute(1,0,2) # permute the tensor
        # save the hidden layers as a tensor
        
        # initialize self.h1
        self.hidden = torch.zeros(self.n_hidden_layers,BATCH_SIZE,self.n_hidden).to(device) # initialize the hidden layer
        
        # set the hidden layer value
        for i in range(x0.size(0)):
            self.hidden[0,:,:] = self.sig(self.dt)*self.a*torch.tanh(self.encoder(x0[i,:,:]))
        
#         self.hidden = torch.zeros(self.n_hidden_layers,BATCH_SIZE,self.n_hidden).to(device) # initialize the hidden layer
        # the first layer will consist of the encoded inputs
#         for i in range(x0.size(0)):
#             self.hlayers[0,:,:] = self.sig(self.dt)*self.a*torch.tanh(self.encoder(x0[i,:,:])) # (1-self.sig(self.dt))*self.h1+self.sig(self.dt)*self.a*torch.tanh(self.encoder(x0[i,:,:])+self.recurrent(self.h1))
#         # update the additional hidden layers
        self.hidden = self.update_hidden_layer()

        self.y1 = self.decoder(self.n_hidden_neurons,self.n_output)
        return self.y1 


# # Load data

# In[12]:


# MNIST

BATCH_SIZE = 100

# list all transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


# # Train on MNIST

# In[23]:


# training on MNIST
# parameters 
N_STEPS = 28
N_INPUTS = 28
N_HIDDEN_NEURONS = 100
N_HIDDEN_LAYERS = 2
N_OUTPUTS = 10
N_EPHOCS = 10

model = MLP(N_INPUTS,N_HIDDEN_NEURONS,N_HIDDEN_LAYERS,N_OUTPUTS,0.1,device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_running_loss = np.zeros(N_EPHOCS)
train_acc = np.zeros(N_EPHOCS)
nparams = 0
for param in model.parameters(): 
  if param.requires_grad:
    nparams += param.data.numpy().size
Phist = np.zeros((nparams,N_EPHOCS))

for epoch in range(N_EPHOCS):
    nps = 0
    running_loss=0
    running_acc=0
    for p,param in enumerate(model.parameters()):
        if param.requires_grad:
            plist = param.data.numpy().flatten()
            for j in range(plist.size):
                Phist[nps,epoch]=plist[j]
                nps+=1
  
    for i, data in enumerate(trainloader):
        
        optimizer.zero_grad()
        x, y_tar = data
        y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar)
                
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE)
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:




