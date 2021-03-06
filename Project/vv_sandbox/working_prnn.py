#!/usr/bin/env python
# coding: utf-8

# March 23, 2019
# 
# Chong coded this RNN, which works. Today, we will be doing the following
# * Raising the learning rate
#   * Rate increased by using (1) LSTM,and (2) 
# * Have the RNN complete a classification task with at least 30% accuracy
# * Speculate on a "smarter" RNN structure
# 
# Hold-over from March 21, 2019
# * encode `tanh` activation functions and learn weights for the functions in the partitioned and unpartitioned models
# * Try out `dropout`, $L_2$, and `weight decay` regularization methods for the partitioned and unpartitioned models  
# 
# Next steps
# * Permuted Sequential MNIST (scramble images and train the RNN one pixel at a time)
# * Regularization for the weights

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np


# In[ ]:


# BATCH_SIZE = 128

# # generate permutation control
# # suggested from https://discuss.pytorch.org/t/permutate-mnist-help-needed/22901 
# rng_permute = np.random.RandomState(92916) # set random number generator
# idx_permute = torch.from_numpy(rng_permute.permutation(784)) # creates a tensor from a numpy array # create a random ordering of 28x28 = 784 pixels (i.e. generate a random image)
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.1307,), (0.3081,)),
#      transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) )]) # generate a transform such that any image put through the transformation will be "scrambled"


# # download and load training dataset
# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=2)

# # download and load testing dataset
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=2)

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


# In[ ]:


# build the RNN
# class PRNN(nn.Module):
#     def __init__(self, n_inputs, n_hidden,n_output,mr):
#         super(PRNN, self).__init__()
#         self.n_inputs = n_inputs
#         self.n_hidden = n_hidden
#         self.n_output = n_output
#         self.mr = mr
#         self.encoder = nn.Linear(n_inputs,n_hidden)
#         self.recurrent = nn.Linear(n_hidden,n_hidden)
#         self.decoder = nn.Linear(n_hidden, n_output)
#         # self.lstm = nn.LSTM(n_inputs, n_output, batch_first=True)
        
#     def forward(self, x0):
#         T = int(x0.shape[2]/2)
#         #self.h1 = Variable(torch.zeros(self.n_hidden))
#         self.h1 = Variable(torch.zeros(x0.size(0), self.n_hidden))
#         #for t in range(T):
#            #self.h1 = self.mr*self.h1+(1-self.mr)*torch.relu(self.encoder(x0[:,t+7,:])+self.recurrent(self.h1))
#            # self.h1,_ = self.lstm(x0[:,:,:],self.h1)
#         self.y1 = self.decoder(self.h1)
        
#         return self.y1

class PRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,mr):
        super(PRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.mr = mr
        self.encoder = nn.Linear(n_inputs,n_hidden)
        self.recurrent = nn.Linear(n_hidden,n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        self.rnn = nn.RNN(n_inputs, n_hidden)
        
    def forward(self, x0):
        x0=x0.permute(1,0,2)
        self.h1 = torch.zeros(1,BATCH_SIZE,self.n_hidden)
        #self.h1 = Variable(torch.zeros(x0.size(0), self.n_hidden))
        #for t in range(T):
           #self.h1 = self.mr*self.h1+(1-self.mr)*torch.relu(self.encoder(x0[:,t+7,:])+self.recurrent(self.h1))
        self.y0, self.h1 = self.rnn(x0,self.h1)
        self.y1 = self.decoder(self.h1[0])
        
        return self.y1
# can I modify this to set an arbitrary number of layers?
        
# RNN LSTM  
class PRNN_LSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,mr):
        super(PRNN_LSTM, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.mr = mr
#         self.encoder = nn.Linear(n_inputs,n_hidden)
#         self.recurrent = nn.Linear(n_hidden,n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        self.lstm = nn.LSTM(n_inputs, n_hidden)
        
    def forward(self, x0):
        x0=x0.permute(1,0,2)
        self.h1 = (torch.zeros(1,BATCH_SIZE,self.n_hidden),torch.zeros(1,BATCH_SIZE,self.n_hidden))
        #self.h1 = Variable(torch.zeros(x0.size(0), self.n_hidden))
        #for t in range(T):
           #self.h1 = self.mr*self.h1+(1-self.mr)*torch.relu(self.encoder(x0[:,t+7,:])+self.recurrent(self.h1))
        self.y0, self.h1 = self.lstm(x0,self.h1)
        self.y1 = self.decoder(self.h1[0])
        
        return self.y1
# can I modify this to set an arbitrary number of layers?

# accuracy and one-hot encoding functions
def onehotTensor(category,n_categories):
    tensor = torch.zeros(1, n_categories,dtype=torch.long)
    tensor[0][category] = 1
    return tensor
        
def get_accuracy(logit, target, batch_size):
#     corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data)#.sum()
#     print(corrects)
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


# In[ ]:


# run RNN-LSTM
# parameters 
N_STEPS = 28
N_INPUTS = 28
N_HIDDEN = 100
N_OUTPUTS = 10
N_EPHOCS = 10

model = PRNN_LSTM(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)
# model = MyModel()
# model.cuda()


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
        
        # print shapes
#         print("shape of the label:"),
#         print(y_tar.shape)
#         print("shape of the machine-generated label"),
#         print(y_pred.shape)
        
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE)
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


plt.subplot(1,2,1)
plt.plot(Phist[::1000,:].T)
plt.subplot(2,1,2)
plt.plot(train_running_loss)
plt.show()


# In[ ]:


# run Vanilla RNN
# parameters 
N_STEPS = 28
N_INPUTS = 28
N_HIDDEN = 100
N_OUTPUTS = 10
N_EPHOCS = 10

model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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


plt.subplot(1,2,1)
plt.plot(Phist[::1000,:].T)
plt.subplot(1,2,2)
plt.plot(train_running_loss)
plt.show()


# In[ ]:


# # weights
# l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
# for W in mdl.parameters():
#   l2_reg = l2_reg + W.norm(2)
# batch_loss = (1/N_train)*(y_pred - batch_ys).pow(2).sum() + l2_reg * reg_lambda
# batch_loss.backward()


# 
# The trainloader consists of a series of images. We need to funnel each image one pixel at a time as opposed to one row at a time. 
# 

# In[ ]:


# Sparse Sequential MNIST -- feed in the dataset one pixel at a time and every other pixel. 
# Use many different neurons in the hidden layer as well, since the dynamic range of out

# Round 1: feed in the sparser image row by row. 
# parameters 
N_STEPS_SPARSE = 14 # since we fed in every other entry, we now have a 14x14 matrix 
N_INPUTS = 14 #14 # since the model has to take in one value -- i.e. one pixel -- at a time # and subsequently, 1 input at a time
N_HIDDEN = 400 # let's raise the number of hidden neurons
N_OUTPUTS = 50 # 50 output values
N_EPHOCS = 30

model_1 = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1) # vanilla RNN

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
#         x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x)
#         print(x[0]) # 100 images at a time
        optimizer.zero_grad() # set the gradients for the optimizer function at 0
        x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x.shape)
        x = x[:,:,::2,::2]
        x = x.contiguous()
        # print(x.shape)
        # print(x.view)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS_SPARSE,N_INPUTS)) # make the prediction
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar) # calculate the loss
        loss.backward() # backpropogate the loss
        optimizer.step() # set the next step in the weights using the optimization function
        running_loss+=loss.item() # track the running loss
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE) # compute the accuracy of the prediction
    train_running_loss[epoch] = running_loss # compute this epoch's losss
    train_acc[epoch] = running_acc/i # compute this epoch's accuracy
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# Sequential MNIST -- feed in the dataset pixel by pixel
# parameters 
N_STEPS = 28 # NSTEPs is the number of traversals through each datapoint that the RNN must make
N_STEPS_SPARSE = 1# feed the vector in all at once rather than one at a time 
N_INPUTS = 14*14#feed the vector in at once. No real recurrent property
N_HIDDEN = 400 # let's raise the number of hidden neurons
N_OUTPUTS = 50 # 50 output values
N_EPHOCS = 30

model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
#         x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x)
#         print(x[0]) # 100 images at a time
        optimizer.zero_grad() # set the gradients for the optimizer function at 0
        x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x.shape)
        x = x[:,:,::2,::2]
        x = x.contiguous()
        # print(x.shape)
        # print(x.view)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS_SPARSE,N_INPUTS)) # make the prediction
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar) # calculate the loss
        loss.backward() # backpropogate the loss
        optimizer.step() # set the next step in the weights using the optimization function
        running_loss+=loss.item() # track the running loss
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE) # compute the accuracy of the prediction
    train_running_loss[epoch] = running_loss # compute this epoch's losss
    train_acc[epoch] = running_acc/i # compute this epoch's accuracy
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# Sequential MNIST -- Feed in the data four pixels at a time
# parameters 
N_STEPS = 28 # NSTEPs is the number of traversals through each datapoint that the RNN must make
N_STEPS_SPARSE = 7*7# feed the vector in all at once rather than one at a time 
N_INPUTS = 4#feed the image in four pixels at a time. No real recurrent property
N_HIDDEN = 400 # let's raise the number of hidden neurons
N_OUTPUTS = 50 # 50 output values
N_EPHOCS = 30

model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
#         x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x)
#         print(x[0]) # 100 images at a time
        optimizer.zero_grad() # set the gradients for the optimizer function at 0
        x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x.shape)
        x = x[:,:,::2,::2]
        x = x.contiguous()
        # print(x.shape)
        # print(x.view)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS_SPARSE,N_INPUTS)) # make the prediction
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar) # calculate the loss
        loss.backward() # backpropogate the loss
        optimizer.step() # set the next step in the weights using the optimization function
        running_loss+=loss.item() # track the running loss
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE) # compute the accuracy of the prediction
    train_running_loss[epoch] = running_loss # compute this epoch's losss
    train_acc[epoch] = running_acc/i # compute this epoch's accuracy
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# Sequential MNIST -- feed in the dataset pixel by pixel
# parameters 
N_STEPS = 28 # NSTEPs is the number of traversals through each datapoint that the RNN must make
N_STEPS_SPARSE = 14*14# feed the vector in all at once rather than one at a time 
N_INPUTS = 1#feed the vector in at once. No real recurrent property
N_HIDDEN = 400 # let's raise the number of hidden neurons
N_OUTPUTS = 50 # 50 output values
N_EPHOCS = 40

model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
#         x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x)
#         print(x[0]) # 100 images at a time
        optimizer.zero_grad() # set the gradients for the optimizer function at 0
        x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x.shape)
        x = x[:,:,::2,::2]
        x = x.contiguous()
        # print(x.shape)
        # print(x.view)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS_SPARSE,N_INPUTS)) # make the prediction
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar) # calculate the loss
        loss.backward() # backpropogate the loss
        optimizer.step() # set the next step in the weights using the optimization function
        running_loss+=loss.item() # track the running loss
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE) # compute the accuracy of the prediction
    train_running_loss[epoch] = running_loss # compute this epoch's losss
    train_acc[epoch] = running_acc/i # compute this epoch's accuracy
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


## Sequential-Sparse MNIST with RNN-LSTM
# parameters: same task as the 4-pixels-at-a-time task
N_STEPS_SPARSE = 7*7 # the number of traversals through each dataset that the RNN must make 
N_INPUTS = 4 # feed the image in four pixels at a time. No real recurrent property
N_HIDDEN = 400 # let's raise the number of hidden neurons
N_OUTPUTS = 50 # 50 output values
N_EPHOCS = 30

model = PRNN_LSTM(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
#         x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x)
#         print(x[0]) # 100 images at a time
        optimizer.zero_grad() # set the gradients for the optimizer function at 0
        x, y_tar = data # get the data: x is the string of pixels, and y is the class
#        print(x.shape)
        x = x[:,:,::2,::2]
        x = x.contiguous()
        # print(x.shape)
        # print(x.view)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS_SPARSE,N_INPUTS)) # make the prediction
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar) # calculate the loss
        loss.backward() # backpropogate the loss
        optimizer.step() # set the next step in the weights using the optimization function
        running_loss+=loss.item() # track the running loss
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE) # compute the accuracy of the prediction
    train_running_loss[epoch] = running_loss # compute this epoch's losss
    train_acc[epoch] = running_acc/i # compute this epoch's accuracy
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


for data in trainloader:
  x, y_tar = data
  print(x.shape)
  break


# The training takes too long. We will feed in every other image. 

# # OLD

# In[ ]:





# In[ ]:


# # parameters 
# N_STEPS = 28
# N_INPUTS = 28
# N_HIDDEN = 128
# N_OUTPUTS = 10
# N_EPHOCS = 10

# model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
# criterion = nn.CrossEntropyLoss()

# train_running_loss = np.zeros(N_EPHOCS)
# train_acc = np.zeros(N_EPHOCS)
# nparams = 0
# for param in model.parameters(): 
#   if param.requires_grad:
#     nparams += param.data.numpy().size
# Phist = np.zeros((nparams,N_EPHOCS))

# for epoch in range(N_EPHOCS):
#     nps = 0
#     running_loss=0
#     running_acc=0
#     for p,param in enumerate(model.parameters()):
#         if param.requires_grad:
#             plist = param.data.numpy().flatten()
#             for j in range(plist.size):
#                 Phist[nps,epoch]=plist[j]
#                 nps+=1
  
#     for i, data in enumerate(trainloader):
        
#         optimizer.zero_grad()
#         x, y_tar = data
#         y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
#         loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE)
#     train_running_loss[epoch] = running_loss
#     train_acc[epoch] = running_acc/i
    
#     print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# plt.plot(Phist[::1000,:].T)
# plt.show()

# plt.plot(train_running_loss)
# plt.show()


# In[ ]:


# l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
# for W in mdl.parameters():
#   l2_reg = l2_reg + W.norm(2)
# batch_loss = (1/N_train)*(y_pred - batch_ys).pow(2).sum() + l2_reg * reg_lambda
# batch_loss.backward()


# Strategies we can use to improve the learning rate for a Vanilla RNN
# * Better gradient descent methods (best not to play with )
# * Online optimization methods--can we get the network to try out different orders of the data ("training regimes"), find which training regime produces the fastest rate, and determine 
#   * Read the curriculum learning paper: https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf 
# 
# 
# More 
# * play with initialization of parameters
# * spit out the hidden state of the RNN over time
# * Dropout for regularization
# * add more hidden layers
# 
# 
# Let's play with the above:
# 1. Try different parameter initializations
# 2. Monitor the hidden state of the RNN over time and monitor the backpropogation values
# 3. Add more hidden layers
# 4. Initial curriculum learning simulations
# 5. Try out the partitioning of a hidden layer into separate memory units. 

# In[ ]:


# # Add more hidden layers -- parameterize the number of hidden layers

# # build the RNN
# class PRNN2(nn.Module):
#     def __init__(self, n_inputs, n_hidden,n_output,mr, n_hidden_layers):
#         super(PRNN, self).__init__()
#         self.n_inputs = n_inputs
#         self.n_hidden = n_hidden # number of hidden neurons
#         self.n_hidden_layers = n_hidden_layers
#         self.n_output = n_output
#         self.mr = mr
#         self.encoder = nn.Linear(n_inputs,n_hidden)
#         # add a function here that will 
#         self.recurrent = nn.Linear(n_hidden,n_hidden)
#         self.decoder = nn.Linear(n_hidden, n_output)
#         # self.lstm = nn.LSTM(n_inputs, n_output, batch_first=True)
        
#     def forward(self, x0):
#         T = int(x0.shape[2]/2)
#         #self.h1 = Variable(torch.zeros(self.n_hidden))
#         self.h1 = Variable(torch.zeros(x0.size(0), self.n_hidden))
#         #for t in range(T):
#            #self.h1 = self.mr*self.h1+(1-self.mr)*torch.relu(self.encoder(x0[:,t+7,:])+self.recurrent(self.h1))
#            # self.h1,_ = self.lstm(x0[:,:,:],self.h1)
#         self.y1 = self.decoder(self.h1)
        
#         return self.y1


# # RNN with LSTM unit

# In[ ]:


# class PRNN(nn.Module):
#     def __init__(self, n_inputs, n_hidden,n_output,mr):
#         super(PRNN, self).__init__()
#         self.n_inputs = n_inputs
#         self.n_hidden = n_hidden
#         self.n_output = n_output
#         self.mr = mr
#         self.encoder = nn.Linear(n_inputs,n_hidden)
#         self.recurrent = nn.Linear(n_hidden,n_hidden)
#         self.decoder = nn.Linear(n_hidden, n_output)
#         self.lstm = nn.LSTM(n_inputs, n_hidden)
        
#     def forward(self, x0):
#         x0=x0.permute(1,0,2)
#         self.h1 = (torch.zeros(1,BATCH_SIZE,self.n_hidden),torch.zeros(1,BATCH_SIZE,self.n_hidden))
#         self.y0, self.h1 = self.lstm(x0,self.h1)
#         self.y1 = self.decoder(self.h1[0])
        
#         return self.y1
      
# def get_accuracy(logit, target, batch_size):
#     corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
#     accuracy = 100.0 * corrects/batch_size
#     return accuracy.item()


# In[ ]:




# # parameters 
# N_STEPS = 28
# N_INPUTS = 28
# N_HIDDEN = 100
# N_OUTPUTS = 10
# N_EPHOCS = 10

# model = PRNN_LSTM(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
# criterion = nn.CrossEntropyLoss()

# train_running_loss = np.zeros(N_EPHOCS)
# train_acc = np.zeros(N_EPHOCS)
# nparams = 0
# for param in model.parameters(): 
#   if param.requires_grad:
#     nparams += param.data.numpy().size
# Phist = np.zeros((nparams,N_EPHOCS))

# for epoch in range(N_EPHOCS):
#     nps = 0
#     running_loss=0
#     running_acc=0
#     for p,param in enumerate(model.parameters()):
#         if param.requires_grad:
#             plist = param.data.numpy().flatten()
#             for j in range(plist.size):
#                 Phist[nps,epoch]=plist[j]
#                 nps+=1
  
#     for i, data in enumerate(trainloader):
        
#         optimizer.zero_grad()
#         x, y_tar = data
#         y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
#         loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE)
#     train_running_loss[epoch] = running_loss
#     train_acc[epoch] = running_acc/i
    
#     print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# # test the trained RNN-LSTM on the testloader data
# for epoch in range(N_EPHOCS):
#     nps = 0
#     running_loss=0
#     running_acc=0
#     for p,param in enumerate(model.parameters()):
#         if param.requires_grad:
#             plist = param.data.numpy().flatten()
#             for j in range(plist.size):
#                 Phist[nps,epoch]=plist[j]
#                 nps+=1
  
#     for i, data, j, tdata in enumerate(testloader):
        
#         optimizer.zero_grad()
#         x, y_tar = data
#         y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
#         loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),y_tar)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS), y_tar, BATCH_SIZE)
#     train_running_loss[epoch] = running_loss
#     train_acc[epoch] = running_acc/i
    
#     print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))


# In[ ]:


# plt.plot(Phist[::1000,:].T)
# plt.show()


# In[ ]:


# l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
# for W in mdl.parameters():
#   l2_reg = l2_reg + W.norm(2)
# batch_loss = (1/N_train)*(y_pred - batch_ys).pow(2).sum() + l2_reg * reg_lambda
# batch_loss.backward()

