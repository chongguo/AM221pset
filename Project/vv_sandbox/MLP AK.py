#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # imports
# from tqdm import tnrange
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import numpy as np
# import gc
# import os
# %matplotlib inline  
# from sklearn.model_selection import train_test_split

# # alphabet
# import string

# # date and time
# import datetime

# # set the device on which to run the experiment
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


# imports
from tqdm import tnrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import gc
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime

# alphabet
import string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


# functions
def get_accuracy(logit, target):
    batch_size = len(target)
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def nparam(ninputs,nhidden,noutputs):
    return ninputs*(nhidden+1) + nhidden*(nhidden+1)+nhidden*(noutputs+1)

# define the nnumber of parameters we need
def nparam_MLP(N_INPUTS,N_HIDDEN,N_OUTPUTS):
    input_to_hidden1 = (N_INPUTS+1)*N_HIDDEN #+1 for bias
    hidden1_to_hidden2 = (N_HIDDEN + 1)*N_HIDDEN
    hidden2_to_output = (N_OUTPUTS)*(N_HIDDEN+1)
    return(sum([input_to_hidden1,hidden1_to_hidden2,hidden2_to_output]))


# In[3]:


# a prototype 2-layer MLP
class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_neurons, n_output,  device):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs # set the number of neurons in the input layer
        self.n_hidden_neurons = n_hidden_neurons # how many neurons are in each hidden layer
        self.n_output = n_output # set the number of neurons in the output layer
        self.sig = nn.Sigmoid() # set the activation function 
        self.tanh = nn.Tanh()
        self.n_hidden = n_hidden_neurons
        self.encoder = nn.Linear(n_inputs, n_hidden_neurons) # encode input
        self.recurrent = nn.Linear(n_hidden_neurons,n_hidden_neurons) # recurrent connections
        self.decoder = nn.Linear(n_hidden_neurons, n_output) # decode output
                
    def forward(self, x):
        self.hidden1 = self.tanh(self.encoder(x))
        self.hidden2 = self.tanh(self.recurrent(self.hidden1))
        self.output = self.decoder(self.hidden2)
        return self.output
    


# In[4]:


# # Test MLP on Anna Karenina
# # Load Anna Karenina
# from torch.utils.data import DataLoader # dataloader 
# import sys
# sys.path.insert(0,'../final_project/Data/')
# # from AnnaDataset import AnnaDataset, InvertAnna # import AK dataset; has 1/100th of values
# from AnnaDataset_MLP import AnnaDataset, InvertAnna # import AK dataset
# import torchvision
# import torchvision.transforms as transforms

# # params
# BATCH_SIZE = 500 # how many batches we are running
# N_STEPS = 10 # How many characters are we inputting into the list at a time
# N_HIDDEN_NEURONS = 512 # how many neurons per hidden layer
# # N_LAYERS = 2 # 2 hidden layers
# N_EPOCHS = 20 # how many training epocs
# learning_rates = np.asarray([2]) # learning rates
# N_REPS = 3 # len(learning_rates) # the number of learning repetitions
# gidx = int(N_HIDDEN_NEURONS/2) # partition of the hidden neurons for block regularization

# # regularization parameters
# # lambdas = np.arange(0,1e-2,3e-3,dtype=np.float)
# # lambdas = 10**np.arange(-5,5,1,dtype=np.float) # order-of-magnitude sweep
# lambdas = np.arange(0,1e-1,1e-2,dtype=np.float) # full sweep
# N_LAMBDA = len(lambdas)

# # load data
# # list all transformations
# transform = transforms.Compose(
#     [transforms.Normalize((0,), (0.3,))])

# dataset = AnnaDataset(N_STEPS) # load the dataset

# N_INPUTS = len(dataset.categories)*N_STEPS
# N_OUTPUTS = len(dataset.categories)

# N_PARAMS = nparam_MLP(N_INPUTS,N_HIDDEN_NEURONS,N_OUTPUTS)

# Test MLP on Anna Karenina
# Load Anna Karenina
from torch.utils.data import DataLoader # dataloader 
import sys
sys.path.insert(0,'../final_project/Data/')
from AnnaDataset_MLP import AnnaDataset, InvertAnna # import AK dataset
# from AnnaDataset import AnnaDataset, InvertAnna # import AK dataset
import torchvision
import torchvision.transforms as transforms

# params
BATCH_SIZE = 500 # how many batches we are running
N_STEPS = 10 # How many characters are we inputting into the list at a time
N_HIDDEN_NEURONS = 512 # how many neurons per hidden layer
N_LAYERS = 2 # 2 hidden layers
N_EPOCHS = 15 # how many training epocs
learning_rates = np.asarray([2]) # learning rates
N_REPS = 1 # len(learning_rates) # the number of learning repetitions
gidx = int(N_HIDDEN_NEURONS/2)

# regularization parameters
# lambdas = np.arange(0,1e-2,3e-3,dtype=np.float)
# lambdas = np.arange(0,1e-1,1e-2,dtype=np.float) # full sweep
lambdas = np.arange(0,1,1e-1) # short sweep
print(lambdas)
# lambdas = np.arange(0,1,1e-1,dtype=np.float) # full sweep
N_LAMBDA = len(lambdas)

# load data
# list all transformations
transform = transforms.Compose(
    [transforms.Normalize((0,), (0.3,))])

dataset = AnnaDataset(N_STEPS) # load the dataset

N_INPUTS = len(dataset.categories)*N_STEPS
N_OUTPUTS = len(dataset.categories)
N_PARAMS = nparam_MLP(N_INPUTS,N_HIDDEN_NEURONS,N_OUTPUTS)


# trainloader = DataLoader(dataset, batch_size=BATCH_SIZE,
#                         shuffle=False, num_workers=4) # create a DataLoader. We want a batch of BATCH_SIZE entries
# testloader = DataLoader(dataset, batch_size=BATCH_SIZE,
#                         shuffle=False, num_workers=4) # create a DataLoader. We want a batch of BATCH_SIZE entries


# In[5]:


# # # test_split = torch.split(dataset.onehot_encoded,20,dim=0)
# # # print(len(test_split))
# # print(dataset.onehot_encoded.shape)
# # print(test_split[0].shape)
# # print(test_split[1].shape)

# # train-test-split
# train_fraction,test_fraction,valid_fraction = (0.8,0.1,0.1)
# random_seed = 0
# shuffle_dataset = True

# # from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets 
# # Creating data indices for training and validation splits:
# dataset_size = len(dataset.onehot_encoded)
# indices = list(range(dataset_size))
# train_split = int(np.floor(train_fraction * dataset_size))
# print(train_split)
# valid_split = train_split + int(np.floor(valid_fraction * dataset_size))
# test_split = valid_split + int(np.floor(test_fraction * dataset_size))

# print(train_split,valid_split,test_split,dataset_size)

# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# # train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:valid_split], indices[valid_split:]
# train_indices = indices[:train_split]
# val_indices = indices[train_split:valid_split]
# test_indices = indices[valid_split:]

# print(max(train_indices))
# print(max(val_indices))
# print(max(test_indices))

# # Creating PT data samplers and loaders:
# train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
# test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
# valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
# #                                            sampler=train_sampler)
# # validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
# #                                                 sampler=valid_sampler)



# train-test-split
train_fraction = 0.8

random_seed = 0
shuffle_dataset = True

# from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets 
# Creating data indices for training and validation splits:
dataset_size = len(dataset.onehot_encoded)
indices = list(range(dataset_size))
train_split = int(np.floor(train_fraction * dataset_size))
print(train_split)


if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices = indices[:train_split]
test_indices = indices[train_split:]

print(len(train_indices))
print(len(test_indices))

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SequentialSampler(train_indices)
test_sampler = torch.utils.data.SequentialSampler(test_indices)


# In[6]:


# train-test-validate split

trainloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4,
                        sampler = train_sampler) # create a DataLoader. We want a batch of BATCH_SIZE entries
# validloader = DataLoader(dataset, batch_size=BATCH_SIZE,
#                         sampler = valid_sampler) # create a DataLoader. We want a batch of BATCH_SIZE entries
testloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4,
                        sampler = test_sampler) # create a DataLoader. We want a batch of BATCH_SIZE entries


# In[7]:


# modelkey = ''.join([str(np.random.randint(0,9)) for i in range(10)])

runnow = datetime.datetime.now()
modelkey = str(runnow.isoformat())
print(modelkey)


# In[8]:


# # regularizing digonal blocks of the partitioned RNN
# # initialize arrays of loss values and weights over the number of epohcs, the number of lambdas we are testing, and the number of reps. 
# train_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS)) 
# train_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# test_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# test_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# Phist_P = np.zeros((N_PARAMS,N_EPOCHS,N_LAMBDA,N_REPS))

# model_P = [None]*N_LAMBDA*N_REPS # array of models
# regval_P = [] # array of regularization values

# # generate a unique key for these models

# for r in tnrange(N_REPS): # loop over the number of reps
#     for k in tnrange(N_LAMBDA): # loop over the number of different lambda values
#         reg_lambda = lambdas[k] # set the regularization lambda
#         model_path = './models/model'+str(modelkey)+'_P_rep_{}_lambda_{:d}_10.pt'.format(r,int(reg_lambda*10)) # path to which we will save the model
#         model_P[k+r*N_LAMBDA] = MLP(N_INPUTS,N_HIDDEN_NEURONS,N_OUTPUTS,device).to(device) # create the model
#         l2_reg = torch.tensor(1,device=device) # create the l2 regularization value tensor
# #        optimizer = torch.optim.SGD(model_P[k+r*N_LAMBDA].parameters(), lr=1e-1, momentum=0.9) # set the function for SGD
#         optimizer = torch.optim.SGD(model_P[k+r*N_LAMBDA].parameters(), lr=1e-2, momentum=0.9) # set the function for SGD
#         criterion = nn.CrossEntropyLoss() # set the loss function
        
#         # note that cross-entropy loss expects the indices of the class, not the one-hot. So, for A = [1,0,0,...] and B = [0,1,0,...], A is 0 and B is 1
        
#         for epoch in range(N_EPOCHS): # for each training epoch
#             nps = 0
#             running_train_loss=0
#             running_train_acc=0
#             model_P[k+r*N_LAMBDA].train() 
#             for p, param in enumerate(model_P[k+r*N_LAMBDA].parameters()): # go through all the model parameters
#                 if param.requires_grad:
#                     plist = torch.flatten(param.data) # set the list of parameters
#                     for j in range(plist.size(0)):
#                         while nps < Phist_P.shape[0]:
#                             Phist_P[nps,epoch,k,r]=plist[j].item() # update the parameters
#                             nps+=1

#             for i, (x, y_tar) in enumerate(trainloader):
#                 # print(i)
#                 l2_reg = 0
#                 x, y_tar = x.to(device), y_tar.to(device) # x is the training set, y_tar is the output label
#                 x = x-0.3
#                 optimizer.zero_grad() # set gradients to 0
#                 y_pred = model_P[k+r*N_LAMBDA](x.view(x.shape[0],x.shape[1]*x.shape[2])) # compute the prediction. 
#                 loss = criterion(y_pred,y_tar) 
#                 for p,param in enumerate(model_P[k+r*N_LAMBDA].parameters()):
#                     if param.requires_grad and len(param.shape)==2:
#                         if param.shape[0]==N_HIDDEN_NEURONS and param.shape[1]==N_HIDDEN_NEURONS:
#                             l2_reg = l2_reg + param[:gidx,:gidx].norm(p=1) # update the l2 regularization constant
#                             l2_reg = l2_reg + param[gidx:,gidx:].norm(p=1)
# #                         elif param.shape[1]==N_HIDDEN_NEURONS: # regularization 
# #                             l2_reg = l2_reg + param[:,gidx:].norm(p=1)
# #                         elif param.shape[0]==N_HIDDEN_NEURONS:
# #                             l2_reg = l2_reg + param[:gidx,:].norm(p=1)
#                 regval_P.append(l2_reg.item()) # add the l2 regularization to  the running list
#                 loss = loss + l2_reg*reg_lambda/BATCH_SIZE # compute the loss
#                 loss.backward() # backpropogate the loss
#                 optimizer.step() # run SGD
#                 running_train_loss+=loss.item()
#                 running_train_acc+=get_accuracy(y_pred, y_tar) # compute accuracy
            
#             running_test_acc=0
#             running_test_loss=0
#             model_P[k+r*N_LAMBDA].eval()
#             for i,(x_test, y_test_tar) in enumerate(testloader):
#                 x_test, y_test_tar = x_test.to(device), y_test_tar.to(device)
#                 x_test = x_test - 0.3
#                 y_test_pred = model_P[k+r*N_LAMBDA](x_test.view(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
#                 loss = criterion(y_test_pred,y_test_tar)
                
#                 running_test_loss+=loss.item()
#                 running_test_acc+=get_accuracy(y_test_pred, y_test_tar)
                
#             for i,(x_valid, y_valid_tar) in enumerate(validloader):
#                 x_valid, y_valid_tar = x_test.to(device), y_test_tar.to(device)
#                 x_valid = x_valid - 0.3
#                 y_valid_pred = model_P[k+r*N_LAMBDA](x_test.view(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
#                 loss = criterion(y_test_pred,y_test_tar)
                
#                 running_valid_loss+=loss.item()
#                 running_valid_acc+=get_accuracy(y_valid_pred, y_valid_tar)

                
#             train_loss_P[epoch,k,r] = running_train_loss/len(trainloader)
#             train_acc_P[epoch,k,r] = running_train_acc/len(trainloader)
#             test_loss_P[epoch,k,r] = running_test_loss/len(testloader)
#             test_acc_P[epoch,k,r] = running_test_acc/len(testloader)
#             valid_loss_P[epoch,k,r] = running_valid_loss/len(validloader)
#             valid_acc_P[epoch,k,r] = running_valid_acc/len(validloader)

#             print("Epoch %d; rep %d; lambda %f; train accuracy %f; train loss %f; test accuracy %f; test loss %f; valid accuracy $f; valid loss %f; reg val %f"
#                   %(epoch,
#                     r,
#                     k,
#                     train_acc_P[epoch,k,r],
#                     train_loss_P[epoch,k,r],
#                     test_acc_P[epoch,k,r],
#                     valid_acc_P[epoch,k,r],
#                     valid_loss_P[epoch,k,r],
#                     test_loss_P[epoch,k,r],l2_reg.item()))

#             # print(train_acc_P[epoch,k,r])
            
#         # save the model and free the memory  
#         torch.save(model_P[k+r*N_LAMBDA].state_dict(), model_path)
#         model_P[k+r*N_LAMBDA] = [None]
#         del(l2_reg,loss,optimizer,criterion,plist,param)


# In[ ]:


# regularizing digonal blocks of the partitioned RNN
# initialize arrays of loss values and weights over the number of epohcs, the number of lambdas we are testing, and the number of reps. 
train_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS)) 
train_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
test_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
test_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
Phist_P = np.zeros((N_PARAMS,N_EPOCHS,N_LAMBDA,N_REPS))

model_P = [None]*N_LAMBDA*N_REPS # array of models
regval_P = [] # array of regularization values

lr = 1e-2
momentum = 0.9

for r in tnrange(N_REPS): # loop over the number of reps
    for k in tnrange(N_LAMBDA): # loop over the number of different lambda values
        reg_lambda = lambdas[k] # set the regularization lambda
        model_path = './models/model_'+modelkey+'_P_rep_{}_lambda_{:d}_10.pt'.format(r,int(reg_lambda*10)) # path to which we will save the model
        model_P[k+r*N_LAMBDA] = MLP(N_INPUTS,N_HIDDEN_NEURONS,N_OUTPUTS,device).to(device) # create the model
        l2_reg = torch.tensor(1,device=device) # create the l2 regularization value tensor
        optimizer = torch.optim.SGD(model_P[k+r*N_LAMBDA].parameters(), lr=lr, momentum=momentum) # set the function for SGD
        criterion = nn.CrossEntropyLoss() # set the loss function
        
        # note that cross-entropy loss expects the indices of the class, not the one-hot. So, for A = [1,0,0,...] and B = [0,1,0,...], A is 0 and B is 1
        
        for epoch in range(N_EPOCHS): # for each training epoch
            nps = 0
            running_train_loss=0
            running_train_acc=0
            model_P[k+r*N_LAMBDA].train() 
            for p, param in enumerate(model_P[k+r*N_LAMBDA].parameters()): # go through all the model parameters
                if param.requires_grad:
                    plist = torch.flatten(param.data) # set the list of parameters
                    for j in range(plist.size(0)):
                        while nps < Phist_P.shape[0]:
                            Phist_P[nps,epoch,k,r]=plist[j].item() # update the parameters
                            nps+=1

            for i, (x, y_tar) in enumerate(trainloader):
                # print(i,x,y_tar)
                l2_reg = 0
                x, y_tar = x.to(device), y_tar.to(device) # x is the training set, y_tar is the output label
                x = x-0.3
                optimizer.zero_grad() # set gradients to 0
                # print(x.shape)
                y_pred = model_P[k+r*N_LAMBDA](x.view(x.shape[0],x.shape[1]*x.shape[2])) # compute the prediction. # size mismatch
                
                
                loss = criterion(y_pred,y_tar) 
                for p,param in enumerate(model_P[k+r*N_LAMBDA].parameters()):
                    if param.requires_grad and len(param.shape)==2:
                        if param.shape[0]==N_HIDDEN_NEURONS and param.shape[1]==N_HIDDEN_NEURONS:
                            l2_reg = l2_reg + param[:gidx,:gidx].norm(p=1) # update the l1 regularization constant
                            l2_reg = l2_reg + param[gidx:,gidx:].norm(p=1)
#                         elif param.shape[1]==N_HIDDEN_NEURONS:
#                             l2_reg = l2_reg + param[:,gidx:].norm(p=1)
#                         elif param.shape[0]==N_HIDDEN_NEURONS:
#                             l2_reg = l2_reg + param[:gidx,:].norm(p=1)
                regval_P.append(l2_reg.item()) # add the l2 regularization to  the running list
                loss = loss + l2_reg*reg_lambda/BATCH_SIZE # compute the loss
                loss.backward() # backpropogate the loss
                optimizer.step() # run SGD
                running_train_loss+=loss.item()
                running_train_acc+=get_accuracy(y_pred, y_tar) # compute accuracy
            
            running_test_acc=0
            running_test_loss=0
            model_P[k+r*N_LAMBDA].eval()
            for i,(x_test, y_test_tar) in enumerate(testloader):
                x_test, y_test_tar = x_test.to(device), y_test_tar.to(device)
                x_test = x_test - 0.3
                y_test_pred = model_P[k+r*N_LAMBDA](x_test.view(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
                loss = criterion(y_test_pred,y_test_tar)
                
                running_test_loss+=loss.item()
                running_test_acc+=get_accuracy(y_test_pred, y_test_tar)
                
            train_loss_P[epoch,k,r] = running_train_loss/len(trainloader)
            train_acc_P[epoch,k,r] = running_train_acc/len(trainloader)
            test_loss_P[epoch,k,r] = running_test_loss/len(testloader)
            test_acc_P[epoch,k,r] = running_test_acc/len(testloader)
            print("Epoch %d; rep %d; lambda %f; train accuracy %f; train loss %f; test accuracy %f; test loss %f; reg val %f"
                  %(epoch,
                    r,
                    reg_lambda,
                    train_acc_P[epoch,k,r],
                    train_loss_P[epoch,k,r],
                    test_acc_P[epoch,k,r],
                    test_loss_P[epoch,k,r],
                   l2_reg.item()))
            
        # save the model and free the memory  
        torch.save(model_P[k+r*N_LAMBDA].state_dict(), model_path)
        model_P[k+r*N_LAMBDA] = [None]
        del(l2_reg,loss,optimizer,criterion,plist,param)


# In[ ]:


import pickle

# BATCH_SIZE = 500 # how many batches we are running
# N_STEPS = 10 # How many characters are we inputting into the list at a time
# N_HIDDEN_NEURONS = 512 # how many neurons per hidden layer
# N_INPUTS = 77*N_STEPS
# N_OUTPUTS = 77
# N_LAYERS = 2 # 2 hidden layers
# N_EPOCHS = 11 # how many training epocs
# learning_rates = np.asarray([2]) # learning rates
# N_REPS = 3 # len(learning_rates) # the number of learning repetitions
# N_PARAMS = nparam_MLP(N_INPUTS,N_HIDDEN_NEURONS,N_OUTPUTS)
# gidx = int(N_HIDDEN_NEURONS/2)


# train_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS)) 
# train_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# test_loss_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# test_acc_P = np.zeros((N_EPOCHS,N_LAMBDA,N_REPS))
# Phist_P = np.zeros((N_PARAMS,N_EPOCHS,N_LAMBDA,N_REPS))

# model_P = [None]*N_LAMBDA*N_REPS # array of models
# regval_P = [] # array of regularization values



pickle.dump([lambdas,N_EPOCHS,N_REPS,N_HIDDEN_NEURONS,learning_rates,
             N_PARAMS,N_EPOCHS,N_LAMBDA,N_REPS,
             model_P,regval_P,
             train_loss_P,train_acc_P,
             test_loss_P,test_acc_P,
             Phist_P], 
            open(modelkey+"_longepochs_mlp_ak_quickset.pkl", "wb" ) )


# In[ ]:


# #plt.imshow(x[0,:,:])
# #plt.plot(y_pred.detach().numpy()[0,:])
# #torch.max(y_pred,1)
# plt.figure(1)
# plt.plot(np.mean(test_acc_P,2))
# plt.xlabel("Epoch")
# plt.ylabel("Test accuracy")
# plt.plot()

# plt.figure(2)
# plt.plot(np.mean(test_loss_P,2))
# plt.xlabel("Epoch")
# plt.ylabel("Test loss")
# plt.plot()


# for i,j in enumerate(zip(np.mean(test_acc_P,2),np.mean(test_loss_P,2))):
#     # plt.figure(i+3) # Here's the part I need, but numbering starts at 1!
#     fig,axs = plt.subplots(1,2)
#     fig.suptitle("Epoch %d"%i)
#     axs[0].plot(lambdas,j[0])
#     axs[0].set_xlabel("Lambda")
#     axs[0].set_ylabel("Test accuracy")
#     # axs[0].title("Epoch %d"%i)
#     axs[1].plot(lambdas,j[1])
#     axs[1].set_xlabel("Lambda")
#     axs[1].set_ylabel("Test loss")
#     # axs[1].title("Epoch %d"%i)
    
# plt.plot(regval_P)


# # plt.plot(np.mean(test_acc_P,1))
# # plt.plot()

# # plt.plot(np.mean(test_acc_P,2))
# # plt.plot()

# # plt.plot(np.mean(test_acc_P,3))
# # plt.plot()


# In[ ]:



# number of lambdas
n_epochs_plot,n_lambdas_plot = np.mean(test_acc_P,2).shape

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(hspace=20.0)

ax1,ax2 = ax
cm1 = plt.get_cmap('copper') # plt.get_cmap('gist_rainbow') # plt.get_cmap('gist_rainbow')
# fig = plt.figure()
NUM_COLORS = n_lambdas_plot
ax1.set_prop_cycle('color', [cm1(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
ax2.set_prop_cycle('color', [cm1(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

fig.suptitle("Epoch %d; reps %d; min/max lambda [%f,%f]"
             %(epoch,
               N_REPS,
               min(lambdas),
               max(lambdas)))

# train accuracy %f; train loss %f; test accuracy %f; test loss %f"
# train_acc_P[epoch,k,r],
#                train_loss_P[epoch,k,r],
#                test_acc_P[epoch,k,r],
#                test_loss_P[epoch,k,r]
            
# ax = fig.add_subplot(111)
for i in enumerate(lambdas):
    ax1.plot(np.mean(test_acc_P,2)[:,i[0]],label=i[1],)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    
# ax2 = fig.add_subplot(011)
for i in enumerate(lambdas):
    ax2.plot(np.mean(test_loss_P,2)[:,i[0]],label=i[1],)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Loss")
ax2.legend(loc="upper right")


# In[ ]:


plt.plot(regval_P)


# # Check past model

# In[ ]:


# import pickle
# # picked_data = pickle.load("mlp_ak_set.pkl")

# with open('mlp_ak_set.pkl','rb') as pickle_file:
#     picked_data = pickle.load(pickle_file)
#     [lambdas,N_EPOCHS,N_REPS,N_HIDDEN_NEURONS,learning_rates,N_REPS,N_PARAMS,
#              model_P,
#              train_loss_P,train_acc_P,
#              test_loss_P,test_acc_P,
#              Phist_P] = picked_data
#     N_HIDDEN = N_HIDDEN_NEURONS


# In[ ]:


# # load training models
# import glob
# modelfiles = glob.glob("./**.pt")
# past_models = []
# for i in modelfiles:
#     past_models.append(torch.load(i))


# In[ ]:


# past_models[0]


# In[ ]:


# plt.figure(figsize=(14,5))
# plt.subplot(1,2,1)
# sidx = int((N_INPUTS+N_OUTPUTS+1)*N_HIDDEN)
# ridx = int(N_HIDDEN*N_HIDDEN)
# rval_P = np.asarray(regval_P)
# rval_P = np.mean(rval_P.reshape(N_REPS,int(len(rval_P)/N_REPS)).T,axis=1)
# rval_P = rval_P.reshape(N_LAMBDA,int(len(rval_P)/N_LAMBDA)).T
# # rval_C = np.asarray(regval_C)
# # rval_C = np.mean(rval_C.reshape(N_REPS,int(len(rval_C)/N_REPS)).T,axis=1)
# # rval_C = rval_C.reshape(N_LAMBDA,int(len(rval_C)/N_LAMBDA)).T
# # rval_R = np.asarray(regval_R)
# # rval_R = np.mean(rval_R.reshape(N_REPS,int(len(rval_R)/N_REPS)).T,axis=1)
# # rval_R = rval_R.reshape(N_LAMBDA,int(len(rval_R)/N_LAMBDA)).T
# plt.plot(lambdas,rval_P[-1,],label='diagonal')
# # plt.plot(lambdas,rval_C[-1,],label='off-diagonal')
# # plt.plot(lambdas,rval_R[-1,],label='random')
# plt.title('L1 norm of regularized hidden-hidden weight')
# plt.ylabel('L1 norm')
# plt.xlabel('lambda')
# plt.subplot(1,2,2)
# plt.plot(lambdas,np.mean(np.sum(np.abs(Phist_P[sidx:(sidx+ridx),-1,:,:].squeeze())<0.0025,axis=0),axis=1)/ridx,'-',label='diagonal')
# # plt.plot(lambdas,np.mean(np.sum(np.abs(Phist_C[sidx:(sidx+ridx),-1,:,:].squeeze())<0.0025,axis=0),axis=1)/ridx,'-',label='off-diagonal')
# # plt.plot(lambdas,np.mean(np.sum(np.abs(Phist_R[sidx:(sidx+ridx),-1,:,:].squeeze())<0.0025,axis=0),axis=1)/ridx,'-',label='random')
# plt.plot(lambdas,0.5*np.ones_like(lambdas),'k--')
# plt.ylim((0,.55))
# plt.legend()
# plt.title('Fraction of zero weight')
# plt.xlabel('lambda')
# plt.ylabel('Fraction')
# plt.show()


# In[ ]:


# plt.hist(Phist_P[:,-1,-1,0],label='diagonal',normed=1, histtype='step',bins=np.arange(-0.5,0.5,0.005),log=True)
# plt.hist(Phist_C[:,-1,-1,0],label='off-diagonal',normed=1, histtype='step',bins=np.arange(-0.5,0.5,0.005),log=True)
# plt.hist(Phist_R[:,-1,-1,0],label='random',normed=1, histtype='step',bins=np.arange(-0.5,0.5,0.005),log=True)
# plt.xlim((-.25,.25))
# plt.xlabel('Wedight')
# plt.show()

# plt.figure(figsize=(14,5))
# plt.subplot(1,2,1)
# plt.plot(lambdas,1/(1+np.exp(-np.mean(Phist_P[0,-1,:,:],axis=1).squeeze())),'-',label='diagonal')
# plt.plot(lambdas,1/(1+np.exp(-np.mean(Phist_C[0,-1,:,:],axis=1).squeeze())),'-',label='off-diagonal')
# plt.plot(lambdas,1/(1+np.exp(-np.mean(Phist_R[0,-1,:,:],axis=1).squeeze())),'-',label='random')
# plt.ylim((0.5,0.8))
# plt.legend()
# plt.title('Size of time step vs. regularization')
# plt.ylabel('Time step size')
# plt.xlabel('lambda')
# plt.subplot(1,2,2)
# plt.plot(lambdas,np.mean(Phist_P[1,-1,:,:],axis=1).squeeze(),'-',label='diagonal')
# plt.plot(lambdas,np.mean(Phist_C[1,-1,:,:],axis=1).squeeze(),'-',label='off-diagonal')
# plt.plot(lambdas,np.mean(Phist_R[1,-1,:,:],axis=1).squeeze(),'-',label='random')
# plt.ylim((2,3.5))
# plt.legend()
# plt.title('Amplification vs. Regularization')
# plt.ylabel('Amplification')
# plt.xlabel('lambda')
# plt.show()


# # Plotting by lambdas

# In[ ]:



# # number of lambdas
# n_epochs_plot,n_lambdas_plot = np.mean(test_acc_P,2).shape

# fig,ax = plt.subplots(1,2)
# fig.subplots_adjust(hspace=20.0)

# ax1,ax2 = ax
# cm1 = plt.get_cmap('copper') # plt.get_cmap('gist_rainbow') # plt.get_cmap('gist_rainbow')
# # fig = plt.figure()
# NUM_COLORS = n_lambdas_plot
# ax1.set_prop_cycle('color', [cm1(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# ax2.set_prop_cycle('color', [cm1(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

# fig.suptitle("Epoch %d; reps %d; min/max lambda [%f,%f]"
#              %(epoch,
#                N_REPS,
#                min(lambdas),
#                max(lambdas)))

# # train accuracy %f; train loss %f; test accuracy %f; test loss %f"
# # train_acc_P[epoch,k,r],
# #                train_loss_P[epoch,k,r],
# #                test_acc_P[epoch,k,r],
# #                test_loss_P[epoch,k,r]
            
# # ax = fig.add_subplot(111)
# for i in enumerate(lambdas):
#     ax1.plot(np.mean(test_acc_P,2)[:,i[0]],label=i[1],)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Test Accuracy")
    
# # ax2 = fig.add_subplot(011)
# for i in enumerate(lambdas):
#     ax2.plot(np.mean(test_loss_P,2)[:,i[0]],label=i[1],)
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Test Loss")
# ax2.legend(loc="upper right")


# In[ ]:


# def readtxt(txt_name = 'anna.txt'):
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     txt_file = os.path.join(dir_path,txt_name)
#     # load the whole book
#     file = open(self.txt_file)
#     alltxt = file.read()
#     # remove newline formmating
#     alltxt = alltxt.replace("\n\n", "&").replace("\n", " ").replace("&", "\n")
#     # define categories
#     categories = list(sorted(set(alltxt)))
#     # integer encode
#     label_encoder = LabelEncoder()
#     label_encoder.fit(categories)
#     integer_encoded = torch.LongTensor(label_encoder.transform(list(alltxt)))
#     return integer_encoded, categories

# # def onehotencode(integer_encoded_batch,n_cat):
    
# # def get_next_batch(dat,batch_size):
# #     x_int = 
# #     y_int = 
# #     x_hot = onehotencode(x_int): 
# #     return x_hot, y_int 
    
    


# In[ ]:


# print(test_acc_P.shape)
# plt.plot(np.mean(test_acc_P,2))
# plt.plot()
# print(np.mean(test_acc_P,1))
# print(np.mean(test_acc_P,1).shape)

