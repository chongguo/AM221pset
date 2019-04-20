# -*- coding: utf-8 -*-
"""word_prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aNZESPVMWAbQjN_BvD6gwS7y5y11vzLX
"""



"""# Introduction:

This notebook will experiment with a character-based word prediction task. We will be following the tutorial here: https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
# %matplotlib inline  

import numpy as np

# alphabet
import string

# load the data
import os,sys,re
from google.colab import files
from google.colab import drive

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

text_chunk = 'hamlet.txt'
poem = uploaded[text_chunk].decode("utf-8")
# strip away punctuation; keep spaces!
# poem_string = [i.replace('.','') for i in poem.split('\n') if i is not '']
# poem_string = [i.replace(',','') for i in poem_string]
# poem_string = [i.replace(';','') for i in poem_string]
# poem_string = [i.replace('\'','') for i in poem_string]
# poem_string = [i.replace(':','') for i in poem_string]
# poem_string = [i.replace('\[','') for i in poem_string]
# poem_string = [i.replace('\]','') for i in poem_string]


poem_string = [i for i in poem.split('\n') if i is not '']
# poem_string = [i.replace(' ','') for i in poem_string]
poem_string = ' '.join(poem_string)
# lower case everything
poem_string = ''.join([i.lower() for i in poem_string])
# print(poem_string[0:100])
print(poem_string[0])

print(string.punctuation)

# one-hot encoding of characters
def one_hot_character(instr,addspace=True,add_punctuation=True):
  # generate the alphabet
  english_alpha = string.ascii_lowercase + string.punctuation + string.digits
  # if addspace, then incorporate into the alphabet
  if addspace:
    english_alpha = english_alpha+' ' # add the space
  if add_punctuation:
    english_alpha = english_alpha+',.;\'' # add punctuation
  # now, generate one-hot encoding vectors...
#   source_Vector = np.zeros(len(english_alpha))
  one_hot = []
  for i,j in enumerate(instr):
    # alphabet index
    try:
      alpha_ind = english_alpha.index(j)
      source_Vector = np.zeros(len(english_alpha))
      np.put(source_Vector,alpha_ind,float(1))
      one_hot.append(source_Vector)
    except IOError:
      continue
    else:
      continue
  return(one_hot)

def alphabet_index_encode(instr,addspace=True,add_punctuation=True):
  # generate the alphabet
  english_alpha = string.ascii_lowercase + string.punctuation + string.digits
  if addspace:
    english_alpha = english_alpha+' ' # add the space
  if add_punctuation:
    english_alpha = english_alpha+',.;\'' # add punctuation
  # now, generate one-hot encoding vectors...
#   source_Vector = np.zeros(len(english_alpha))
  ind_encode = []
  for i,j in enumerate(instr):
    # print(i,j)
    # alphabet index
    try:
#       print(j)
      alpha_ind = english_alpha.index(j)
      ind_encode.append(alpha_ind)
    except IOError:
#       print(j)
      continue
    else:
#       print(j)
      continue
  return(ind_encode)


def make_string_tensor(instr,addspace,add_punctuation,N_OUTPUT=1,N_INPUT=10):
#   N_OUTPUT = 1
#   N_INPUT = 10
  total_len = N_INPUT + N_OUTPUT

  # one-hot encode the poem
#   poem_string_onehot = one_hot_character(poem_string)
  poem_string_onehot = alphabet_index_encode(instr,addspace=addspace,add_punctuation=add_punctuation)
  
  # tile the string:
  string_tiles = []
  for i in range(total_len,len(poem_string_onehot)+1):
    j = poem_string_onehot[i - total_len:i]
    # turn this into a tensor:
    input_tensor = torch.FloatTensor(j[:-1])
    output_tensor = torch.FloatTensor([j[-1]])#,dtype=torch.float64)
    string_tiles.append([input_tensor,output_tensor])
  return (string_tiles)

def make_onehot_string_tensor(instr,addspace,add_punctuation,N_OUTPUT=1,N_INPUT=10):
#   N_OUTPUT = 1
#   N_INPUT = 10
  total_len = N_INPUT + N_OUTPUT

  # one-hot encode the poem
#   poem_string_onehot = one_hot_character(poem_string)
  poem_string_onehot = one_hot_character(poem_string,addspace=addspace,add_punctuation=add_punctuation)
  
  # tile the string:
  string_tiles = []
  for i in range(total_len,len(poem_string_onehot)+1):
    j = poem_string_onehot[i - total_len:i]
    # turn this into a tensor:
    input_tensor = torch.FloatTensor(j[:-1])
    # print(j)
#     print(input_tensor)
    output_tensor = torch.FloatTensor([j[-1]])#,dtype=torch.float64)
#     print(output_tensor)
    string_tiles.append([input_tensor,output_tensor])
  return(string_tiles)



def decode_string_vec(invec,addspace=True):
  # generate the alphabet
  english_alpha = string.ascii_lowercase
  # if addspace, then incorporate into the alphabet
  if addspace:
    english_alpha = english_alpha+' ' # add the space
  #
  outchar = english_alpha[list(invec).index(1)]
  return(outchar)

print(len(string.ascii_lowercase + string.punctuation + string.digits + ' '))
print(len(poem_string))
print(len(poem_string[:100000]))
print(len(range(10,len(poem_string[:100000])+1)))
shorter_poem = poem_string[:176200]
print(len(shorter_poem))

## INPUT preparation
# one-hot encode the poem string
# break the text chunks into N_INPUT input characters and N_OUTPUT output characters
poem_string_input = make_string_tensor(shorter_poem,addspace=True,add_punctuation=False,N_INPUT=100)
print(len(poem_string_input))
# poem_string_input = make_string_tensor(poem_string)

# cast as float64
# tensor.cast(poem_string_input, tf.float64)
# # test decoding!
# test_decode_out = []
# for i in poem_string_input:
#   outvec = i[1]
#   test_decode_out.append(decode_string_vec(outvec))

# generate trainloader and testloader sets
BATCH_SIZE = 100

# # list all transformations
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.1307,), (0.3081,))])

# # download and load training dataset
# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(poem_string_input, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# # download and load testing dataset
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(poem_string_input, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# run Vanilla RNN
# parameters 
N_STEPS = 10
N_INPUTS = 1
N_HIDDEN = 20
N_OUTPUTS = 10
N_EPHOCS = 50

model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)
# model = MyModel()
# model.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss() # this does not work

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
#         print(x.shape)
#         print(y_tar.shape)
# #         y_tar = torch.tensor(y_tar, dtype=torch.long)

#         print(y_tar)
#         print(y_tar.size())
        
        y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
#         print(y_pred.view(BATCH_SIZE,N_OUTPUTS).shape)
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(),y_tar.float())
#         print(loss)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(), y_tar.long(), BATCH_SIZE)
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))

# character/word RNN
# include a method in the class to perform one-hot encoding of inputs and outputs.

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
  # print(torch.max(logit, 1)[1].view(target.size()).data)
  corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum() # input size is the same as batch size. 
  accuracy = 100.0 * corrects/batch_size
  return accuracy.item()

# with RNN-LSTM
# run Vanilla RNN
# parameters 
N_STEPS = 10
N_INPUTS = 1
N_HIDDEN = 20
N_OUTPUTS = 10
N_EPHOCS = 50

model = PRNN_LSTM(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)
# model = MyModel()
# model.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss() # this does not work

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
#         y_tar = torch.tensor(y_tar, dtype=torch.long)

#         print(y_tar)
#         print(y_tar.size())
        
        y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
#         print(y_pred.view(BATCH_SIZE,N_OUTPUTS).shape)
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(),y_tar.float())
#         print(loss)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(), y_tar.long(), BATCH_SIZE)
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))

"""Both aren't working very well. Let's try the following:


1.   Use one-hot encoding and go back to CrossEntropy Loss Criterion
2.   Expand the number of input/output characters.
"""

# switch to one-hot encodings
#poem_string_onehot_input = make_onehot_string_tensor(poem_string,addspace=True,add_punctuation=False)
poem_string_input = make_string_tensor(poem_string,addspace=True,add_punctuation=False)

# train/test loader
# generate trainloader and testloader sets
BATCH_SIZE = 12

trainloader = torch.utils.data.DataLoader(poem_string_input, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# # download and load testing dataset
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(poem_string_input, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# run Vanilla RNN
# parameters 
N_STEPS = 10
N_INPUTS = 10
N_HIDDEN = 200
N_OUTPUTS = 69 # number of possible outputs
N_EPHOCS = 50

# N_STEPS = 28
# N_INPUTS = 28
# N_HIDDEN = 100
# N_OUTPUTS = 10
# N_EPHOCS = 10


model = PRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,0.1)
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
#         input_batch_size = int(x.shape[1]*x.shape[0]/N_STEPS)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
        
#         print(y_pred.shape)
        # print(y_tar.shape)
#         loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),Variable(y_tar.squeeze().long()))
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),Variable(y_tar.squeeze().long()))
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(), Variable(y_tar.long()), BATCH_SIZE)
        
        
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))

# RNN-LSTM
# parameters 
N_STEPS = 10
N_INPUTS = 10
N_HIDDEN = 200
N_OUTPUTS = 69 # number of possible outputs
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
#         input_batch_size = int(x.shape[1]*x.shape[0]/N_STEPS)
        y_pred = model(x.view(BATCH_SIZE,N_STEPS,N_INPUTS))
        
#         print(y_pred.shape)
        # print(y_tar.shape)
#         loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),Variable(y_tar.squeeze().long()))
        loss = criterion(y_pred.view(BATCH_SIZE,N_OUTPUTS),Variable(y_tar.squeeze().long()))
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_acc+=get_accuracy(y_pred.view(BATCH_SIZE,N_OUTPUTS).float(), Variable(y_tar.long()), BATCH_SIZE)
        
        
    train_running_loss[epoch] = running_loss
    train_acc[epoch] = running_acc/i
    
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_running_loss[epoch], train_acc[epoch]))