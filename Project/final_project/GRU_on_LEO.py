#!/usr/bin/env python
# coding: utf-8

# In[35]:


from Data import AnnaDataset, InvertAnna
from Models.CharRNN import CharRNN
from Utils.HelperFunctions import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tnrange, tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
get_ipython().run_line_magic('matplotlib', 'inline')
# use gpu when possible
mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(mydevice)


# In[19]:


# parameters
BATCH_SIZE = 2048
N_STEPS = 10
N_HIDDEN = 512
N_LAYERS = 2
N_EPOCHS = 11
learning_rates = np.asarray([1e-4,1e-6,1e-8])
N_REPS = len(learning_rates)

dataset = AnnaDataset(N_STEPS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

N_INPUTS = len(dataset.categories)
N_OUTPUTS = N_INPUTS


# In[20]:


train_loss = np.zeros((N_EPOCHS,N_REPS))
train_acc = np.zeros((N_EPOCHS,N_REPS))

model = [None]*N_REPS
for rep in tnrange(N_REPS):
    model[rep] = CharRNN(N_INPUTS,N_HIDDEN,N_OUTPUTS,N_LAYERS,"gru",mydevice).to(mydevice)
    optimizer = torch.optim.RMSprop(model[rep].parameters(), lr=learning_rates[rep], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model[rep].train()
    for epoch in tnrange(N_EPOCHS):
        running_train_loss = 0
        running_train_acc = 0
        for i, (x,y_tar) in enumerate(dataloader):
            x, y_tar = x.to(mydevice), y_tar.to(mydevice)
            y_pred, hidden = model[rep](x)
            loss = criterion(y_pred,y_tar)
            loss.backward()
            optimizer.step()
            running_train_loss+=loss.item()
            running_train_acc+=get_accuracy(y_pred, y_tar)
        train_loss[epoch,rep] = running_train_loss/(i+1)
        train_acc[epoch,rep] = running_train_acc/(i+1)


# In[21]:


plt.plot(learning_rates*1e6,train_acc[-1,:],'.')
plt.ylabel('Final Accuracy')
plt.xlabel('Learning Rate')
plt.show()
plt.plot(train_acc[:,np.argmax(train_acc[-1,:])])
plt.legend(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[49]:


# import importlib
# importlib.reload(InvertAnna)
# from Data import AnnaDataset, InvertAnna
from Data import AnnaDataset, InvertAnna

model[0].eval()
x, y_tar = next(iter(dataloader))
y_pred, hidden = model[0](x.to(mydevice))
print(''.join(InvertAnna(torch.max(y_pred, 1)[1].data[1,].to(mydevice))))
print(''.join(InvertAnna(y_tar[1,].to(mydevice))))


# In[37]:





# In[ ]:




