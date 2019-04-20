#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Today, we will be doing the following:
# * Create a vanilla RNN (encode `tanh` activation functions and learn weights for the functions in the partitioned and unpartitioned models)
# * Write our own loss function
# 
# Optional:
# * Try out `dropout`, $L_2$, and `weight decay` regularization methods for the partitioned and unpartitioned models  

# # Setup

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
import torch


# In[ ]:


# Import tensorflow's MNIST data handle
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[ ]:


# generate a training dataset
image_count = 10000
data = mnist.train.next_batch(image_count)

images = data[0]
labels = data[1]


# Each training instance will scan down the 28x28 image; at time $t_i ,\; i = 1 \dots 28$, the model will take in a 1x28 row vector of pixels. The pattern of outputs will be associated with the class label that we present to the network. 

# In[ ]:


# create an input from the MNIST numbers in `data`
input_mnist = torch.tensor(data = images) # a tensor of the MNIST digits.
# note that Tensorflow's MNIST class labels are already one-hot encoded
output_mnist = torch.tensor(data = labels)


# # Building an RNN
# 
# We are building the RNN with PyTorch (see https://deepsense.ai/keras-or-pytorch/ for a comparison). We will be closely hewing to the tutorials https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html and https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# 
# 

# In[ ]:


# create a basic RNN

class basicRNN (torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      # initialize
    super(basicRNN, self).__init__()
    
    self.hidden_size = hidden_size
    self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size) # linear combination
    self.i2o = torch.nn.Linear(input_size + hidden_size, output_size) # linear combination
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, instate, hidden):
    # the forward function
    combined = torch.cat((instate, hidden), 1) # combine the new inputs and the state of the hidden layer from the past training timepoint
    hidden = self.i2h(combined) # generate the state of the hidden layer
    output = self.i2o(combined) # generate the input going to the output
    output = self.tanh(output) # put the combined input through the tanh activation function as the output
    return output, hidden # return the output and the hidden state

  def initHidden(self):
    return torch.zeros(1, self.hidden_size) # initialize the hidden layer. This will be useful when starting the training sequence


# In[ ]:


n_hidden = 10 # is there an algorithmic way to search for the number of hidden neurons we would like per layer?
n_row_pixels = 28 # our RNN will scan through each image row layer in order to learn the digit being reported from the network. 
n_number_labels = 10 # MNIST digits go from 0 - 9, but the class label is effectively categorical. we will represent the output as a one-hot 1x10 vector. 
test_rnn = basicRNN(input_size=n_row_pixels, hidden_size = n_hidden, output_size = n_number_labels) # initialize the RNN


# In[ ]:


# specify a loss function
criterion = torch.nn.NLLLoss() # The negative log likelihood loss. It is useful to train a classification problem with `C` classes. See https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html


# # Training

# ## Setup for training

# In[ ]:


learning_rate = 0.005 # How can we determine when to change the learning rate and to what? This itself is an interesting optimization problem

def train(input_tensor, output_tensor, in_rnn, loss_criterion):
    hidden = in_rnn.initHidden() # the hidden state is empty, since the network has not seen anything yet

    in_rnn.zero_grad() # initially, set parameter gradients to zero
    
    # for every training example that we see, get the network's output and input.
    # Remember to feed the hidden state of the network into the next example that the network sees. 
    # go through each block of 28 
    block_size = math.sqrt(input_tensor.size()[0])
    for i in range(input_tensor.size()[0]):
#      print(input_tensor)
      output, hidden = in_rnn(input_tensor[i,], hidden)
    
    # compute the value of the loss function
    loss = loss_criterion(output, category_tensor)
    # backpropogate the errors
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate. This is the parameter update step
    for p in in_rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

  
import random

# 
def randomChoice(l):
  # given some tensor, this function spits out a random row
    return(random.randint(0, len(l) - 1))

def randomTrainingExample(input_tensor,output_tensor):
  # spit out a random digit (really, a 1x(28^2) row vector)
  rand_row = randomChoice(output_mnist)
  randinput = input_tensor[rand_row,]
  randoutput = output_tensor[rand_row,]
  return randinput, randoutput

# for i in range(10):
#     i_input, i_output = randomTrainingExample(input_mnist,output_mnist)
# #    plt.plot(i_input.numpy())
#     plt.imshow(i_input.numpy)
# #    print('digit =', category, '/ line =', line)


# ## Loading training and testing sets
# 
# Taken from https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79

# In[ ]:


import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 64

# list all transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

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


hidden = test_rnn.initHidden() # the hidden state is empty, since the network has not seen anything yet
print(hidden)

test_rnn.zero_grad() # initially, set parameter gradients to zero

# initialize the class
output, hidden = test_rnn(input_mnist[0,], hidden)
# error is being thrown within the forward function. 


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))


# ## Run training

# In[ ]:


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for i in range(1, n_iters + 1):
    i_input, i_output = randomTrainingExample(input_mnist,output_mnist)
    output, loss = train(input_tensor = i_input, output_tensor = i_output, loss_criterion = criterion,in_rnn = test_rnn)
    current_loss += loss

    # Print i number, loss, name and guess
    if i % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if i % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# # Partitioned hidden layers

# In[ ]:


# create a basic RNN with partitioned hidden states--effectively two hidden layers that will communicate with each other as well as with themselves

class basicRNN (torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      # initialize
    super(RNN, self).__iniit__()
    
    self.hidden_size = hidden_size # both hidden layers will have the same size
    self.i2h1 = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)
  

  def forward(self, input, hidden):
    # the forward function
    combined = torch.cat((input, hidden), 1) # combine the new inputs and the state of the hidden layer from the past training timepoint
    hidden = self.i2h(combined) # generate the state of the hidden layer
    output = self.i2o(combined) # generate the input going to the output
    output = self.tanh(output) # put the combined input through the tanh activation function as the output
    return output, hidden # return the output and the hidden state

  def initHidden(self):
    return torch.zeros(1, self.hidden_size) # initialize the hidden layer. This will be useful when starting the training sequence
  


# In[ ]:


"""
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
"""

Hello

