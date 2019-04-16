import torch
import torch.nn as nn
import torch.nn.functional as F

class PRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,dt,device):
        super(PRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dt = nn.Parameter(torch.Tensor([dt])).to(device)
        self.a = nn.Parameter(torch.Tensor([1])).to(device)
        self.sig = nn.Sigmoid()
        self.decoder = nn.Linear(n_hidden, n_output)
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.recurrent = nn.Linear(n_hidden,n_hidden)
        self.device = device
        
    def forward(self, x):
        x=x.permute(1,0,2)
        self.h = torch.zeros(1,x.shape[1],self.n_hidden).to(self.device)
        self.y = torch.zeros(x.shape[0],x.shape[1],self.n_output).to(self.device)
        for i in range(x.shape[0]):
            self.h = (1-self.sig(self.dt))*self.h+self.sig(self.dt)*self.a*torch.tanh(self.encoder(x[i,:,:])+self.recurrent(self.h))
            self.y[i,:,:] = self.decoder(self.h)
        
        return self.y, self.h