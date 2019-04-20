import torch
import torch.nn as nn
import torch.nn.functional as F

# character RNN based on GRU but using the PRNN formulation
class PCRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,device):
        super(CharRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        self.rnn = nn.GRU(n_hidden, n_hidden, n_layers)
        self.device = device
        
    def forward(self, x):
        x=x.permute(1,0,2)
        hidden = torch.zeros(self.n_layers,x.shape[1],self.n_hidden).to(self.device)
        output = torch.zeros(x.shape[0],x.shape[1],self.n_output).to(self.device)
        output, hidden = self.rnn(self.encoder(x),hidden)
        output = self.decoder(output)
        
        return output, hidden