import torch
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,n_layers,model,device):
        super(CharRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.model = model
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        if self.model == "gru":
            self.rnn = nn.GRU(n_hidden, n_hidden, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers)
        self.device = device
        
    def forward(self, x, hidden):
        x=x.permute(1,0,2)
        hidden = torch.zeros(self.n_layers,x.shape[1],self.n_hidden).to(self.device)
        outhist, hidden = self.rnn(self.encoder(x),hidden)
        output = self.decoder(outhist)
        
        return output[-1,:,:], hidden