import torch

def get_accuracy(logit, target):
    batch_size = target.shape[0]
    n_steps = target.shape[1]
    accuracy = (torch.max(logit, 1)[1].data == target.data).type(torch.DoubleTensor).mean()*100
    return accuracy.item()

def nparam(ninputs,nhidden,noutputs):
    return ninputs*(nhidden+1) + nhidden*(nhidden+1)+nhidden*(noutputs+1)