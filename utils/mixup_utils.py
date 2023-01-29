
import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def mixup_old_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = int(x.size()[0]/2)
    index = torch.randperm(batch_size).cuda()
    index = index + batch_size
    if lam < 0.5:
        lam = 1. - lam
    mixed_x = lam * x[:batch_size,:] + (1 - lam) * x[index,:]
    y_a, y_b = y[:batch_size], y[index]
    return mixed_x, y_a, y_b, lam

# mixed_x, y_a, y_b, lam = mixup_old_data(x, y, alpha=1.0)
# print(mixed_x)
# print(x)
# # print(y_a)
# # print(y_b)