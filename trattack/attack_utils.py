from __future__ import print_function
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad

def test_acc(model, adv_data, Y_test, batch_size = 100, cuda = True):
    num_data = adv_data.size()[0]
    num_iter = num_data // batch_size
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(num_iter):
            data, target = adv_data[batch_size * i:batch_size * (i + 1), :], Y_test[batch_size * i:batch_size * (i + 1)]
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim = True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    return correct*1./num_data

def distance(X_adv, X_clean, norm=2):
    n = len(X_adv)
    dis = 0.
    large_dis = 0.
    for i in range(n):
        if norm == 2:
            tmp_dis = torch.norm(X_adv[i,:] - X_clean[i,:], p = norm) / torch.norm(X_clean[i,:], p=norm)
        if norm == 8:
            tmp_dis = torch.max(torch.abs(X_adv[i, :] - X_clean[i, :])) / torch.max(torch.abs(X_clean[i,:]))
        dis += tmp_dis
        large_dis = max(large_dis, tmp_dis)
    return dis / n, large_dis
