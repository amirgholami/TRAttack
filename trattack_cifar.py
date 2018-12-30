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

from utils import *
from models.resnet import *

import HessianFlow.hessianflow as hf
import HessianFlow.hessianflow.optimizer.optm_utils as hf_optm_utils

import trattack
from trattack.attack_utils import *

# settings 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--test-batch-size', type = int, default = 1000, metavar =
        'N', help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action = 'store_true', default = False,
        help='disables CUDA training') 
parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help='random seed (default: 1)')
parser.add_argument('--norm', type = int, default = 2, metavar = 'S',
                    help='2 or 8 (infinity norm)')
parser.add_argument('--classes', type = int, default = 9, metavar = 'S',
                    help='select the best/hardest class of 9 classes')
parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eps', type = float, default = 0.001, metavar = 'E',
                        help='how far to perturb input')
parser.add_argument('--resume', type = str, default = 'net.pkl', help = 'choose an existing model')
parser.add_argument('--worst-case', type = int, default = 0, help =  'attack the best/worst (hardest) case')
parser.add_argument('--iter', type = int, default = 5000, help = 'largest number of iterations')
parser.add_argument('--adap', action = 'store_true', default = False,
        help='Using adaptive method or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

_, test_loader = getData(name = 'cifar10da', train_bs = args.test_batch_size, test_bs = args.test_batch_size)
bz = args.test_batch_size


for arg in vars(args):
    print(arg, getattr(args, arg))

# loading model

model = resnet(depth = 20)
if args.cuda:
    model.cuda()

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.resume))
model.eval()


######## begin attack
stat_time = time.time()
num_d = 10000 
X_ori = torch.Tensor(num_d, 3, 32, 32)
X_tr_first = torch.Tensor(num_d, 3, 32, 32)
iter_tr_first = 0.
Y_test = torch.LongTensor(num_d)


for i, (data, target) in enumerate(test_loader):
    X_ori [i * bz:(i + 1) * bz, :] = data
    Y_test[i * bz:(i + 1) * bz] = target
   
    if not args.adap:
        X_tr_first[i * bz:(i + 1) * bz,:], a = trattack.tr_attack_iter(model, data, target, args.eps, c = args.classes,
                p = args.norm, iter = args.iter, worst_case = args.worst_case)
        iter_tr_first += a
    else:
        X_tr_first[i * bz:(i + 1) * bz, :], a = trattack.tr_attack_adaptive_iter(model, data, target, args.eps, c = args.classes,
                p = args.norm, iter = args.iter, worst_case = args.worst_case)
        iter_tr_first += a

    print('current batch: %d' % i)

print('\ntotal generating data time: %.2f' % (time.time() - stat_time))


clean_acc = test_acc(model, X_ori, Y_test)
print('\nOriginal Accuracy %.4f' % (clean_acc))
result_acc = test_acc(model, X_tr_first, Y_test)
result_dis, result_large = distance(X_tr_first, X_ori, norm=args.norm)

print('\nAcc after TR attack: %.2f | Ave Distance (Perturbation) %.4f | Max Distance (Perturbation) %.4f' % (result_acc, result_dis, result_large))



