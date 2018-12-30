'''
This code is mainly copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import numpy as np
import trattack
from trattack.attack_utils import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default = 1, type = int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--eps', default=0.005, type=float,
                     help='perturbation magnitude')
parser.add_argument('--classes', default = 9, type = int,
                    help='attack the best/hardest classes of out 9 classes')
parser.add_argument('--norm', default = 2, type = int,
                    help='2 or 8 (Infity) norm')
parser.add_argument('--iter', default = 5000, type = int,
                    help='Largest number of iterations')
parser.add_argument('--worst-case', default = 0, type = int,
                    help='use best/hardest (worst) case attack')
parser.add_argument('--adap', action = 'store_true', default = False,
        help='Using adaptive method or not')
# for plotting
parser.add_argument('--plotting', action = 'store_true',
                    help='plot one of the example')
parser.add_argument('--plotting_path', type = str, default = './',
                    help='location for storing perturbed images')


def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend = args.dist_backend, init_method = args.dist_url,
                                world_size = args.world_size)

    if args.norm == 2:
        print('\nPerforming TR L2 Attack')
    elif args.norm == 8:
        print('\nPerforming TR Linf Attack')
    else:
        print('\nError! Incorrect option passed for norm')
        return

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)


    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers)


    bz = args.batch_size
    
    
    stat_time = time.time()
    X_ori = torch.Tensor(bz, 3, 224, 224)
    X_tr_first = torch.Tensor(bz, 3, 224, 224)


    Y_test = torch.LongTensor(bz)
    
    result_acc = 0.
    result_dis = 0.
    result_large = 0.

    start_time = time.time()
    for i, (data, target) in enumerate(test_loader):
        X_ori  = data
        Y_test = target
        
        # Apply TR Attack
        if not args.adap:
            X_tr_first, _ = trattack.tr_attack_iter(model, data, target, args.eps, c = args.classes, p = args.norm, iter = args.iter, worst_case = args.worst_case)
        else:
            X_tr_first, _ = trattack.tr_attack_adaptive_iter(model, data, target, args.eps, c = args.classes, p = args.norm, iter = args.iter, worst_case = args.worst_case)

        acc = validate_all(X_ori, Y_test, model, criterion)
        acc = validate_all(X_tr_first, Y_test, model, criterion)
        result_acc += acc
        
        dis, ldis = distance(X_tr_first.cpu(), X_ori)
        result_dis += dis
        result_large = max(ldis, result_large)
        if args.plotting:
            print('\nSaving purturbed images to ', args.plotting_path)
            plotting(X_ori[0:1, :], X_tr_first[0:1, :], model)
    print('time: %.4f' % (time.time() - start_time))
    print('\nAccuracy after TR perturbation: %.4f' % (result_acc / (i + 1)))
    print('\nAverage TR relative perturbation (among input images): %.4f' % (result_dis / (i + 1)))
    print('\nMaximum TR relative perturbation (among input images): %.4f' % result_large)



def plotting(X1, X2, model):
    model.eval()
    output1 = model(X1.cuda())
    output2 = model(X2.cuda())

    _, pred1 = torch.max(output1, dim=1)
    _, pred2 = torch.max(output2, dim=1)

    # transfer processing image back to [0,1]
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])

    X1_np = invTrans(X1[0, :]).cpu().numpy()
    X2_np = invTrans(X2[0, :]).cpu().numpy()

    X1_np = np.swapaxes(X1_np, 0, 2)
    X2_np = np.swapaxes(X2_np, 0, 2)

    X1_np = np.rot90(X1_np, k = -1, axes = (0, 1))
    X2_np = np.rot90(X2_np, k = -1, axes = (0, 1))

    X1_np[X1_np > 1] = 1
    X1_np[X1_np < 0] = 0

    X2_np[X2_np > 1] = 1
    X2_np[X2_np < 0] = 0
    
    fig=plt.figure(figsize=(10, 3))

    plt.subplot(131)
    plt.imshow(X1_np)
    plt.axis('off')
    plt.title('Origianl with Pred %d' % pred1)

    plt.subplot(132)
    plt.imshow(X2_np)
    plt.axis('off')
    plt.title('TR Perturbed with Pred %d' % pred2)

    plt.subplot(133)
    plt.imshow(np.sum(np.abs(X1_np-X2_np), axis=2), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.title('TR Perturbation')
    plt.colorbar()

    plt.savefig(args.plotting_path + 'image.png')

def validate_all(X, Y, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_data = X.size()[0]
    num_iter = num_data//1
    with torch.no_grad():
        end = time.time()
        for i in range(num_iter):
            #if args.gpu is not None:
            input = X[i : (i + 1),:].cuda(args.gpu, non_blocking = True)
            target = Y[i : (i + 1)].cuda(args.gpu, non_blocking = True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
