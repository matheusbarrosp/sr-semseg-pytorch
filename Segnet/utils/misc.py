import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
    	#print(inputs.shape)
        return self.nll_loss(F.log_softmax(inputs, dim = 1), targets)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def confusion_matrix(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
        
    return hist


def kappa_score(confusion):
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0
    
    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def evaluate(predictions, gts, num_classes, task_name):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
        
    #print(hist)
    # axis 0: gt, axis 1: prediction
       
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    kappa = kappa_score(hist)
    return acc, acc_cls, mean_iu, iu, fwavacc, kappa


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_list = list()
        #self.val = 0
        #self.avg = 0
        #self.sum = 0
        #self.count = 0

    def update(self, val, n=1):
        self.val_list.append(val)
        #self.val = val
        #self.sum += val * n
        #self.count += n
        #self.avg = self.sum / self.count
        #self.std = self.sum / self.count

    def avg(self):
        return np.asarray(self.val_list).mean()

    def std(self):
        return np.asarray(self.val_list).std()
