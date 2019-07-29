import datetime
import os
import random
import time
import gc
import sys
import numpy as np
from skimage import io
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets import list_dataset
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

# Predefining directories.
ckpt_path = './ckpt'
outp_path = './outputs'

# Reading system parameters.
conv_name = sys.argv[1]
fold_name = sys.argv[2]
data_name = sys.argv[3]
task_name = sys.argv[4]

# Setting experiment name.
exp_name  = conv_name + '_' + data_name + '_' + task_name + '_' + fold_name

# Setting predefined arguments.
args = {
    'epoch_num': 1000,     # Number of epochs.
    'lr': 1e-4,           # Learning rate.
    'weight_decay': 5e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
    'lr_patience': 100,   # Patience for the scheduler to reduce learning rate.
    'num_workers': 0,     # Number of workers on data loader.
    'snapshot': '',       # Network to use as basis for training.
    'batch_size': 4,      # Mini-batch size.
    'print_freq': 10,      # Printing frequency for mini-batch loss.
    'val_freq': 1,         # Validate each val_freq epochs.
    'patch_size': 480     # image patch size during training
}

# Main function.
def main(train_args):

    if (conv_name == 'segnet'):
        net = segnet(num_classes=list_dataset.num_classes, in_channels=3).cuda()
        net.init_vgg16_params()

    # Loading optimizer state in case of resuming training.
    if len(train_args['snapshot']) == 0:

        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    else:

        print 'training resumes from ' + train_args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'lr': optimizer.param_groups[1]['lr'], 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'iou': float(split_snapshot[9])}

    # Setting datasets.
    train_set = list_dataset.ListDataset('train', data_name, task_name, fold_name, 'statistical', train_args['patch_size'])
    train_loader = DataLoader(train_set, batch_size=train_args['batch_size'], num_workers=train_args['num_workers'], shuffle=True)

    val_set = list_dataset.ListDataset('validation', data_name, task_name, fold_name, 'statistical')
    val_loader = DataLoader(val_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    test_set = list_dataset.ListDataset('test', data_name, task_name, fold_name, 'statistical')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    # Setting criterion.
    if data_name == 'grss_semantic':
        criterion = CrossEntropyLoss2d(ignore_index = -1).cuda()
    else:
        criterion = CrossEntropyLoss2d().cuda()
    # Setting optimizer.
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.99))

    # Loading optimizer state in case of resuming training.
    if len(train_args['snapshot']) > 0:

        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']

    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))

    # Writing training args to experiment log file.
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')

    # Setting plateau scheduler for learning rate.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=train_args['lr_patience'], min_lr=2.5e-5, verbose=True)

    # Iterating over epochs.
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):

        # Training function.
        train_iou = train(train_loader, net, criterion, optimizer, epoch, train_args)

        if epoch % 1 == 0:

            # Computing validation loss and test loss.
            val_iou = validate(val_loader, net, criterion, optimizer, epoch, train_args, task_name)
            #test_iou = test(test_loader, net, criterion, optimizer, epoch, train_args, task_name)

            # Scheduler step.
            scheduler.step(val_iou)

# Training function.
def train(train_loader, net, criterion, optimizer, epoch, train_args):

    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = AverageMeter()

    # Lists for whole epoch loss.
    inps_all, labs_all, prds_all = [], [], []

    # Iterating over batches.
    for i, data in enumerate(train_loader):

        # Obtaining images, labels and paths for batch.
        inps, labs, img_name = data

        # Casting tensors to cuda.
        inps, labs = inps.cuda(), labs.cuda()

        # Sanity check on image/label sizes.
        #print(inps.size(), labs.size())
        assert inps.size()[2:] == labs.size()[1:]

        # Casting to cuda variables.
        inps = Variable(inps).cuda()
        labs = Variable(labs).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)
        
        # Computing loss.
        loss = criterion(outs, labs)

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        # Sanity check on output, label and num_class.
        assert outs.size()[2:] == labs.size()[1:]
        assert outs.size()[1] == list_dataset.num_classes

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Appending images for epoch loss calculation.
        inps_all.append(inps.data.squeeze_(0).cpu())
        labs_all.append(labs.data.squeeze_(0).cpu().numpy())
        prds_all.append(prds)

        # Updating loss meter.
        train_loss.update(loss.data[0], inps.size(0))

        # Printing.
        #if (i + 1) % train_args['print_freq'] == 0:
        #    print '[epoch %d], [iter %d / %d], [train loss %.5f]' % (
        #        epoch, i + 1, len(train_loader), train_loss.avg()
        #    )

    # Computing error metrics for whole epoch.
    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, labs_all, list_dataset.num_classes, task_name)

    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [lr %.9f], [train loss %.4f], [acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % (
        epoch, optimizer.param_groups[1]['lr'], train_loss.avg(), acc, acc_cls, mean_iou, fwavacc, kappa))
    print('--------------------------------------------------------------------')

    # Returning iou.
    return mean_iou


def validate(val_loader, net, criterion, optimizer, epoch, train_args, task_name):
    outp_path = './outputs'
    # Setting network for evaluation mode.
    net.eval()

    # Average Meter for batch loss.
    val_loss = AverageMeter()

    # Lists for whole epoch loss.
    inps_all, labs_all, prds_all = [], [], []
    
    save_img_epoch = 20

    if epoch % save_img_epoch == 0:
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))

    # Iterating over batches.
    for i, data in enumerate(val_loader):

        # Obtaining images, labels and paths for batch.
        inps, labs, img_name = data

        # Casting tensors to cuda.
        inps, labs = inps.cuda(), labs.cuda()

        # Casting to cuda variables.
        inps = Variable(inps, volatile=True).cuda()
        labs = Variable(labs, volatile=True).cuda()

        # Forwarding.
        outs = net(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        label = labs.data.squeeze_(0).cpu().numpy()

        # Saving prediction image
        if epoch % save_img_epoch == 0:
            tmp_path = os.path.join(outp_path, exp_name, 'epoch_{}'.format(epoch), img_name[0])
            h, w = prds.shape
            
            if data_name == 'grss_semantic':
                new = np.zeros((h, w, 3), dtype=np.uint8)
                for i in range(h):
                    for j in range(w):
                        if label[i][j] == -1:
                            new[i][j] = [0,0,0]
                        
                        elif prds[i][j] == 0:
                            new[i][j] = [255,0,255]
                            
                        elif prds[i][j] == 1:
                            new[i][j] = [0,255,0]
                            
                        elif prds[i][j] == 2:
                            new[i][j] = [255,0,0]
                            
                        elif prds[i][j] == 3:
                            new[i][j] = [0,255,255]
                            
                        elif prds[i][j] == 4:
                            new[i][j] = [160,32,240]
                            
                        elif prds[i][j] == 5:
                            new[i][j] = [46,139,87]
                            
                        else:
                            sys.exit('Invalid prediction')
                
                io.imsave(tmp_path+'.png', new)
            elif data_name == 'coffee_1_semantic' or data_name == 'coffee_2_semantic' or data_name == 'coffee_3_semantic' :
                new = np.zeros((h, w), dtype=np.uint8)
                            
                for i in range(h):
                    for j in range(w):
                        if prds[i][j] == 0:
                            new[i][j] = 0
                            
                        elif prds[i][j] == 1:
                            new[i][j] = 255
                            
                        else:
                            sys.exit('Invalid prediction')
                
                io.imsave(tmp_path+'.png', new)
                
            elif data_name == 'vaihingen_semantic':
                new = np.zeros((h, w, 3), dtype=np.uint8)
                for x in range(h):
                    for y in range(w):
                        if prds[x][y] == 0:
                            new[x][y] = [255,255,255]
                            
                        elif prds[x][y] == 1:
                            new[x][y] = [0,0,255]
                            
                        elif prds[x][y] == 2:
                            new[x][y] = [0,255,255]
                            
                        elif prds[x][y] == 3:
                            new[x][y] = [0,255,0]
                            
                        elif prds[x][y] == 4:
                            new[x][y] = [255,255,0]
                            
                        elif prds[x][y] == 5:
                            new[x][y] = [255,0,0]
                            
                        else:
                            sys.exit('Invalid prediction')
                io.imsave(tmp_path+'.png', new)
            
        # Appending images for epoch loss calculation.
        inps_all.append(inps.data.squeeze_(0).cpu())
        labs_all.append(label)
        prds_all.append(prds)

        # Updating loss meter.
        val_loss.update(loss.data[0], inps.size(0))

    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, labs_all, list_dataset.num_classes, task_name)
    
    if mean_iou > train_args['best_record']['iou']:

        train_args['best_record']['val_loss'] = val_loss.avg()
        train_args['best_record']['lr'] = optimizer.param_groups[1]['lr']
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['iou'] = mean_iou
        train_args['best_record']['fwavacc'] = fwavacc
        train_args['best_record']['kappa'] = kappa
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_iou_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg(), acc, acc_cls, mean_iou, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'best.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_best.pth'))

    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [lr %.9f], [val loss %.9f], [acc %.4f], [acc_cls %.4f] [iou %.4f] [kappa %.4f]' % (epoch, optimizer.param_groups[1]['lr'], val_loss.avg(), acc, acc_cls, mean_iou, kappa))

    # Printing best epoch loss so far.
    print('best record: [lr %.9f], [val loss %.9f], [acc %.4f], [acc_cls %.4f], [iou %.4f], [kappa %.4f], [epoch %d]' % (train_args['best_record']['lr'], train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'], train_args['best_record']['iou'], train_args['best_record']['kappa'] , train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    # Returning iou.
    return mean_iou

def test(test_loader, net, criterion, optimizer, epoch, train_args, task_name):

    # Setting network for evaluation mode.
    net.eval()

    # Average Meter for batch loss.
    test_loss = AverageMeter()

    # Lists for whole epoch loss.
    inps_all, labs_all, prds_all = [], [], []

    # Iterating over batches.
    for i, data in enumerate(test_loader):

        # Obtaining images, labels and paths for batch.
        inps, labs, img_name = data

        # Casting tensors to cuda.
        inps, labs = inps.cuda(), labs.cuda()

        # Casting to cuda variables.
        inps = Variable(inps, volatile=True).cuda()
        labs = Variable(labs, volatile=True).cuda()

        # Forwarding.
        outs = net(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        # Appending images for epoch loss calculation.
        inps_all.append(inps.data.squeeze_(0).cpu())
        labs_all.append(labs.data.squeeze_(0).cpu().numpy())
        prds_all.append(prds)

        # Updating loss meter.
        test_loss.update(loss.data[0], inps.size(0))

    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, labs_all, list_dataset.num_classes, task_name)
    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [lr %.9f], [test loss %.4f], [acc %.4f], [acc_cls %.4f], [iou %.4f +/- %.4f]' % (
        epoch, optimizer.param_groups[1]['lr'], test_loss.avg(), acc, acc_cls, mean_iou, iou_list.std()))
    print('--------------------------------------------------------------------')

    # Returning iou.
    return mean_iou #test_loss.avg


if __name__ == '__main__':
    main(args)
