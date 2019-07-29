import matplotlib
matplotlib.use('Agg')

import datetime
import os
import random
import time
import gc

import numpy as np

from PIL import Image
from skimage import io as skio
import torch
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from datasets import list_dataset
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

cudnn.benchmark = True

outp_path = './outputs'

conv_name = sys.argv[1]
fold_name = sys.argv[2]
data_name = sys.argv[3]
task_name = sys.argv[4]

exp_name  = conv_name + '_' + data_name + '_' + task_name + '_' + fold_name

def normalize_rows(array):
    sum = array.sum(axis=1)
    new = np.zeros(array.shape)
    for i in range(array.shape[0]):
        new[i] = array[i]/sum[i]
    return new


def main():

    if (conv_name == 'segnet'):
        net = segnet(num_classes=list_dataset.num_classes, in_channels=3).cuda()

    model_path = './ckpt/' + exp_name + '/best.pth'

    net.load_state_dict(torch.load(model_path))
    #print('net', net)
    #net = net.cuda()

    net.eval()
    
    batch_size = 1

    num_workers = 1

    test_set = []
    test_loader = []

    test_set = list_dataset.ListDataset('test', data_name, task_name, fold_name, 'statistical')
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))

    test(test_loader, net, task_name)

def test(test_loader, net, task_name):

    net.eval()

    check_mkdir(os.path.join(outp_path, exp_name, 'best'))

    criterion = CrossEntropyLoss2d(size_average=False).cuda()
    
    hr_preds = []
    hr_gts = []
    
    bicubic_preds = []
    bicubic_gts = []
    
    sr_preds = []
    sr_gts = []
    for vi, data in enumerate(test_loader):
        inputs, gts, img_name = data
        inputs = inputs.float()

        inputs, gts = inputs.cuda(), gts.cuda()

        #inputs = inputs.sub(inputs.mean()).div(inputs.std())

        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        gts = Variable(gts).cuda()

        outputs = net(inputs)
        
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        loss = criterion(outputs, gts)
        
        if 'bicubic' in img_name[0]:
            bicubic_preds.append(predictions)
            bicubic_gts.append(gts.data.squeeze_(0).cpu().numpy())
        elif 'result' in img_name[0]:
            sr_preds.append(predictions)
            sr_gts.append(gts.data.squeeze_(0).cpu().numpy())
        else:
            hr_preds.append(predictions)
            hr_gts.append(gts.data.squeeze_(0).cpu().numpy())
        
        acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate([predictions], [gts.data.squeeze_(0).cpu().numpy()], list_dataset.num_classes, task_name)
        
        print(img_name[0], loss.item())
        print('[acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % (acc, acc_cls, mean_iou, fwavacc, kappa))
        
        tmp_path = os.path.join(outp_path, exp_name, 'best', img_name[0])
        
        
        prds = predictions
        h, w = prds.shape
        if data_name == 'grss_semantic':
            new = np.zeros((h, w, 3), dtype=np.uint8)
            label = gts.data.squeeze_(0).cpu().numpy()
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
            
            skio.imsave(tmp_path+'.png', new)
        elif data_name == 'coffee_1_semantic' or data_name == 'coffee_2_semantic' or data_name == 'coffee_3_semantic':
            new = np.zeros((h, w), dtype=np.uint8)
                        
            for i in range(h):
                for j in range(w):
                    if prds[i][j] == 0:
                        new[i][j] = 0
                        
                    elif prds[i][j] == 1:
                        new[i][j] = 255
                        
                    else:
                        sys.exit('Invalid prediction')
            
            skio.imsave(tmp_path+'.png', new)
            
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
            skio.imsave(tmp_path+'.png', new)

            
    if data_name == 'grss_semantic':
        y_labels = ['Road', 'Tree', 'Red roof', 'Grey roof', 'Concrete\nroof', 'Vegetation']
        sr_heatmap = normalize_rows(confusion_matrix(sr_preds, sr_gts, list_dataset.num_classes))
        bicubic_heatmap = normalize_rows(confusion_matrix(bicubic_preds, bicubic_gts, list_dataset.num_classes))
        hr_heatmap = normalize_rows(confusion_matrix(hr_preds, hr_gts, list_dataset.num_classes))
        
        print('\nFinal:')
        print('HR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(hr_preds, hr_gts, list_dataset.num_classes, task_name))
        print('SR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(sr_preds, sr_gts, list_dataset.num_classes, task_name))
        print('Bicubic')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(bicubic_preds, bicubic_gts, list_dataset.num_classes, task_name))
        
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
        
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(bicubic_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'bicubic.png')
        
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(hr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'hr.png')
            
    elif data_name == 'coffee_1_semantic' or data_name == 'coffee_2_semantic' or data_name == 'coffee_3_semantic':
        y_labels = ['non-coffee', 'coffee']
        sr_heatmap = normalize_rows(confusion_matrix(sr_preds, sr_gts, list_dataset.num_classes))
        bicubic_heatmap = normalize_rows(confusion_matrix(bicubic_preds, bicubic_gts, list_dataset.num_classes))
        hr_heatmap = normalize_rows(confusion_matrix(hr_preds, hr_gts, list_dataset.num_classes))
        
        print('\nFinal:')
        print('HR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(hr_preds, hr_gts, list_dataset.num_classes, task_name))
        print('SR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(sr_preds, sr_gts, list_dataset.num_classes, task_name))
        print('Bicubic')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(bicubic_preds, bicubic_gts, list_dataset.num_classes, task_name))
        
        sns.set(font_scale=1.3) 
        fig = plt.figure(figsize=(3.5,3.5))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
        
        fig = plt.figure(figsize=(3.5,3.5))
        ax = sns.heatmap(bicubic_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'bicubic.png')
        
        fig = plt.figure(figsize=(3.5,3.5))
        ax = sns.heatmap(hr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'hr.png')
        
    elif data_name == 'vaihingen_semantic':
        y_labels = ['Impervious\nsurfaces', 'Building', 'Low\nvegetation', 'Tree', 'Car']
        sr_heatmap = normalize_rows(confusion_matrix(sr_preds, sr_gts, list_dataset.num_classes))
        bicubic_heatmap = normalize_rows(confusion_matrix(bicubic_preds, bicubic_gts, list_dataset.num_classes))
        hr_heatmap = normalize_rows(confusion_matrix(hr_preds, hr_gts, list_dataset.num_classes))
        sr_heatmap = np.delete(sr_heatmap, -1, axis=0)
        sr_heatmap = np.delete(sr_heatmap, -1, axis=1)
        bicubic_heatmap = np.delete(bicubic_heatmap, -1, axis=0)
        bicubic_heatmap = np.delete(bicubic_heatmap, -1, axis=1)
        hr_heatmap = np.delete(hr_heatmap, -1, axis=0)
        hr_heatmap = np.delete(hr_heatmap, -1, axis=1)

        
        print('\nFinal:')
        print('HR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(hr_preds, hr_gts, list_dataset.num_classes, task_name))
        print('SR')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(sr_preds, sr_gts, list_dataset.num_classes, task_name))
        print('Bicubic')
        print('acc: %.4f\nacc_cls: %.4f\nmean_iou: %.4f\niou: %s\nfwavacc: %.4f\nkappa: %.4f' % evaluate(bicubic_preds, bicubic_gts, list_dataset.num_classes, task_name))
        
        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
        
        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(bicubic_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'bicubic.png')
        
        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(hr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'hr.png')
    

if __name__ == '__main__':
    main()

