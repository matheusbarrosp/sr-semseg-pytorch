from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from data import get_eval_set
from functools import reduce

import time
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, help='Test input directory')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='Name of the test dataset')
parser.add_argument('--model_type', type=str, default='DBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--model', default='models/DBPN_x8.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset), opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'DBPN':
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) ###D-DBPN
    
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()
        with torch.no_grad():
            prediction = model(input)
                
        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, 'result_'+name[0])
        save_img(bicubic.cpu().data, 'bicubic_'+name[0])

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

##Eval Start!!!!
eval()
