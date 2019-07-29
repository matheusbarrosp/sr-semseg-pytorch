from __future__ import print_function
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from data import get_training_set, get_eval_set, get_validation_set
#import pdb
import socket
import time
from math import log10

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')
parser.add_argument('--input_dir', type=str, default='Input', help='Test dir')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--test_dataset', type=str, default='Set5_LR_x8')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def test():
    model.eval()
    avg_psnr = 0
    avg_psnr_bicubic = 0
    
    for iteration, batch in enumerate(testing_data_loader, 1):
        with torch.no_grad():
            input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])
        optimizer.zero_grad()
        t0 = time.time()
        with torch.no_grad():
            prediction = model(input)
                
        if opt.residual:
            prediction = prediction + bicubic
        mse = criterion_psnr(prediction, target)
        psnr = 10 * log10(1 / mse.data)
        avg_psnr += psnr
        mse = criterion_psnr(bicubic, target)
        psnr = 10 * log10(1 / mse.data)
        avg_psnr_bicubic += psnr
        
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. PSNR Bicubic: {:.4f} dB".format(avg_psnr_bicubic / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)

    
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = get_validation_set(opt.input_dir, opt.test_dataset, opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()
criterion_psnr = nn.MSELoss()

#print('---------- Networks architecture -------------')
#print_network(model)
#print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    print(model_name)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])
    criterion_psnr = criterion_psnr.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

best_psnr = None
best_model = None
epoch_since_best = -1
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)
    current_psnr = test()
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch) % (opt.snapshots) == 0:
        checkpoint(epoch)

    if best_psnr == None or current_psnr > best_psnr:
        epoch_since_best = -1
        print('New best PSNR found: {}'.format(current_psnr))
        best_model = model.state_dict().copy()
        best_psnr = current_psnr
    epoch_since_best += 1
    if epoch_since_best == 30:
        torch.save(best_model, opt.save_folder+"tmp_best_{0}.pth".format(best_psnr))        

best_psnr = str(best_psnr).replace('.','-')
torch.save(best_model, opt.save_folder+"best_{0}.pth".format(best_psnr))
