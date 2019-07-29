import os
import numpy as np
import torch
import random

from torch.utils import data
from skimage import io
from skimage import transform
from scipy import misc

# Constants.
num_classes = 2 #coffee
#num_classes = 6 #vaihingen
#num_classes = 6 #grss
root = '/home/datasets/'

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.
class ListDataset(data.Dataset):

    def __init__(self, mode, dataset, task, fold, normalization='default', patch_size = 480, sample='-1', in_channels=3):

        # Initializing variables.
        self.mode = mode
        self.dataset = dataset
        self.task = task
        self.fold = fold
        self.patch_size = patch_size
        self.normalization = normalization
        self.sample = sample
        self.in_channels = in_channels

        # Creating list of paths.
        self.imgs = self.make_dataset()

        # Check for consistency in list.
        if len(self.imgs) == 0:

            raise (RuntimeError('Found 0 images, please check the data set'))

    def trim(self, img, msk):

        tolerance = 0.05 * float(img.max())

        # Mask of non-black pixels (assuming image has a single channel).
        bin = img > tolerance

        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        # Get the contents of the bounding box.
        img_crop = img[x0:x1, y0:y1]
        msk_crop = msk[x0:x1, y0:y1]

        return img_crop, msk_crop


    def make_dataset(self):

        # Making sure the mode is correct.
        assert self.mode in ['train', 'validation', 'test']
        items = []

        # Setting string for the mode.
        mode_str = ''
        if self.mode == 'train':
            if self.sample == '-1':
                mode_str = 'trn'
            else:
                mode_str = 'trn_' + self.sample

        elif self.mode == 'validation':
            mode_str = 'val'

        elif self.mode == 'test':
            mode_str = 'tst'

        # Joining input paths.
        img_path = os.path.join(root, self.dataset, 'images')
        msk_path = os.path.join(root, self.dataset, 'ground_truths', self.task)

        # Reading paths from file.
        data_list = [l.strip('\n') for l in open(os.path.join(root, self.dataset, self.task + '_' + mode_str + '_f' + self.fold + '.txt')).readlines()]

        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(msk_path, it))
            items.append(item)

        # Returning list.
        return items

    """
    def get_patch(self, img, msk):
        if self.mode == 'test' or self.mode == 'validation':
            return img[0:self.patch_size, 0:self.patch_size,:], msk[0:self.patch_size, 0:self.patch_size]
        (ih, iw, _) = img.shape

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)
        img_patch = img[iy:iy+self.patch_size, ix:ix+self.patch_size,:]
        msk_patch = msk[iy:iy+self.patch_size, ix:ix+self.patch_size]

        return img_patch, msk_patch
    """
    

    def __getitem__(self, index):

        # Reading items from list.
        img_path, msk_path = self.imgs[index]

        # Reading images.
        img = misc.imread(img_path)
        msk = misc.imread(msk_path)

        # Removing unwanted channels. For the case of RGB images.
        if self.in_channels == 1:
            if len(img.shape) > 2:
                img = img[:, :, 0]

        if len(msk.shape) > 2:
            msk = msk[:, :, 0]

        #img = transform.resize(img, msk.shape, preserve_range=True)

        if self.in_channels == 1:
            # Trimming borders.
            img, msk = self.trim(img, msk)

        # Casting images to the appropriate dtypes.
        img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        if num_classes == 2:
            msk[msk < 127] = 0
            msk[msk >= 127] = 1
            
        if self.dataset == 'grss_semantic':
            msk = msk-1

        # Normalization.

        if self.normalization == 'statistical':

            if self.in_channels == 1:
                img = (img - img.mean()) / img.std()
            else:
                for i in range(img.shape[2]):
                    img[:,:,i] = (img[:,:,i] - img[:,:,i].mean())

        else:

            if self.in_channels == 1:
                mn = img.min()
                mx = img.max()
                img = ((img - mn) / (mx - mn))
            else:
                for i in range(img.shape[2]):
                    mn = img[:,:,i].min()
                    mx = img[:,:,i].max()
                    img[:,:,i] = ((img[:,:,i] - mn) / (mx - mn))

        # Splitting path.
        spl = img_path.split("/")
        
        #if self.mode == 'train':
        #    img, msk = self.get_patch(img, msk)
        #img, msk = self.get_patch(img, msk)

        # Adding channel dimension.
        if self.in_channels == 1:

            img = np.expand_dims(img, axis=0)

        else:
            tmp = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            tmp[0] = img[:,:,0]
            tmp[1] = img[:,:,1]
            tmp[2] = img[:,:,2]
            img = tmp

        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)

        # Returning to iterator.
        return img, msk, spl[-1]

    def __len__(self):

        return len(self.imgs)
