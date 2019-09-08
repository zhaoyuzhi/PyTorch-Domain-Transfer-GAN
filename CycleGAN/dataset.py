import os
import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

class NormalRGBDataset(Dataset):
    def __init__(self, opt):                                                # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = os.path.join(self.baseroot, self.imglist[index])          # path of one image
        colorimg = Image.open(imgpath)                                      # read one image
        colorimg = colorimg.convert('RGB')
        img = colorimg.crop((256, 0, 512, 256))
        target = colorimg.crop((0, 0, 256, 256))
        img = self.transform(img)
        target = self.transform(target)
        return img, target
    
    def __len__(self):
        return len(self.imglist)
        
class ColorizationDataset(Dataset):
    def __init__(self, opt):                                                # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = os.path.join(self.baseroot, self.imglist[index])          # path of one image
        colorimg = Image.open(imgpath)                                      # read one image
        greyimg = colorimg.convert('L').convert('RGB')                      # convert to grey scale, and concat to 3 channels
        colorimg = colorimg.convert('RGB')                                  # convert to color RGB
        img = self.transform(greyimg)
        target = self.transform(colorimg)
        return img, target
    
    def __len__(self):
        return len(self.imglist)

class DomainTransferDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A)
        self.imglist_B = utils.get_files(opt.baseroot_B)
        self.len_A = len(self.imglist_A)
        self.len_B = len(self.imglist_B)

    def img_aug(self, img):
        # random scale
        '''
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size and remain the ratio of H to W
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        '''
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            if H_in < W_in:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_in < H_in
                W_out = self.opt.crop_size
                H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        else:
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        # random crop
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))  
        img = cropper(img)
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode = 1)
        return img

    def __getitem__(self, index):
        ## Image A
        random_A = random.randint(0, self.len_A - 1)
        imgpath_A = self.imglist_A[random_A]
        img_A = cv2.imread(imgpath_A, cv2.IMREAD_GRAYSCALE)
        img_A = np.expand_dims(img_A, 2)
        # normalized to [-1, 1]
        img_A = (img_A.astype(np.float64) - 128) / 128
        # data augmentation
        img_A = self.img_aug(img_A)
        # to PyTorch Tensor
        img_A = np.expand_dims(img_A, 2)
        img_A = torch.from_numpy(img_A.transpose(2, 0, 1).astype(np.float32)).contiguous()
        ## Image B
        random_B = random.randint(0, self.len_B - 1)
        imgpath_B = self.imglist_B[random_B]
        img_B = cv2.imread(imgpath_B, cv2.IMREAD_GRAYSCALE)
        img_B = np.expand_dims(img_B, 2)
        # normalized to [-1, 1]
        img_B = (img_B.astype(np.float64) - 128) / 128
        # data augmentation
        img_B = self.img_aug(img_B)
        # to PyTorch Tensor
        img_B = np.expand_dims(img_B, 2)
        img_B = torch.from_numpy(img_B.transpose(2, 0, 1).astype(np.float32)).contiguous()
        return img_A, img_B
    
    def __len__(self):
        return min(self.len_A, self.len_B)
