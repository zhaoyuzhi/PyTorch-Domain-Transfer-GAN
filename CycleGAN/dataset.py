import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

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
        self.baseroot_A = opt.baseroot_A
        self.baseroot_B = opt.baseroot_B
        self.imglist_A = utils.get_jpgs(opt.baseroot_A)
        self.imglist_B = utils.get_jpgs(opt.baseroot_B)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath_A = os.path.join(self.baseroot_A, self.imglist_A[index])    # path of one image
        img_A = Image.open(imgpath_A).convert('RGB')                        # read one image
        img_A = self.transform(img_A)
        imgpath_B = os.path.join(self.baseroot_B, self.imglist_B[index])    # path of one image
        img_B = Image.open(imgpath_B).convert('RGB')                        # read one image
        img_B = self.transform(img_B)
        return img_A, img_B
    
    def __len__(self):
        return len(self.imglist_A)
