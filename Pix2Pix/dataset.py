import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NormalRGBDataset(Dataset):
    def __init__(self, opt, imglist):                                   # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = imglist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = self.baseroot + '\\' + self.imglist[index]            # path of one image
        colorimg = Image.open(imgpath)                                  # read one image
        colorimg = colorimg.convert('RGB')
        img = colorimg.crop((256, 0, 512, 256))
        target = colorimg.crop((0, 0, 256, 256))
        img = self.transform(img)
        target = self.transform(target)
        return img, target
    
    def __len__(self):
        return len(self.imglist)
        
class ColorizationDataset(Dataset):
    def __init__(self, opt, imglist):                                   # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = imglist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = self.baseroot + '\\' + self.imglist[index]            # path of one image
        colorimg = Image.open(imgpath)                                  # read one image
        greyimg = colorimg.convert('L').convert('RGB')                  # convert to grey scale, and concat to 3 channels
        colorimg = colorimg.convert('RGB')                              # convert to color RGB
        img = self.transform(greyimg)
        target = self.transform(colorimg)
        return img, target
    
    def __len__(self):
        return len(self.imglist)
