import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage import color
import random

class SingleImageDataset(Dataset):
    def __init__(self, opt, imglist):                                   # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = imglist
        self.transform = transforms.ToTensor()
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_lab(self, imgpath):
        img = Image.open(imgpath)                                       # read one image
        # pre-processing, let all the images are in RGB color space
        img = img.resize((256, 256), Image.ANTIALIAS).convert('RGB')    # PIL Image RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img = np.array(img)                                             # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        # convert RGB to Lab, finally get Tensor
        img = color.rgb2lab(img).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        img = self.transform(img)                                       # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        l = img[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = img[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        return l, ab

    def get_rgb(self, imgpath):
        img = Image.open(imgpath)                                       # read one image
        # pre-processing, let all the images are in RGB color space
        img = img.resize((256, 256), Image.ANTIALIAS).convert('RGB')    # PIL Image RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        l = img.convert('L')                                            # PIL Image L: L [0, 255], order [H, W]
        # normalization
        l = self.transform_gray(l)                                      # L, normalized to [-1, 1]
        rgb = self.transform_rgb(img)                                   # rgb, normalized to [-1, 1]
        return l, rgb

    def __getitem__(self, index):
        imgpath = self.baseroot + self.imglist[index]                   # path of one image
        l, target = self.get_rgb(imgpath)
        return l, target
    
    def __len__(self):
        return len(self.imglist)

class UnpairImageDataset(Dataset):
    def __init__(self, opt, imglist_a, imglist_b):                      # root: list ; transform: torch transform
        self.imglist_a = imglist_a
        self.imglist_b = imglist_b
        self.transform = transforms.Compose([
            transforms.Resize((opt.imgsize, opt.imgsize), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.len_a = len(imglist_a)
        self.len_b = len(imglist_b)

    def __getitem__(self, index):
        # Randomly choose the image pairs
        randint_a = random.randint(0, self.len_a - 1)
        randint_b = random.randint(0, self.len_b - 1)
        imgpath_a = self.imglist_a[randint_a]
        imgpath_b = self.imglist_b[randint_b]
        # Read image pairs
        img_a = Image.open(imgpath_a).convert('RGB')
        img_a = self.transform(img_a)
        img_b = Image.open(imgpath_b).convert('RGB')
        img_b = self.transform(img_b)
        # The output is in range [-1, 1]
        return img_a, img_b
    
    def __len__(self):
        return min(len(self.imglist_a), len(self.imglist_b))
