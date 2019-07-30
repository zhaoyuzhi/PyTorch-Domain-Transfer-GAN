import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

import network
import dataset

# There are many functions:
# ----------------------------------------
# 1. text_readlines:
# In: a str nominating the a txt
# Parameters: None
# Out: list
# ----------------------------------------
# 2. create_generator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: colorizationnet
# ----------------------------------------
# 3. create_discriminator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: discriminator_coarse_color, discriminator_coarse_sal, discriminator_fine_color, discriminator_fine_sal
# ----------------------------------------
# 4. create_perceptualnet:
# In: None
# Parameters: None
# Out: perceptualnet
# ----------------------------------------
# 5. load_dict
# In: process_net (the net needs update), pretrained_net (the net has pre-trained dict)
# Out: process_net (updated)
# ----------------------------------------
# 6. create_dataloader
# In: opt
# Out: dataloader for MUNIT
# ----------------------------------------
# 7. savetxt
# In: list
# Out: txt
# ----------------------------------------
# 8. get_files
# In: path
# Out: txt
# ----------------------------------------
# 9. get_names
# In: path
# Out: txt
# ----------------------------------------
# 10. text_save
# In: list
# Out: txt
# ----------------------------------------

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def create_generator(opt):
    if opt.pre_train == True:
        # Initialize the network
        generator_a = network.Generator(opt)
        generator_b = network.Generator(opt)
        # Init the network
        network.weights_init(generator_a, init_type = opt.init_type, init_gain = opt.init_gain)
        network.weights_init(generator_b, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Load the weights
        generator_a = torch.load(opt.load_name + '_a.pth')
        generator_b = torch.load(opt.load_name + '_b.pth')
        print('Generator is loaded!')
    return generator_a, generator_b

def create_discriminator(opt):
    # Initialize the network
    discriminator_a = network.PatchDiscriminator70(opt)
    discriminator_b = network.PatchDiscriminator70(opt)
    # Init the network
    network.weights_init(discriminator_a, init_type = opt.init_type, init_gain = opt.init_gain)
    network.weights_init(discriminator_b, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator_a, discriminator_b
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def create_dataloader(opt):
    # Define the image list
    imglist_a = get_files(opt.baseroot + opt.dataset_name_a)
    imglist_b = get_files(opt.baseroot + opt.dataset_name_b)
    # Define the dataset
    trainset = dataset.UnpairImageDataset(opt, imglist_a, imglist_b)
    print('The overall number of images:', len(trainset))
    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    return dataloader
    
def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()
