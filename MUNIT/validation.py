import argparse
import os
import numpy as np
import cv2
import torch

import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--load_name', type = str, default = 'WGAN_MUNIT_epoch200_bs8', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--save_path', type = str, default = 'results', help = 'images saving path')
    # Training parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './', help = 'color image baseroot')
    parser.add_argument('--dataset_name_a', type = str, default = 'cat_test', help = 'the folder name of the a domain')
    parser.add_argument('--dataset_name_b', type = str, default = 'human_test', help = 'the folder name of the b domain')
    parser.add_argument('--imgsize', type = int, default = 128, help = 'the image size')
    opt = parser.parse_args()

    utils.check_path(opt.save_path)

    # Define the dataset
    # a = 'cat'; b = 'human'
    testloader = utils.create_dataloader(opt)
    print('The overall number of images:', len(testloader))

    # Define networks
    generator_a, generator_b = utils.create_generator(opt)
    generator_a = generator_a.cuda()
    generator_b = generator_b.cuda()

    # Forward
    for i, (img_a, img_b) in enumerate(testloader):
        # To device
        img_a = img_a.cuda()
        img_b = img_b.cuda()
        # Forward
        with torch.no_grad():
            out = generator_b(img_a, img_a)
        out = out.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
        out = (out + 1) * 128
        out = out.astype(np.uint8)[:, :, [2, 1, 0]]
        # Save
        imgname = os.path.join(opt.save_path, str(i) + '.png')
        cv2.imwrite(imgname, out)
    
