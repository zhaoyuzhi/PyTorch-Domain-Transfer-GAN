import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def CycleGAN_LSGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    # A is for grayscale image
    # B is for color RGB image
    G_AB = utils.create_generator(opt)
    G_BA = utils.create_generator(opt)
    D_A = utils.create_discriminator(opt)
    D_B = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        G_AB = nn.DataParallel(G_AB)
        G_AB = G_AB.cuda()
        G_BA = nn.DataParallel(G_BA)
        G_BA = G_BA.cuda()
        D_A = nn.DataParallel(D_A)
        D_A = D_A.cuda()
        D_B = nn.DataParallel(D_B)
        D_B = D_B.cuda()
    else:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, G_AB, G_BA):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(G_AB.module, 'G_AB_LSGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        torch.save(G_BA.module, 'G_BA_LSGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(G_AB.module, 'G_AB_LSGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        torch.save(G_BA.module, 'G_BA_LSGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(G_AB, 'G_AB_LSGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        torch.save(G_BA, 'G_BA_LSGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(G_AB, 'G_AB_LSGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        torch.save(G_BA, 'G_BA_LSGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DomainTransferDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_A, true_B) in enumerate(dataloader):

            # To device
            # A is for grayscale image
            # B is for color RGB image
            true_A = true_A.cuda()
            true_B = true_B.cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((true_A.shape[0], 1, 16, 16)))
            fake = Tensor(np.zeros((true_A.shape[0], 1, 16, 16)))

            # Train Generator
            optimizer_G.zero_grad()

            # Indentity Loss
            loss_indentity_A = criterion_L1(G_BA(true_A), true_A)
            loss_indentity_B = criterion_L1(G_AB(true_B), true_B)
            loss_indentity = (loss_indentity_A + loss_indentity_B) / 2

            # GAN Loss
            fake_B = G_AB(true_A)
            loss_GAN_AB = criterion_MSE(D_B(fake_B), valid)
            fake_A = G_BA(true_B)
            loss_GAN_BA = criterion_MSE(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle-consistency Loss
            recon_A = G_BA(fake_B)
            loss_cycle_A = criterion_L1(recon_A, true_A)
            recon_B = G_AB(fake_A)
            loss_cycle_B = criterion_L1(recon_B, true_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Overall Loss and optimize
            loss = loss_GAN + opt.lambda_cycle * loss_cycle + opt.lambda_identity * loss_indentity
            loss.backward()
            optimizer_G.step()

            # Train Discriminator A
            optimizer_D_A.zero_grad()

            # Fake samples
            fake_scalar_d = D_A(fake_A.detach())
            loss_fake = criterion_MSE(fake_scalar_d, fake)

            # True samples
            true_scalar_d = D_A(true_A)
            loss_true = criterion_MSE(true_scalar_d, valid)
            
            # Overall Loss and optimize
            loss_D_A = 0.5 * (loss_fake + loss_true)
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()

            # Fake samples
            fake_scalar_d = D_B(fake_B.detach())
            loss_fake = criterion_MSE(fake_scalar_d, fake)

            # True samples
            true_scalar_d = D_B(true_B)
            loss_true = criterion_MSE(true_scalar_d, valid)
            
            # Overall Loss and optimize
            loss_D_B = 0.5 * (loss_fake + loss_true)
            loss_D_B.backward()
            optimizer_D_B.step()
    
            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [D_A Loss: %.4f] [D_B Loss: %.4f] [G GAN Loss: %.4f] [G Cycle Loss: %.4f] [G Indentity Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_D_A.item(), loss_D_B.item(), loss_GAN.item(), loss_cycle.item(), loss_indentity.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), G_AB, G_BA)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_A)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_B)
