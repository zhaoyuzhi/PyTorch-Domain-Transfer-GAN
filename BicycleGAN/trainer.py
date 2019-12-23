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

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_L2 = torch.nn.MSELoss().cuda()

    # Initialize Generator
    G = utils.create_generator(opt)
    D_cVAE = utils.create_discriminator(opt)
    D_cLR = utils.create_discriminator(opt)
    E = utils.create_encoder(opt)

    # To device
    if opt.multi_gpu:
        G = nn.DataParallel(G)
        G = G.cuda()
        D_cVAE = nn.DataParallel(D_cVAE)
        D_cVAE = D_cVAE.cuda()
        D_cLR = nn.DataParallel(D_cLR)
        D_cLR = discriminator_cLR.cuda()
        E = nn.DataParallel(E)
        E = E.cuda()
    else:
        G = G.cuda()
        D_cVAE = D_cVAE.cuda()
        D_cLR = D_cLR.cuda()
        E = E.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D_cVAE = torch.optim.Adam(D_cVAE.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D_cLR = torch.optim.Adam(D_cLR.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_E = torch.optim.Adam(E.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        decay_rate = 1.0 - (max(0, epoch - opt.start_decrease_epoch) // opt.lr_decrease_divide)
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        lr = opt.lr_g * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

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
        for i, (true_input, true_target) in enumerate(dataloader):

            # To device, and seperate data for cVAE_GAN and cLR_GAN
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            cVAE_data = {'img': true_input[[0], :, :, :], 'ground_truth': true_target[[0], :, :, :]}
            cLR_data = {'img': true_input[[1], :, :, :], 'ground_truth': true_target[[1], :, :, :]}

            ''' ----------------------------- 1. Train D ----------------------------- '''
            #############   Step 1. D loss in cVAE-GAN #############

            # Encoded latent vector
            mu, log_variance = E(cVAE_data['ground_truth'])
            std = torch.exp(log_variance / 2)
            random_z = torch.randn(1, opt.z_dim).cuda()
            encoded_z = (random_z * std) + mu

            # Generate fake image
            fake_img_cVAE = G(cVAE_data['img'], encoded_z)

            # Get scores and loss
            real_d_cVAE_1, real_d_cVAE_2 = D_cVAE(cVAE_data['ground_truth'])
            fake_d_cVAE_1, fake_d_cVAE_2 = D_cVAE(fake_img_cVAE.detach())
            
            # mse_loss for LSGAN
            D_loss_cVAE_1 = criterion_L2(real_d_cVAE_1, 1) + criterion_L2(fake_d_cVAE_1, 0)
            D_loss_cVAE_2 = criterion_L2(real_d_cVAE_2, 1) + criterion_L2(fake_d_cVAE_2, 0)
            
            #############   Step 2. D loss in cLR-GAN   #############

            # Random latent vector
            random_z = torch.randn(1, opt.z_dim).cuda()

            # Generate fake image
            fake_img_cLR = G(cLR_data['img'], random_z)

            # Get scores and loss
            real_d_cLR_1, real_d_cLR_2 = D_cLR(cLR_data['ground_truth'])
            fake_d_cLR_1, fake_d_cLR_2 = D_cLR(fake_img_cLR.detach())
            
            D_loss_cLR_1 = criterion_L2(real_d_cLR_1, 1) + criterion_L2(fake_d_cLR_1, 0)
            D_loss_cLR_2 = criterion_L2(real_d_cLR_2, 1) + criterion_L2(fake_d_cLR_2, 0)

            D_loss = D_loss_cVAE_1 + D_loss_cLR_1 + D_loss_cVAE_2 + D_loss_cLR_2

            # Update
            optimizer_D_cVAE.zero_grad()
            optimizer_D_cLR.zero_grad()
            D_loss.backward()
            optimizer_D_cVAE.step()
            optimizer_D_cLR.step()

            ''' ----------------------------- 2. Train G & E ----------------------------- '''
            ############# Step 1. GAN loss to fool discriminator (cVAE_GAN and cLR_GAN) #############

            # Encoded latent vector
            mu, log_variance = E(cVAE_data['ground_truth'])
            std = torch.exp(log_variance / 2)
            random_z = torch.randn(1, opt.z_dim).cuda()
            encoded_z = (random_z * std) + mu

            # Generate fake image and get adversarial loss
            fake_img_cVAE = G(cVAE_data['img'], encoded_z)
            fake_d_cVAE_1, fake_d_cVAE_2 = D_cVAE(fake_img_cVAE)

            GAN_loss_cVAE_1 = criterion_L2(fake_d_cVAE_1, 1)
            GAN_loss_cVAE_2 = criterion_L2(fake_d_cVAE_2, 1)

            # Random latent vector
            random_z = torch.randn(1, opt.z_dim).cuda()

            # Generate fake image and get adversarial loss
            fake_img_cLR = G(cLR_data['img'], random_z)
            fake_d_cLR_1, fake_d_cLR_2 = D_cLR(fake_img_cLR)

            GAN_loss_cLR_1 = criterion_L2(fake_d_cLR_1, 1)
            GAN_loss_cLR_2 = criterion_L2(fake_d_cLR_2, 1)

            G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2
            G_GAN_loss = opt.lambda_gan * G_GAN_loss

            ############# Step 2. KL-divergence with N(0, 1) (cVAE-GAN) #############
            
            KL_div_loss = opt.lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_variance) - log_variance - 1))

            ############# Step 3. Reconstruction of ground truth image (|G(A, z) - B|) (cVAE-GAN) #############
            img_recon_loss = opt.lambda_recon * criterion_L1(fake_img_cVAE, cVAE_data['ground_truth'])

            EG_loss = G_GAN_loss + KL_div_loss + img_recon_loss
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            EG_loss.backward(retain_graph = True)
            optimizer_G.step()
            optimizer_E.step()

            ''' ----------------------------- 3. Train ONLY G ----------------------------- '''
            ############ Step 1. Reconstrution of random latent code (|E(G(A, z)) - z|) (cLR-GAN) ############
            
            # This step should update ONLY G.
            mu_, log_variance_ = E(fake_img_cLR)

            G_alone_loss = opt.lambda_z * criterion_L1(mu_, random_z)

            optimizer_G.zero_grad()
            G_alone_loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [D Loss: %.4f] [GAN Loss: %.4f] [Recon Loss: %.4f] [KL Loss: %.4f] [z Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), D_loss.item(), D_loss.item(), G_GAN_loss.item(), img_recon_loss.item(), KL_div_loss.item(), G_alone_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), G)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), optimizer_D_cVAE)
            adjust_learning_rate(opt, (epoch + 1), optimizer_D_cLR)
            adjust_learning_rate(opt, (epoch + 1), optimizer_E)
