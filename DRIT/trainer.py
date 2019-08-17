import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.backends.cudnn as cudnn

import utils

def Trainer_LSGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    generator_a, generator_b = utils.create_generator(opt)
    discriminator_a, discriminator_b = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        generator_a = nn.DataParallel(generator_a)
        generator_a = generator_a.cuda()
        generator_b = nn.DataParallel(generator_b)
        generator_b = generator_b.cuda()
        discriminator_a = nn.DataParallel(discriminator_a)
        discriminator_a = discriminator_a.cuda()
        discriminator_b = nn.DataParallel(discriminator_b)
        discriminator_b = discriminator_b.cuda()
    else:
        generator_a = generator_a.cuda()
        generator_b = generator_b.cuda()
        discriminator_a = discriminator_a.cuda()
        discriminator_b = discriminator_b.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator_a.parameters(), generator_b.parameters()),
        lr = opt.lr_g, betas = (opt.b1, opt.b2),
        weight_decay = opt.weight_decay
    )
    optimizer_D_a = torch.optim.Adam(discriminator_a.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    optimizer_D_b = torch.optim.Adam(discriminator_b.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
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
    def save_model(opt, epoch, iteration, len_dataset, generator_a, generator_b):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator_a.module, 'LSGAN_DRIT_epoch%d_bs%d_a.pth' % (epoch, opt.batch_size))
                        torch.save(generator_b.module, 'LSGAN_DRIT_epoch%d_bs%d_b.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator_a.module, 'LSGAN_DRIT_iter%d_bs%d_a.pth' % (iteration, opt.batch_size))
                        torch.save(generator_b.module, 'LSGAN_DRIT_iter%d_bs%d_b.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator_a, 'LSGAN_DRIT_epoch%d_bs%d_a.pth' % (epoch, opt.batch_size))
                        torch.save(generator_b, 'LSGAN_DRIT_epoch%d_bs%d_b.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator_a, 'LSGAN_DRIT_iter%d_bs%d_a.pth' % (iteration, opt.batch_size))
                        torch.save(generator_b, 'LSGAN_DRIT_iter%d_bs%d_b.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    dataloader = utils.create_dataloader(opt)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (img_a, img_b) in enumerate(dataloader):

            # To device
            img_a = img_a.cuda()
            img_b = img_b.cuda()

            # Sampled style codes (prior)
            prior_s_a = Tensor(torch.randn(img_a.shape[0], opt.style_dim))
            prior_s_b = Tensor(torch.randn(img_a.shape[0], opt.style_dim))

            # Adversarial ground truth
            valid = Tensor(np.ones((img_a.shape[0], 1, 30, 30)))
            fake = Tensor(np.zeros((img_a.shape[0], 1, 30, 30)))

            # ----------------------------------------
            #              Train Generator
            # ----------------------------------------
            # Note that:
            # input / output image dimension: [B, 3, 256, 256]
            # content_code dimension: [B, 256, 64, 64]
            # style_code dimension: [B, 8]
            # generator_a is related to domain a / style a
            # generator_b is related to domain b / style b
            
            optimizer_G.zero_grad()

            # Get shared latent representation
            c_a, s_a = generator_a.encode(img_a)
            c_b, s_b = generator_b.encode(img_b)
            
            # Reconstruct images
            img_aa_recon = generator_a.decode(c_a, s_a)
            img_bb_recon = generator_b.decode(c_b, s_b)
            
            # Translate images
            img_ba = generator_a.decode(c_b, prior_s_a)
            img_ab = generator_b.decode(c_a, prior_s_b)

            # Cycle code translation
            c_b_recon, s_a_recon = generator_a.encode(img_ba)
            c_a_recon, s_b_recon = generator_b.encode(img_ab)

            # Cycle image translation
            img_aa_recon_cycle = generator_a.decode(c_a_recon, s_a) if opt.lambda_cycle > 0 else 0
            img_bb_recon_cycle = generator_b.decode(c_b_recon, s_b) if opt.lambda_cycle > 0 else 0

            # Losses
            loss_id_1 = opt.lambda_id * criterion_L1(img_aa_recon, img_a)
            loss_id_2 = opt.lambda_id * criterion_L1(img_bb_recon, img_b)
            loss_s_1 = opt.lambda_style * criterion_L1(s_a_recon, prior_s_a)
            loss_s_2 = opt.lambda_style * criterion_L1(s_b_recon, prior_s_b)
            loss_c_1 = opt.lambda_content * criterion_L1(c_a_recon, c_a.detach())
            loss_c_2 = opt.lambda_content * criterion_L1(c_b_recon, c_b.detach())
            loss_cycle_1 = opt.lambda_cycle * criterion_L1(img_aa_recon_cycle, img_a) if opt.lambda_cycle > 0 else 0
            loss_cycle_2 = opt.lambda_cycle * criterion_L1(img_bb_recon_cycle, img_b) if opt.lambda_cycle > 0 else 0

            # GAN Loss
            fake_scalar_a = discriminator_a(img_ba)
            fake_scalar_b = discriminator_b(img_ab)
            loss_gan1 = opt.lambda_gan * criterion_MSE(fake_scalar_a, valid)
            loss_gan2 = opt.lambda_gan * criterion_MSE(fake_scalar_b, valid)

            # Overall Losses and optimization
            loss_G = loss_id_1 + loss_id_2 + loss_s_1 + loss_s_2 + loss_c_1 + loss_c_2 + loss_cycle_1 + loss_cycle_2 + loss_gan1 + loss_gan2
            loss_G.backward()
            optimizer_G.step()

            # ----------------------------------------
            #            Train Discriminator
            # ----------------------------------------
            
            optimizer_D_a.zero_grad()
            optimizer_D_b.zero_grad()

            # D_a
            fake_scalar_a = discriminator_a(img_ba.detach())
            true_scalar_a = discriminator_a(img_a)
            loss_D_a = 0.5 * (criterion_MSE(fake_scalar_a, fake) + criterion_MSE(true_scalar_a, valid))
            loss_D_a.backward()
            optimizer_D_a.step()
            
            # D_b
            fake_scalar_b = discriminator_b(img_ab.detach())
            true_scalar_b = discriminator_b(img_b)
            loss_D_b = 0.5 * (criterion_MSE(fake_scalar_b, fake) + criterion_MSE(true_scalar_b, valid))
            loss_D_b.backward()
            optimizer_D_b.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.4f] [Style Loss: %.4f] [Content Loss: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), (loss_id_1 + loss_id_2).item(),
                (loss_s_1 + loss_s_2).item(), (loss_c_1 + loss_c_2).item(),
                (loss_gan1 + loss_gan2).item(), (loss_D_a + loss_D_b).item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator_a, generator_b)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_a)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_b)
            
def Trainer_WGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator_a, generator_b = utils.create_generator(opt)
    discriminator_a, discriminator_b = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        generator_a = nn.DataParallel(generator_a)
        generator_a = generator_a.cuda()
        generator_b = nn.DataParallel(generator_b)
        generator_b = generator_b.cuda()
        discriminator_a = nn.DataParallel(discriminator_a)
        discriminator_a = discriminator_a.cuda()
        discriminator_b = nn.DataParallel(discriminator_b)
        discriminator_b = discriminator_b.cuda()
    else:
        generator_a = generator_a.cuda()
        generator_b = generator_b.cuda()
        discriminator_a = discriminator_a.cuda()
        discriminator_b = discriminator_b.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator_a.parameters(), generator_b.parameters()),
        lr = opt.lr_g, betas = (opt.b1, opt.b2),
        weight_decay = opt.weight_decay
    )
    optimizer_D_a = torch.optim.Adam(discriminator_a.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    optimizer_D_b = torch.optim.Adam(discriminator_b.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
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
    def save_model(opt, epoch, iteration, len_dataset, generator_a, generator_b):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator_a.module, 'WGAN_DRIT_epoch%d_bs%d_a.pth' % (epoch, opt.batch_size))
                        torch.save(generator_b.module, 'WGAN_DRIT_epoch%d_bs%d_b.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator_a.module, 'WGAN_DRIT_iter%d_bs%d_a.pth' % (iteration, opt.batch_size))
                        torch.save(generator_b.module, 'WGAN_DRIT_iter%d_bs%d_b.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator_a, 'WGAN_DRIT_epoch%d_bs%d_a.pth' % (epoch, opt.batch_size))
                        torch.save(generator_b, 'WGAN_DRIT_epoch%d_bs%d_b.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator_a, 'WGAN_DRIT_iter%d_bs%d_a.pth' % (iteration, opt.batch_size))
                        torch.save(generator_b, 'WGAN_DRIT_iter%d_bs%d_b.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    dataloader = utils.create_dataloader(opt)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (img_a, img_b) in enumerate(dataloader):

            # To device
            img_a = img_a.cuda()
            img_b = img_b.cuda()

            # Sampled style codes (prior)
            prior_s_a = torch.randn(img_a.shape[0], opt.style_dim).cuda()
            prior_s_b = torch.randn(img_a.shape[0], opt.style_dim).cuda()

            # ----------------------------------------
            #              Train Generator
            # ----------------------------------------
            # Note that:
            # input / output image dimension: [B, 3, 256, 256]
            # content_code dimension: [B, 256, 64, 64]
            # style_code dimension: [B, 8]
            # generator_a is related to domain a / style a
            # generator_b is related to domain b / style b
            
            optimizer_G.zero_grad()

            # Get shared latent representation
            c_a, s_a = generator_a.encode(img_a)
            c_b, s_b = generator_b.encode(img_b)
            
            # Reconstruct images
            img_aa_recon = generator_a.decode(c_a, s_a)
            img_bb_recon = generator_b.decode(c_b, s_b)
            
            # Translate images
            img_ba = generator_a.decode(c_b, prior_s_a)
            img_ab = generator_b.decode(c_a, prior_s_b)

            # Cycle code translation
            c_b_recon, s_a_recon = generator_a.encode(img_ba)
            c_a_recon, s_b_recon = generator_b.encode(img_ab)

            # Cycle image translation
            img_aa_recon_cycle = generator_a.decode(c_a_recon, s_a) if opt.lambda_cycle > 0 else 0
            img_bb_recon_cycle = generator_b.decode(c_b_recon, s_b) if opt.lambda_cycle > 0 else 0

            # Losses
            loss_id_1 = opt.lambda_id * criterion_L1(img_aa_recon, img_a)
            loss_id_2 = opt.lambda_id * criterion_L1(img_bb_recon, img_b)
            loss_s_1 = opt.lambda_style * criterion_L1(s_a_recon, prior_s_a)
            loss_s_2 = opt.lambda_style * criterion_L1(s_b_recon, prior_s_b)
            loss_c_1 = opt.lambda_content * criterion_L1(c_a_recon, c_a.detach())
            loss_c_2 = opt.lambda_content * criterion_L1(c_b_recon, c_b.detach())
            loss_cycle_1 = opt.lambda_cycle * criterion_L1(img_aa_recon_cycle, img_a) if opt.lambda_cycle > 0 else 0
            loss_cycle_2 = opt.lambda_cycle * criterion_L1(img_bb_recon_cycle, img_b) if opt.lambda_cycle > 0 else 0

            # GAN Loss
            fake_scalar_a = discriminator_a(img_ba)
            fake_scalar_b = discriminator_b(img_ab)
            loss_gan1 = - opt.lambda_gan * torch.mean(fake_scalar_a)
            loss_gan2 = - opt.lambda_gan * torch.mean(fake_scalar_b)

            # Overall Losses and optimization
            loss_G = loss_id_1 + loss_id_2 + loss_s_1 + loss_s_2 + loss_c_1 + loss_c_2 + loss_cycle_1 + loss_cycle_2 + loss_gan1 + loss_gan2
            loss_G.backward()
            optimizer_G.step()

            # ----------------------------------------
            #            Train Discriminator
            # ----------------------------------------
            
            optimizer_D_a.zero_grad()
            optimizer_D_b.zero_grad()

            # D_a
            fake_scalar_a = discriminator_a(img_ba.detach())
            true_scalar_a = discriminator_a(img_a)
            loss_D_a = torch.mean(fake_scalar_a) - torch.mean(true_scalar_a)
            loss_D_a.backward()
            optimizer_D_a.step()
            
            # D_b
            fake_scalar_b = discriminator_b(img_ab.detach())
            true_scalar_b = discriminator_b(img_b)
            loss_D_b = torch.mean(fake_scalar_b) - torch.mean(true_scalar_b)
            loss_D_b.backward()
            optimizer_D_b.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.4f] [Style Loss: %.4f] [Content Loss: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), (loss_id_1 + loss_id_2).item(),
                (loss_s_1 + loss_s_2).item(), (loss_c_1 + loss_c_2).item(),
                (loss_gan1 + loss_gan2).item(), (loss_D_a + loss_D_b).item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator_a, generator_b)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_a)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_b)
