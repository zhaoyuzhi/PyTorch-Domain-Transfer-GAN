import argparse
import os

### Note that, the author of DRIT points out the difference between MUNIT and DRIT is how to connect content code and style code
# MUNIT: AdaIN
# DRIT: for color transfer, concatenation; for shape transfer, elemernt-wise transformation
# In this implemention, we use concatenation
if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = 'Pre_CAGAN_epoch1_bs16', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0002, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'iter', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 1, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 200000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_id', type = float, default = 5, help = 'coefficient for ID Loss')
    parser.add_argument('--lambda_content', type = float, default = 0.01, help = 'coefficient for content Loss')
    parser.add_argument('--lambda_style', type = float, default = 0.01, help = 'coefficient for style Loss')
    parser.add_argument('--lambda_cycle', type = float, default = 0.01, help = 'coefficient for cycle Loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.01, help = 'coefficient for gan Loss')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm_content', type = str, default = 'in', help = 'normalization type of content encoder')
    parser.add_argument('--norm_decoder', type = str, default = 'ln', help = 'normalization type of decoder')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--in_dim', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_dim', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_dim', type = int, default = 64, help = 'bottommost channels for the main stream of generator')
    parser.add_argument('--mlp_dim', type = int, default = 256, help = 'the channels of latent MLP layers')
    parser.add_argument('--n_down_content', type = int, default = 2, help = 'down sample blocks for the content encoder')
    parser.add_argument('--n_down_style', type = int, default = 4, help = 'down sample blocks for the style encoder')
    parser.add_argument('--n_res', type = int, default = 4, help = 'number of res blocks for the generator')
    parser.add_argument('--n_mlp', type = int, default = 3, help = 'number of linear blocks for the style mlp')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'WGAN', help = 'type of GAN: [LSGAN | WGAN], WGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\ILSVRC2012_train_256\\', help = 'color image baseroot')
    parser.add_argument('--dataset_name_a', type = str, default = 'cat', help = 'the folder name of the a domain')
    parser.add_argument('--dataset_name_b', type = str, default = 'human', help = 'the folder name of the b domain')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'the image size')
    opt = parser.parse_args()
    
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    import trainer
    print('The MUNIT settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Saving mode: %s] [GAN_mode: %s]'
        % (opt.epochs, opt.batch_size, opt.lr_g, opt.save_mode, opt.gan_mode))
    print('[lambda_id: %.2f] [lambda_content: %.2f] [lambda_style: %.2f] [lambda_cycle: %.2f] [lambda_gan: %.2f]'
        % (opt.lambda_id, opt.lambda_content, opt.lambda_style, opt.lambda_cycle, opt.lambda_gan))
    if opt.gan_mode == 'LSGAN':
        trainer.Trainer_LSGAN(opt)
    if opt.gan_mode == 'WGAN':
        trainer.Trainer_WGAN(opt)
