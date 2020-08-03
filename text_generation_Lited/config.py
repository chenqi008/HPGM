import argparse
import torch.nn as nn
import datetime
import os
import torch.nn.functional as F
import torch
# from utils import compute_mean_covariance
parser = argparse.ArgumentParser()

# data path and loading parameters
parser.add_argument('--texturePath', required=True, help='path to texture image folder')
parser.add_argument('--contentPath', default='', help='path to content image folder')
parser.add_argument('--mirror', type=bool, default=False,help='augment style image distribution for mirroring')
parser.add_argument('--contentScale', type=float, default=1.0,help='scale content images')
parser.add_argument('--textureScale', type=float, default=1.0,help='scale texture images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#0 means a single main process
parser.add_argument('--outputFolder', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--trainOverfit', type=bool, default=False,help='always use same image and same templates -- better in sample, worse out of sample')
# neural network parameters
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--imageSize', type=int, default=160, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=80, help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=80, help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--nDep', type=int, default=5, help='depth of Unet Generator')
parser.add_argument('--nDepD', type=int, default=5, help='depth of DiscrimblendMoinator')
parser.add_argument('--N', type=int, default=30, help='count of memory templates')
parser.add_argument('--coordCopy', type=bool, default=True,help='copy  x,y coordinates of cropped memory template')
parser.add_argument('--multiScale', type=bool, default=False,help='multi-scales of mixing features; if False only full resolution; if True all levels')
parser.add_argument('--nBlocks', type=int, default=0,help='additional res blocks for complexity in the unet')
parser.add_argument('--blendMode', type=int, default=0,help='type of blending for parametric/nonparametric output')
parser.add_argument('--refine', type=bool, default=False,help='second unet after initial templates')
parser.add_argument('--skipConnections', type=bool, default=True,help='skip connections in  Unet -- allows better content reconstruct')
parser.add_argument('--Ubottleneck', type=int, default=-1,help='Unet bottleneck, leave negative for default wide bottleneck')
# regularization and loss criteria weighting parameters
parser.add_argument('--fContent', type=float, default=1.0,help='weight of content reconstruction loss')
parser.add_argument('--fAdvM', type=float, default=.0,help='weight of I_M adversarial loss')
parser.add_argument('--fContentM', type=float, default=1.0,help='weight of I_M content reconstruction loss')
parser.add_argument('--cLoss', type=int, default=0,help='type of perceptual distance metric for reconstruction loss')
parser.add_argument('--fAlpha', type=float, default=.1,help='regularization weight of norm of blending mask')
parser.add_argument('--fTV', type=float, default=.1,help='regularization weight of total variation of blending mask')
parser.add_argument('--fEntropy', type=float, default=.5,help='regularization weight of entropy -- forcing low entropy results in 0/1 values in mix tensor A')
parser.add_argument('--fDiversity', type=float, default=1,help='regularization weight of diversity of used templates')
parser.add_argument('--WGAN', type=bool, default=False,help='use WGAN-GP adversarial loss')
parser.add_argument('--LS', type=bool, default=False,help='use least squares GAN adversarial loss')
# optimisation parametersfalp
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')
# noise parameters
parser.add_argument('--zGL', type=int, default=31, help='noise channels, identical on every spatial position')
parser.add_argument('--zLoc', type=int, default=100, help='noise channels, sampled on each spatial position')
parser.add_argument('--zPeriodic', type=int, default=0, help='periodic spatial waves')
parser.add_argument('--firstNoise', type=bool, default=False, help='stochastic noise at bottleneck or input of Unet')
# other parameters
parser.add_argument('--z_material', type=int, default=19, help='material channels')
parser.add_argument('--z_color', type=int, default=12, help='color channels')
parser.add_argument('--netG', default='', help='path to pretrained netG model')
parser.add_argument('--netD', default='', help='path to pretrained netD model')
parser.add_argument('--netE', default='', help='path to pretrained netE model')
parser.add_argument('--train', type=int, default=0, help='test(0), train(1), save_embedding(2), interpolation(3)')
# parser.add_argument('--learnedWN', default='', help='path to pretrained learnedWN model')
parser.add_argument('--use_perceptual_loss', type=int, default=1, help='no(0), yes(1)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the training set?')
parser.add_argument('--coeff_color_loss', type=float, default=0, help='the weight of color loss')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
opt = parser.parse_args()

nDep = opt.nDep
## noise added to the deterministic content mosaic modules -- in some cases it makes a difference, other times can be ignored
bfirstNoise = opt.firstNoise
nz = opt.zGL+opt.zLoc+opt.zPeriodic
# nz=opt.zLoc
bMirror = opt.mirror ##make for a richer distribution, 4x times more data
opt.fContentM *= opt.fContent

##GAN criteria changes given loss options LS or WGAN
if not opt.WGAN and not opt.LS:
    criterion = nn.BCELoss()
elif opt.LS:
    def crit(x, l):
        return ((x-l)**2).mean()
    criterion = crit
else:
    def dummy(val, label):
        return (val*(1-2*label)).mean()#so -1 fpr real. 1 fpr fake
    criterion = dummy


# criterion of material and color
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, )->(n, h, w)
    n, c, h, w = input.size()
    target = target.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.float().sum()
    return loss

# Reconstruction + KL divergence losses summed over all elements and batch
# the sizes of mu and logvar are 8x300x5x5
def kl_reconstruction_loss(recon_x, x, mu, logvar, opt):
    from utils import compute_mean_covariance
    MSE = F.mse_loss(recon_x, x)
    # color-aware loss
    mu1, covariance1 = compute_mean_covariance(recon_x)
    mu2, covariance2 = compute_mean_covariance(x)
    like_mu2 = F.mse_loss(mu1, mu2)
    like_cov2 = 5 * F.mse_loss(covariance1, covariance2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # assert False

    return like_mu2, like_cov2, KLD, MSE


material_criterion = cross_entropy2d
color_criterion = cross_entropy2d

perceptual_criterion = nn.MSELoss()


if opt.outputFolder == '.':
    i = opt.texturePath[:-1].rfind('/')
    i2 = opt.contentPath[:-1].rfind('/')
    opt.outputFolder = "results/"+opt.texturePath[i+1:]+opt.contentPath[i2+1:]##actually 2 nested folders -- cool
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.outputFolder += stamp + "/"
try:
    # os.makedirs(opt.outputFolder)
    os.makedirs(os.path.join(opt.outputFolder, 'train'))
    os.makedirs(os.path.join(opt.outputFolder, 'eval'))
    os.makedirs(os.path.join(opt.outputFolder, 'test'))
except OSError:
    pass
print ("outputFolderolder "+opt.outputFolder)

if opt.train == 1:
    text_file = open(opt.outputFolder+"options.txt", "w")
    text_file.write(str(opt))
    text_file.close()
    print(opt)
