from __future__ import print_function
import os
import random
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from util.maskgradientloss import MaskedGradient
from util.network import weights_init, Discriminator, NetG
from util.utils import TextureDataset
from config import opt, bMirror, nz, nDep
from train_utils import test, train, save_embedding, interpolation, all_composition


def define_model(device):
    # define and record the network G D used
    N = 0
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    desc = "fc" + str(opt.fContent) + "_ngf" + str(ngf) + "_ndf" + str(ndf) + "_dep" + str(nDep) + "-" + str(opt.nDepD)
    if opt.WGAN:
        desc += '_WGAN'
    if opt.LS:
        desc += '_LS'
    if bMirror:
        desc += '_mirror'
    if opt.textureScale != 1:
        desc += "_scale" + str(opt.textureScale)
    # netE = ENCODER(ndf, opt.nDepD, nz, bSigm=not opt.LS and not opt.WGAN, condition=True)
    netG = NetG(ngf, nDep, nz, opt.nc, True)
    netD = Discriminator(ndf, opt.nDepD, opt.nc, bSigm=not opt.LS and not opt.WGAN, condition=True)
    # load model
    if opt.netD != '':
        state_dict = torch.load(opt.netD, map_location='cpu')
        netD.load_state_dict(state_dict)
        print("ok")
        print('Load ', opt.netD)
        netD.to(device)
        print(netD)
    else:
        netD.apply(weights_init)
        netD = netD.to(device)
        print(netD)
    if opt.netG != '':
        state_dict = torch.load(opt.netG, map_location='cpu')
        netG.load_state_dict(state_dict)
        print('Load ', opt.netG)
        netG.to(device)
        print(netG)
    else:
        netG.apply(weights_init)
        netG = netG.to(device)
        print(netG)
    return netG, netD, desc


def data_loader():
    # data transform
    canonicT = [transforms.RandomCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    mirrorT = []
    if bMirror:
        mirrorT += [transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()]
    transformTex = transforms.Compose(mirrorT + canonicT)
    # training set
    dataset = TextureDataset(opt.texturePath, transformTex, opt.textureScale, train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=opt.shuffle,
                                             num_workers=int(opt.workers), drop_last=True)
    # testing set
    dataset_test = TextureDataset(opt.texturePath, transformTex, opt.textureScale, train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False,
                                                  num_workers=int(opt.workers), drop_last=True)
    return dataloader, dataloader_test


if __name__ == "__main__":
    # set manualSeed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    # set gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # set dataloader and model
    if not opt.train == 3 or not opt.train == 4:
        dataloader, dataloader_test = data_loader()

    netG, netD, desc = define_model(device)
    # config loss
    if opt.coeff_color_loss:
        print('use color loss')
    gradient_critcriterion = MaskedGradient(opt).to(device)
    # set noise
    NZ = opt.imageSize // 2 ** nDep
    noise = torch.FloatTensor(opt.batchSize, opt.zLoc, NZ, NZ)
    fixnoise = torch.FloatTensor(opt.batchSize, opt.zLoc, NZ * 4, NZ * 4)
    noise = noise.to(device)
    fixnoise = fixnoise.to(device)
    Noise = [NZ, noise, fixnoise]
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # netD.parameters()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    if opt.train == 0:
        test(opt, dataloader_test, device, netD, netG, Noise, desc)
    elif opt.train == 1:
        train(opt, dataloader, dataloader_test, device, netD, netG, desc, Noise, optimizerD, optimizerG)
    elif opt.train == 2:
        save_embedding(opt, dataloader_test, device, netD)
    elif opt.train == 3:
        interpolation(opt, device, netG, Noise)
    elif opt.train == 4:
        all_composition(opt, device, netG, Noise)
