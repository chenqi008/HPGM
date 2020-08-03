import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn
from config import opt

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class TextureDataset(Dataset):
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, data_dir, transform=None, scale=1, train=True):
        self.data_dir = data_dir
        self.transform = transform

        self.train = train

        # 19
        self.material_class = ['Log', 'Wood Veneer', 'Bamboo Vine', 'Wood Grain',
            'Pure Color Wood', 'Painted Wood', 'Cement Board', 'Ceramic Tile',
            'Ceramics', 'Granite', 'Jade', 'Marble', 'Mosaic', 'Quartz',
            'Rock Plate', 'Stone Brick', 'Wallpaper', 'Wall Cloth', 'Coating']

        # 12
        self.color_class = ['Earth color', 'Wood color', 'Black', 'Gray', 'Yellow',
            'Green', 'Orange', 'White', 'Blue', 'Purple', 'Pink', 'Red']

        # 2
        self.apply = ['floor', 'wall']


        self.filenames, self.train_id, self.test_id = self.load_filenames(self.data_dir)
        self.embeddings, self.descriptions, self.material_labels, \
            self.color_labels, self.apply_labels = self.load_embedding()

        if train:
            # handle the training set
            print('preparing training set ...')
            self.X_train = []
            for file_id in self.train_id:
                material_color_apply_path = self.filenames[file_id]
                # handle image
                img_name = '%s%s' % (self.data_dir, material_color_apply_path[4])
                img = Image.open(img_name).convert('RGB')
                if scale != 1:
                    img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),PIL.Image.LANCZOS)
                # handle embedding
                emb = self.embeddings[file_id]
                # handle description
                desc = self.descriptions[file_id]
                # handle labels
                material_label = self.material_labels[file_id]
                color_label = self.color_labels[file_id]
                apply_label = self.apply_labels[file_id]

                self.X_train += [(img, emb, desc, material_label, color_label, apply_label)]
                sys.stdout.flush()
                # print (img_name.split('/')[-1], "img added", img.size, "total length", len(self.X_train))
                if len(self.X_train) > 4000:
                    break ##usually want to avoid so many files
            # this affects epoch length..
            if len(self.X_train) < 2000:
                c = int(2000/len(self.X_train))
                self.X_train*=c
        else:
            # handle the testing set
            print('preparing testing set ...')
            self.X_test = []
            for file_id in self.test_id:
                material_color_apply_path = self.filenames[file_id]
                # handle image
                img_name = '%s%s' % (self.data_dir, material_color_apply_path[4])
                img = Image.open(img_name).convert('RGB')
                if scale != 1:
                    img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),PIL.Image.LANCZOS)
                # handle embedding
                emb = self.embeddings[file_id]
                # handle description
                desc = self.descriptions[file_id]
                # handle labels
                material_label = self.material_labels[file_id]
                color_label = self.color_labels[file_id]
                apply_label = self.apply_labels[file_id]

                self.X_test += [(img, emb, desc, material_label, color_label, apply_label)]
                sys.stdout.flush()
                # print (img_name.split('/')[-1], "img added", img.size, "total length", len(self.X_test))

    def load_filenames(self, data_dir):
        # filepath = os.path.join(data_dir, 'filenames.pickle')
        filepath = os.path.join(data_dir, 'train_test_random', 'id_file.txt')
        # '5ce3d1aa4846c60001a39b1c': ['Coating', 'EBC175', 'Wood color', 'wall', 'Wall Decoration/Coating/Nippon primroses.jpg']
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))

        train_id_path = os.path.join(data_dir, 'train_test_random', 'train_id.txt')
        with open(train_id_path, 'r') as f:
            # train_id = f.readlines()
            train_id = f.read().splitlines()

        test_id_path = os.path.join(data_dir, 'train_test_random', 'test_id.txt')
        with open(test_id_path, 'r') as f:
            # test_id = f.readlines()
            test_id = f.read().splitlines()

        return filenames, train_id, test_id

    def load_embedding(self):
        embeddings = {}
        descriptions = {}
        material_labels = {}
        color_labels = {}
        apply_labels = {}
        
        for file_id in self.filenames.keys():
            embedding_temp = np.zeros(19+12)
            
            material_position = self.material_class.index(self.filenames[file_id][0])
            color_position = self.color_class.index(self.filenames[file_id][2])
            apply_position = self.apply.index(self.filenames[file_id][3])

            material_labels[file_id] = material_position
            color_labels[file_id] = color_position
            apply_labels[file_id] = apply_position

            embedding_temp[material_position] = 1.0
            embedding_temp[opt.z_material+color_position] = 1.0

            embeddings[file_id] = torch.from_numpy(embedding_temp.astype(np.float32))
            descriptions[file_id] = '{}, {}, {}'.format(self.filenames[file_id][0],
                self.filenames[file_id][2], self.filenames[file_id][3])
        # embeddings:19 class material + 12 class color + 2 apply(wood+floor)
        return embeddings, descriptions, material_labels, color_labels, apply_labels


    def __getitem__(self, index):
        if self.train:
            img = self.X_train[index][0]
            emb = self.X_train[index][1]
            desc = self.X_train[index][2]
            material_label = self.X_train[index][3]
            color_label = self.X_train[index][4]
            apply_label = self.X_train[index][5]
        else:
            img = self.X_test[index][0]
            emb = self.X_test[index][1]
            desc = self.X_test[index][2]
            material_label = self.X_test[index][3]
            color_label = self.X_test[index][4]
            apply_label = self.X_test[index][5]

        if self.transform is not None:
            img2 = self.transform(img)        
        return img2, emb, desc, material_label, color_label, apply_label


    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)

def GaussKernel(sigma,wid=None):
    if wid is None:
        wid =2 * 2 * sigma + 1+10

    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

    def make_kernel(sigma):
        # kernel radius = 2*sigma, but minimum 3x3 matrix
        kernel_size = max(3, int(wid))
        kernel_size = min(kernel_size,150)
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        # make 2D kernel
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=np.float32)
        # normalize kernel by sum of elements
        kernel = np_kernel / np.sum(np_kernel)
        return kernel
    ker = make_kernel(sigma)
  
    a = np.zeros((3,3,ker.shape[0],ker.shape[0])).astype(dtype=np.float32)
    for i in range(3):
        a[i,i] = ker
    return a

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gsigma=1.##how much to blur - larger blurs more ##+"_sig"+str(gsigma)
gwid=61
kernel = torch.FloatTensor(GaussKernel(gsigma,wid=gwid)).to(device)##slow, pooling better
def avgP(x):
    return nn.functional.avg_pool2d(x,int(16))
def avgG(x):
    pad=nn.functional.pad(x,(gwid//2,gwid//2,gwid//2,gwid//2),'reflect')##last 2 dimensions padded
    return nn.functional.conv2d(pad,kernel)##reflect pad should avoid border artifacts

def plotStats(a,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15))
    names = ["pTrue", "pFake", "pFake2", "contentLoss I", "contentLoss I_M", "norm(alpha)", "entropy(A)", "tv(A)", "tv(alpha)", "diversity(A)"]
    win=50##for running avg
    for i in range(a.shape[1]):
        if i <3:
            ix=0
        elif i <5:
            ix =1
        elif i >=5:
            ix=i-3
        plt.subplot(a.shape[1]-3+1,1,ix+1)
        plt.plot(a[:,i],label= "err"+str(i)+"_"+names[i])
        try:
            av=np.convolve(a[:,i], np.ones((win,))/win, mode='valid')
            plt.plot(av,label= "av"+str(i)+"_"+names[i],lw=3)
        except Exception as e:
            print ("ploterr",e)
        plt.legend(loc="lower left")
    plt.savefig(path+"plot.png")

    def Mstring(v):
        s=""
        for i in range(v.shape[0]):
            s+= names[i]+" "+str(v[i])+";"
        return s

    print("MEAN",Mstring(a.mean(0)))
    print("MEAN",Mstring(a[-100:].mean(0)))
    plt.close()

#large alpha emphasizes new -- conv. generation , less effect on old, the mix template output
#@param I_G is parametric generation
#@param I_M is mixed template image
def blend(I_G, I_M, alpha, beta):
    if opt.blendMode==0:
        out= I_M*(1 - beta) + alpha * I_G[:, :3]
    if opt.blendMode==1:
        out = I_G[:, :3] * alpha * 2 + I_M
    if opt.blendMode==2:##this is the mode described in paper, convex combination
        out= I_G[:, :3] * alpha + (1 - alpha) * I_M
    return torch.clamp(out,-1,1)

##show the different btw final image and mixed image -- this shows the parametric output of our network
def invblend(I,I_M,alpha,beta):
    return torch.clamp(I-I_M,-1,1)


#absolute difference in X and Y directions
def total_variation(y):
    return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


##2D array of the edges of C channels image
def tvArray(x):
    border1 = x[:, :, :-1] - x[:, :, 1:]
    border1 = torch.cat([border1.abs().sum(1).unsqueeze(1), x[:, :1, :1] * 0], 2)  ##so square with extra 0 line
    border2 = x[:, :, :, :-1] - x[:, :, :, 1:]
    border2 = torch.cat([border2.abs().sum(1).unsqueeze(1), x[:, :1, :, :1] * 0], 3)
    border = torch.cat([border1, border2], 1)
    return border


##negative gram matrix
def gramMatrix(x,y=None,sq=True,bEnergy=False):
    if y is None:
        y = x

    B, CE, width, height = x.size()
    hw = width * height

    energy = torch.bmm(x.permute(2, 3, 0, 1).view(hw, B, CE),
                       y.permute(2, 3, 1, 0).view(hw, CE, B), )
    energy = energy.permute(1, 2, 0).view(B, B, width, height)
    if bEnergy:
        return energy
    sqX = (x ** 2).sum(1).unsqueeze(0)
    sqY = (y ** 2).sum(1).unsqueeze(1)
    d=-2 * energy + sqX + sqY
    if not sq:
        return d##debugging
    gram = -torch.clamp(d, min=1e-10)#.sqrt()
    return gram

##some image level content loss
def contentLoss(a,b,netR,opt):
    def nr(x):
        return (x**2).mean()
        return x.abs().mean()

    if opt.cLoss==0:
        a = avgG(a)
        b = avgG(b)
        return nr(a.mean(1) - b.mean(1))
    if opt.cLoss==1:
        a = avgP(a)
        b = avgP(b)
        return nr(a.mean(1) - b.mean(1))

    if opt.cLoss==10:
        return nr(netR(a)-netR(b))
    if opt.cLoss==100:
        return nr(netR(a)-b)
    if opt.cLoss == 101:
        return nr(avgG(netR(a)) - avgG(b))
    if opt.cLoss == 102:
        return nr(avgP(netR(a)) - avgP(b))
    if opt.cLoss == 103:
        return nr(avgG(netR(a)).mean(1) - avgG(b).mean(1))

    raise Exception("NYI")

##visualization routine to show mix arrayA as many colourful channels
def rgb_channels(x):
    N=x.shape[1]
    if N ==1:
        return torch.cat([x,x,x],1)##just white dummy

    cu= int(N**(1/3.0))+1
    a=x[:,:3]*0##RGB image
    for i in range(N):
        c1=int(i%cu)
        j=i//cu
        c2=int(j%cu)
        j=j//cu
        c3=int(j%cu)
        a[:,:1]+= c1/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,1:2]+=c2/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,2:3]+=c3/float(cu+1)*x[:,i].unsqueeze(1)
    return a#*2-1##so 0 1


if opt.zPeriodic:
    # 2*nPeriodic initial spread values
    # slowest wave 0.5 pi-- full cycle after 4 steps in noise tensor
    # fastest wave 1.5pi step -- full cycle in 0.66 steps
    def initWave(nPeriodic):
        buf = []
        for i in range(nPeriodic // 4+1):
            v = 0.5 + i / float(nPeriodic//4+1e-10)
            buf += [0, v, v, 0]
            buf += [0, -v, v, 0]  # #so from other quadrants as well..
        buf=buf[:2*nPeriodic]
        awave = np.array(buf, dtype=np.float32) * np.pi
        awave = torch.FloatTensor(awave).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return awave
    waveNumbers = initWave(opt.zPeriodic).to(device)

    class Waver(nn.Module):
        def __init__(self):
            super(Waver, self).__init__()
            if opt.zGL > 0:
                K=50
                layers=[nn.Conv2d(opt.zGL, K, 1)]
                layers +=[nn.ReLU(True)]
                layers += [nn.Conv2d(K,2*opt.zPeriodic, 1)]
                self.learnedWN =  nn.Sequential(*layers)
            else:##static
                self.learnedWN = nn.Parameter(torch.zeros(opt.zPeriodic * 2).uniform_(-1, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * 0.2)
        def forward(self, c, GLZ=None):
            if opt.zGL > 0:
                return (waveNumbers + 5*self.learnedWN(GLZ)) * c

            return (waveNumbers + self.learnedWN) * c
    learnedWN = Waver()
else:
    learnedWN = None


def setNoise(noise):
    noise = noise.detach()*1.0
    noise.normal_(0, 1)
    return noise


def setNoise_interpolation(noise, mp1=0, mp2=0, cp1=0, cp2=0, ap1=0, ap2=0):
    # 19
    material_class = ['Log', 'Wood Veneer', 'Bamboo Vine', 'Wood Grain',
        'Pure Color Wood', 'Painted Wood', 'Cement Board', 'Ceramic Tile',
        'Ceramics', 'Granite', 'Jade', 'Marble', 'Mosaic', 'Quartz',
        'Rock Plate', 'Stone Brick', 'Wallpaper', 'Wall Cloth', 'Coating']
    # 12
    color_class = ['Earth color', 'Wood color', 'Black', 'Gray', 'Yellow',
        'Green', 'Orange', 'White', 'Blue', 'Purple', 'Pink', 'Red']
    # 2
    apply_class = ['floor', 'wall']

    # embedding
    emb1 = np.zeros(19+12)
    emb2 = np.zeros(19+12)

    # first embedding
    emb1[mp1] = 1.0
    emb1[19+cp1] = 1.0
    # emb1[19+12+ap1] = 1.0

    # second embedding
    emb2[mp2] = 1.0
    emb2[19+cp2] = 1.0
    # emb2[19+12+ap2] = 1.0

    emb1 = torch.from_numpy(emb1.astype(np.float32))
    emb2 = torch.from_numpy(emb2.astype(np.float32))

    description = '{}, {}, {} -> {}, {}, {}'.format( \
        material_class[mp1], color_class[cp1], apply_class[ap1], \
        material_class[mp2], color_class[cp2], apply_class[ap2])

    noise1=noise.detach()*1.0
    noise2=noise.detach()*1.0
    # noise1.uniform_(-1, 1)  # normal_(0, 1)
    # noise2.uniform_(-1, 1)  # normal_(0, 1)
    noise1.normal_(0, 1)
    noise2.normal_(0, 1)
    return noise1, noise2, emb1, emb2, description


def setEmb(mp1=0, cp1=0, ap1=0):
    # 19
    material_class = ['Log', 'Wood Veneer', 'Bamboo Vine', 'Wood Grain',
        'Pure Color Wood', 'Painted Wood', 'Cement Board', 'Ceramic Tile',
        'Ceramics', 'Granite', 'Jade', 'Marble', 'Mosaic', 'Quartz',
        'Rock Plate', 'Stone Brick', 'Wallpaper', 'Wall Cloth', 'Coating']
    # 12
    color_class = ['Earth color', 'Wood color', 'Black', 'Gray', 'Yellow',
        'Green', 'Orange', 'White', 'Blue', 'Purple', 'Pink', 'Red']
    # 2
    apply_class = ['floor', 'wall']

    # embedding
    emb1 = np.zeros(19+12)
    
    # first embedding
    emb1[mp1] = 1.0
    emb1[19+cp1] = 1.0
    # emb1[19+12+ap1] = 1.0

    emb1 = torch.from_numpy(emb1.astype(np.float32))

    description = {}
    description["0"] = "%s, %s, %s"%(material_class[mp1], color_class[cp1], apply_class[ap1])

    return emb1, description


def plot_loss(figure_id, x, y, label, xlabel, ylabel, output_dir, output_name):
    # plot
    plt.figure(figure_id)

    plt.plot(x, y, color="b", linestyle="-", linewidth=1, label='{}'.format(label))

    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))

    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, '{}'.format(output_name)))
    plt.close(figure_id)


def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
    else:
        torch.save(model.state_dict(), '%s/%s_epoch_%03d.pth' % (model_dir, model_name, epoch))


def latent_lerp(gan, z0, z1, nb_frames):
    """Interpolate between two images in latent space"""
    # imgs = []
    for i in range(nb_frames+1):
        alpha = i / float(nb_frames)
        z = (1.0 - alpha) * z0 + alpha * z1
        if i==0:
            imgs = gan(z)
            # imgs = imgs.unsqueeze(0)
        else:
            imgs = torch.cat((imgs, gan(z)), 0)
        # imgs.append(gan(z))
    return imgs


def latent_lerp_square(gan, z0, z1, emb1, emb2, nb_frames):
    """Interpolate as a square between two images in latent space"""
    wh = int(nb_frames ** 0.5)
    # line (material)
    for i in range(wh):
        alpha = i / float(wh)
        # column (color)
        for j in range(wh):
            beta = j / float(wh)
            z = (1.0 - alpha) * z0 + alpha * z1
            z = z[torch.arange(z.size(0))==0]
            # z[:, 19:19+12] = ((1.0 - beta) * z0 + beta * z1)[:, 19:19+12]
            c = (1.0 - alpha) * emb1 + alpha * emb2
            c[19:19+12] = ((1.0 - beta) * emb1 + beta * emb2)[19:19+12]
            c = c.unsqueeze(0)
            if (i+j) == 0:
                imgs = gan(z, c)
            else:
                imgs = torch.cat((imgs, gan(z, c)), 0)
    return imgs


def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance




