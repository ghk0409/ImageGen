import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn

# 네트워크 grad 설정
def set_requires_grad(nets, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# 네트워크 weights 초기화, weights를 N(0, 0.02)로 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    # define the initialization function
    def init_func(m):
        className = m.__class__.__name__

        if hasattr(m, 'weight') and (className.find('Conv') != -1 or className.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies
        elif className.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type}')
    # apply the initialization function <init_func>
    net.apply(init_func)

# 네트워크 저장
# DCGAN은 네트워크가 2개(G, D)이므로 각각 netG, optimG, netD, optimD 저장
def save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch