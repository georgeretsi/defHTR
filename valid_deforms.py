import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.interpolate import interpolate

def local_deform(img, rm=None, dscale=4, a=2.0):

    if rm is None:
        map_size = (2* img.size(0), img.size(2) // dscale, img.size(3) // dscale)
        cm = 2.0 * (torch.rand(2 * img.size(0), 35, 3).to(img.device) - .5)
        rm = interpolate(cm.data, map_size).view(-1, 2, map_size[1], map_size[2]).detach()

    N = img.size(0)
    h, w = img.size(2), img.size(3)

    ig = torch.stack([
        torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1),
        torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w),
    ], 0).to(img.device)
    ig = ig.view(1, 2, h, w).repeat(N, 1, 1, 1)

    nrm = a * rm
    nrm[:, 0] -= torch.stack([l.mean() for l in nrm[:, 0]]).detach().view(-1, 1, 1)
    nrm[:, 1] -= torch.stack([l.mean() for l in nrm[:, 1]]).detach().view(-1, 1, 1)

    if (img.size(2) == rm.size(2)) and (img.size(3) == rm.size(3)):
        nig = ig.permute(0, 2, 3, 1).detach() + nrm.permute(0, 2, 3, 1)
    else:
        nig = ig.permute(0, 2, 3, 1).detach() + 1.0 * F.interpolate(nrm, (h, w), mode='bilinear').permute(0, 2, 3, 1)

    nimg = F.grid_sample(img, nig, mode='bilinear', padding_mode='border', align_corners=None)

    return nimg

def morphological(img, rm=None, dscale=4, a=10.0):

    if rm is None:
        map_size = (img.size(0), img.size(2) // dscale, img.size(3) // dscale)
        cm = 2.0 * (torch.rand(img.size(0), 35, 3).to(img.device) - .5)
        rm = interpolate(cm.data, map_size).view(-1, 1, map_size[1], map_size[2]).detach()

    h, w = img.size(2), img.size(3)

    dimg3 = F.max_pool2d(img, 3, 1, 1)
    dimg5 = F.max_pool2d(img, 5, 1, 2)
    eimg3 = -F.max_pool2d(-img, 3, 1, 1)
    eimg5 = -F.max_pool2d(-img, 5, 1, 2)
    simg = torch.cat([eimg5, eimg3, img, dimg3, dimg5], 1)

    nm = simg.size(1)

    centers = [float(i) / nm + 1.0 / (2 * nm) for i in range(nm)]  # [1.0 / 6, 1.0 / 3 + 1.0 / 6, 2.0 / 3 + 1.0 / 6]
    map1 = rm
    map1 = (map1 - map1.min()) / (map1.max() - map1.min())

    map3 = torch.cat([-(map1 - c) ** 2 for c in centers], 1)
    morph_mask = F.interpolate(map3, (h, w), mode='nearest')

    timg = simg * F.softmax(a * morph_mask, 1)
    timg = timg.sum(1).unsqueeze(1)

    return timg


def uncertainty_reduction(net, img, dscale=16, a=0.1, lr=1e-4, N=5, nc=50):


    sh, sw = img.size(2) // dscale, img.size(3) // dscale


    cm = 2.0 * (torch.rand(2 * img.size(0), nc, 4).to(img.device) - .5)
    cm[:, :, 2] *= .1
    cm[:, :, 3] += 10

    map_size = (2 * img.size(0), sh, sw)
    rm = interpolate(cm, map_size).view(-1, 2, map_size[1], map_size[2])
    rm = nn.Parameter(rm)

    toptimizer = torch.optim.Adam([rm], lr) 

    dh = torch.Tensor([1, -1]).view(1, 1, 2, 1).repeat(2, 1, 1, 1).to(img.device)
    dw = dh.view(-1, 1, 1, 2)
    for i in range(N):
        toptimizer.zero_grad()

        nrm = a * rm #interpolate(cm, map_size).view(-1, 2, map_size[1], map_size[2])
        nimg = local_deform(img, nrm)

        output = net(nimg)

        eloss = F.conv2d(rm, dh, groups=2).norm() + F.conv2d(rm, dw, groups=2).norm()
        loss_val = -1.0 * (output.softmax(2)[:, :, :-1] ** 2).mean() + 0.0 * eloss

        loss_val.backward()
        toptimizer.step()

    nrm = a * rm #interpolate(cm, map_size).view(-1, 2, map_size[1], map_size[2])
    fimg = local_deform(img, nrm)


    return fimg