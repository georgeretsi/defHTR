import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.ops import DeformConv2d
from valid_deforms import local_deform

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deformable=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        
        if deformable:
           self.conv2 = DConv2d(planes, planes, 3, 1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DConv2d(nn.Module):

    def __init__(self, in_dim, out_dim, k=3, stride=1, bias=True):
        super(DConv2d, self).__init__()

        self.offset = nn.Conv2d(in_dim, 2 * k * k, 1, stride, 0)
        self.offset.weight.data *= .1

        self.dconv = DeformConv2d(in_dim, out_dim, k, stride, k//2, bias=False)

    def forward(self, x):

        o = self.offset(x.detach())

        y = self.dconv(x, o)

        return y


class CNN(nn.Module):
    def __init__(self, cnn_cfg, df=False):
        super(CNN, self).__init__()

        self.k = 3

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, (4, 2), 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x, deformable=df))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce=True):

        y = x
        for nn_module in self.features:
            y = nn_module(y)

        if reduce:
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])

        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtopC(nn.Module):
    def __init__(self, input_size, hidden_size, nclasses, sa_head=False):
        super(CTCtopC, self).__init__()

        if sa_head:
            self.sa = SelfAttention(input_size)
        else:
            self.sa = None

        self.temporal = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=1, padding=(0, 2)),
        )

    def forward(self, x):

        if self.sa is not None:
            x, _ = self.sa(x)


        y = self.temporal(x).squeeze(2)
        y = y.permute(2, 0, 1)

        return y


class HTRNetC(nn.Module):
    def __init__(self, cnn_cfg, hidden, nclasses, df=False, stn=False):
        super(HTRNetC, self).__init__()

        if stn:
            self.norm = STN([(2, 16), 'M', (2, 32),], df=stn_df)
        else:
            self.norm = None

        self.features = CNN(cnn_cfg, df=df)
        self.top = CTCtopC(cnn_cfg[-1][-1], hidden, nclasses)

    def forward(self, x):

        if self.norm is None:
            y = x
        else:
            y = self.norm(x)

        y = self.features(y)
        y = self.top(y)

        return y




class STN(nn.Module):

    def __init__(self,cfg, mode=0):
        super(STN, self).__init__()

        self.cnn = CNN(cfg)

        self.mode = mode
        if mode==0:
            self.aff_last = nn.Linear(cfg[-1][-1], 4)
            self.aff_last.weight.data *= 0.01
            self.aff_last.bias.data *= 0.01

            self.ref_aff = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            self.ld_last = nn.Sequential(
                #nn.Conv2d(cfg[-1][-1], cfg[-1][-1], 3, 1, 1), nn.ReLU(),
                nn.Conv2d(cfg[-1][-1], 2, 1, 1, 0),
            )
            self.ld_last[-1].weight.data *= 0.001
            self.ld_last[-1].bias.data *= 0.001

        
    def forward(self, x):

        xd = x
        y = self.cnn(x, reduce=False)

        if self.mode == 0:
            y = F.adaptive_avg_pool2d(y, [1,1])
            y_aff = self.aff_last(y.view(x.size(0), -1))

            aff = self.ref_aff.to(x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
            aff[:, :, :2] += y_aff.view(-1, 2, 2)

            grid = F.affine_grid(aff, x.size())
            xd = F.grid_sample(xd, grid, padding_mode='border')
        else:
            y_ld = self.ld_last(y)
            xd = local_deform(xd, y_ld)

        return  xd