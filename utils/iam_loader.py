'''
@author: georgeretsi
'''

import os

import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset

from os.path import isfile

from utils.auxilary_functions import image_resize, centered

from utils.iam_config import *
from utils.iam_utils import gather_iam_info

import matplotlib.pyplot as plt

def main_loader(set, level):

    info = gather_iam_info(set, level)

    data = []
    for i, (img_path, transcr) in enumerate(info):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

        try:
            img = img_io.imread(img_path + '.png')
            img = 1 - img.astype(np.float32) / 255.0
            img = image_resize(img, height=img.shape[0] // 2)
        except:
            print('main_loader error: Could not load {}'.format(
                img_path
            ))

        data += [(img, transcr.replace("|", " "))]

    return data

class IAMLoader(Dataset):

    def __init__(self, set, level='line', fixed_size=(128, None), transforms=None):

        self.transforms = transforms
        self.set = set
        self.fixed_size = fixed_size

        save_file = dataset_path + '/' + set + '_' + level + '.pt'

        if isfile(save_file) is False:
            # hardcoded path and extension

            # if not os.path.isdir(img_save_path):
            #    os.mkdir(img_save_path)

            data = main_loader(set=set, level=level)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]

        #img = cv2.Canny((255 * img).astype(np.uint8),100,200) / 255.0

        transcr = " " + self.data[index][1] + " "
        #transcr = self.data[index][1]

        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        
        if self.set == 'train':
            # random resize at training !!!
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]
        
        
        #nheight, nwidth = img.shape[0], img.shape[1]

        #nheight, nwidth = int(.85 * img.shape[0]), int(.85 * img.shape[1])
        
        # small pads!!
        nheight, nwidth = max(4, min(fheight-32, nheight)), max(8, min(fwidth-64, nwidth))
        #nheight, nwidth = fheight-32, fwidth-64
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

        img = centered(img, (fheight, fwidth), border_value=0.0)

        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)


        # pad with zeroes
        #img = centered(img, (fheight, fwidth), np.random.uniform(.2, .8, size=2), border_value=0.0)
        img = torch.Tensor(img).float().unsqueeze(0)

        #if self.transforms is not None:
        #    for tr in self.transforms:
        #        if np.random.rand() < .5:
        #            img = tr(img.unsqueeze(0))[0]

        return img, transcr
