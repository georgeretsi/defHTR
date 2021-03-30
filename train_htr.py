import argparse
import logging

import os
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.iam_loader import IAMLoader
from config import *

from models import HTRNetC

from utils.auxilary_functions import affine_transformation
from valid_deforms import local_deform, morphological, uncertainty_reduction

import torch.nn.functional as F

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('HTR-Experiment::train')
logger.info('--- Running HTR Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                    help='lr')
parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                    help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
parser.add_argument('--display', action='store', type=int, default=100,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--df', action='store_true')
parser.add_argument('--batch_size', action='store', type=int, default=20)
parser.add_argument('--epochs', action='store', type=int, default=120)
parser.add_argument('--restart_epochs', action='store', type=int, default=40)


args = parser.parse_args()

print(args.df)

device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
logger.info('###########################################')

# prepare datset loader

logger.info('Loading dataset.')

aug_transforms = [lambda x: affine_transformation(x, s=.1)]

train_set = IAMLoader('train', level=level, fixed_size=fixed_size, transforms=aug_transforms)
test_set = IAMLoader('test', level=level, fixed_size=fixed_size)

batch_size = args.batch_size
# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

# load CNN
logger.info('Preparing Net...')

save_name = 'iam_cnn'
if args.df:
    save_name += '_df'
save_name += "_asd2.pt"


net = HTRNetC(cnn_cfg, cnn_top, len(classes), stn=False, df=args.df)
#net.load_state_dict(torch.load(save_name))
net.to(device)

nlr = args.learning_rate

max_epochs = args.epochs
restart_epochs = args.restart_epochs

parameters = list(net.parameters())
optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)

def train(epoch):

    net.train()


    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):

        optimizer.zero_grad()

        img = Variable(img.to(device))

        # geometrical and morphological deformations
        '''
        rids = torch.BoolTensor(torch.bernoulli(.1 * torch.ones(img.size(0))).bool())
        if sum(rids) > 1:
            u = 4 # 2 ** np.random.randint(2, 5)
            img[rids] = local_deform(img[rids], rm=None, dscale=u, a=np.random.uniform(2.0, 4.0))

        rids = torch.BoolTensor(torch.bernoulli(.1 * torch.ones(img.size(0))).bool())
        if sum(rids) > 1:
            u = 4 # 2 ** np.random.randint(2, 5)
            img[rids] = morphological(img[rids], rm=None, dscale=u)
        '''

        output = net(img.detach()) #.permute(2, 0, 1)

        act_lens = torch.IntTensor(img.size(0)*[output.size(0)]) 
        labels = torch.IntTensor([cdict[c] for c in ''.join(transcr)])
        label_lens = torch.IntTensor([len(t) for t in transcr])

        ls_output = F.log_softmax(output, dim=2).cpu()
        loss_val = F.ctc_loss(ls_output, labels, act_lens, label_lens, zero_infinity=True, reduction='sum') / img.size(0)

        closs += [loss_val.data]

        loss_val.backward()

        optimizer.step()


        # mean runing errors??
        if iter_idx % args.display == args.display-1:
            logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            closs = []

            net.eval()

            tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
            print('orig:: ' + tst_transcr)
            with torch.no_grad():
                tst_o = net(Variable(tst_img.to(device)).unsqueeze(0)) #.permute(2, 0, 1)
                tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                print('gdec:: ' + ''.join([icdict[t] for t in tt]).replace('_', ''))

            net.train()


import editdistance

def test(epoch, ur=False, k=5):
    net.eval()

    logger.info('Testing at epoch %d', epoch)

    tdecs = []
    transcrs = []
    for (img, transcr) in test_loader:
        img = Variable(img.to(device))
        if ur:
            img = uncertainty_reduction(net, img, dscale=4, a=0.01, lr=1e-3, N=k, nc=25).detach()
        with torch.no_grad():
            o = net(img) #.permute(2, 0, 1)
        tdec = o.argmax(2).permute(1, 0).view(img.size(0), -1).cpu().numpy()
        tdecs += [tdec]
        transcrs += list(transcr)


    tdecs = np.concatenate(tdecs)

    cer, wer = [], []
    cntc, cntw = 0, 0
    for tdec, transcr in zip(tdecs, transcrs):
        transcr = transcr.strip()
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()

        # calculate CER and WER
        cc = float(editdistance.eval(dec_transcr, transcr))
        ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
        cntc += len(transcr)
        cntw +=  len(transcr.split(' '))
        cer += [cc]
        wer += [ww]


    cer = sum(cer) / cntc
    wer = sum(wer) / cntw

    logger.info('CER at epoch %d: %f', epoch, cer)
    logger.info('WER at epoch %d: %f', epoch, wer)

    net.train()

cnt = 0
logger.info('Training:')
for epoch in range(1, max_epochs + 1):

    train(epoch)
    scheduler.step()

    if epoch % 2 == 0:
        test(epoch)

    if epoch % 10 == 0:
        logger.info('Saving net after %d epochs', epoch)
        torch.save(net.cpu().state_dict(), save_name)
        net.to(device)

    if epoch % restart_epochs == 0:
        parameters = list(net.parameters())

        optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)

torch.save(net.cpu().state_dict(), save_name)
net.to(device)

test(-1)
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print('-------------------- :: ' + str(k) + ' :: --------------------')
    #for _ in range(3):
    test(-1, ur=True, k=k)
