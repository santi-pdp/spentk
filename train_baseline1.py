import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import argparse
from spentk.models.baseline1 import DNN
from tensorboardX import SummaryWriter
from spentk.datasets.swin_dataset import SWinDataset
from spentk.datasets.transforms import *
import random
import os
import json
import timeit


def eval_epoch(dloader, model, criterion, epoch, writer, log_freq):
    model.eval()
    timings = []
    va_losses = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        # split into (X, Y) pairs
        lps_y, lps_x = batch
        lps_x = Variable(lps_x, volatile=True)
        lps_y = Variable(lps_y, volatile=True)
        lps_x, lps_x_pha = torch.chunk(lps_x, 2, dim=2)
        lps_x = lps_x.squeeze(2)
        lps_y, lps_y_pha = torch.chunk(lps_y, 2, dim=2)
        lps_y = lps_y.squeeze(2)
        if opts.cuda:
            lps_x = lps_x.cuda()
            lps_y = lps_y.cuda()
        y_ = model(lps_x)
        loss = criterion(lps_y, y_)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        va_losses.append(loss.cpu().data[0])
        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('EVAL epoch {}, Batch {}/{} loss: {:.3f} '
                  'btime: {:.3f} s, mbtime: {:.3f}'
                  ''.format(epoch, bidx, len(dloader), loss.cpu().data[0],
                            timings[-1], np.mean(timings)))
    writer.add_scalar('valid/loss', np.mean(va_losses), epoch)
    return va_losses


def train(opts):
    model = DNN()
    if opts.cuda:
        model.cuda()
    writer = SummaryWriter(os.path.join(opts.save_path,
                                        'train'))
    opt = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    dset = SWinDataset(opts.dataset, cache_path=opts.cache_path,
                       transform=wav2fft(logpower=True))
    va_dset = SWinDataset(opts.dataset, cache_path=opts.cache_path,
                          transform=wav2fft(logpower=True),
                          split='valid')
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, num_workers=opts.num_workers)
    va_dloader = DataLoader(va_dset, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers)
    timings = []
    global_step = 0
    patience = opts.patience
    min_va_loss = np.inf
    for epoch in range(opts.epoch):
        model.train()
        beg_t = timeit.default_timer()
        for bidx, batch in enumerate(dloader, start=1):
            # split into (X, Y) pairs
            lps_y, lps_x = batch
            lps_x = Variable(lps_x)
            lps_y = Variable(lps_y)
            lps_x, lps_x_pha = torch.chunk(lps_x, 2, dim=2)
            lps_x = lps_x.squeeze(2)
            lps_y, lps_y_pha = torch.chunk(lps_y, 2, dim=2)
            lps_y = lps_y.squeeze(2)
            if opts.cuda:
                lps_x = lps_x.cuda()
                lps_y = lps_y.cuda()
            opt.zero_grad()
            y_ = model(lps_x)
            loss = criterion(y_, lps_y)
            loss.backward()        
            opt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if bidx % opts.save_freq == 0 or bidx >= len(dloader):
                print('Batch {}/{} (epoch {}) loss: {:.3f} '
                      'btime: {:.3f} s, mbtime: {:.3f}'
                      ''.format(bidx, len(dloader), epoch, loss.cpu().data[0],
                                timings[-1], np.mean(timings)))
                writer.add_scalar('training/loss', loss.cpu().data[0], global_step)
            global_step += 1
        va_losses = eval_epoch(va_dloader, model, criterion, epoch, writer, opts.save_freq)
        mva_loss = np.mean(va_losses)
        if min_va_loss > mva_loss:
            print('Val loss improved {:.3f} --> {:.3f}'.format(min_va_loss,
                                                               mva_loss))
            min_va_loss = mva_loss
            torch.save(model.state_dict(), os.path.join(opts.save_path,
                                                        'model-e{}.ckpt'.format(epoch)))
            patience = opts.patience
        else:
            patience -= 1
            print('Val loss did not improve. Curr patience'
                  '{}/{}'.format(patience, opts.patience))
            if patience <= 0:
                print('Finishing training, out of patience')
                break

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_freq', type=int, default=100, help='Num of '
                        'batches to save/log (Def: 10)')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='baseline1_ckpt')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--dataset', type=str,
                        default='data',
                        help='Root folder where the following subsets '
                             'are stored:\n(1) clean_trainset, (2) '
                             'noisy_trainset\n(3) clean_valset, '
                             '(4) noisy_valset\n, (5) clean_testset, '
                             '(6) noisy_testset.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers in data lader (Def: 1).')
    opts = parser.parse_args()

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))

    train(opts)
