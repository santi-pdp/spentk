import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import argparse
from spentk.models.rnn import LPSRNN
from tensorboardX import SummaryWriter
from spentk.datasets.swin_dataset import SWinDataset, WavPairDataset
from spentk.datasets.swin_dataset import SeqLPSCollater
from spentk.datasets.transforms import *
import random
import os
import json
import timeit


def get_grads(model):
    grads = None
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        if param.grad is None:
            print('WARNING getting grads: {} param grad is None'.format(k))
            continue
        if grads is None:
            grads = param.grad.cpu().data.view((-1, ))
        else:
            grads = torch.cat((grads, param.grad.cpu().data.view((-1,))), dim=0)
    return grads

def eval_epoch(dloader, model, criterion, epoch, writer, log_freq, device):
    model.eval()
    with torch.no_grad():
        timings = []
        va_losses = []
        beg_t = timeit.default_timer()
        for bidx, batch in enumerate(dloader, start=1):
            # split into (X, Y) pairs
            lps_x, lps_y = batch
            lps_x, lps_x_pha = torch.chunk(lps_x, 2, dim=3)
            lps_x = lps_x.squeeze(3)
            lps_y, lps_y_pha = torch.chunk(lps_y, 2, dim=3)
            lps_y = lps_y.squeeze(3)
            lps_x = lps_x.to(device)
            lps_y = lps_y.to(device)
            y_, state = model(lps_x)
            loss = criterion(y_, lps_y)
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
    CUDA = torch.cuda.is_available() and not opts.no_cuda
    device = 'cuda' if CUDA else 'cpu'

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)

    # model build
    model = LPSRNN(dropout=opts.dropout)
    model.to(device)
    print(model)
    writer = SummaryWriter(os.path.join(opts.save_path,
                                        'train'))
    opt = optim.Adam(model.parameters(), lr=opts.lr)
    if opts.loss == 'l2':
        criterion = nn.MSELoss()
    elif opts.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        raise TypeError('Loss function {} not understood'.format(opts.loss))
    dset = WavPairDataset(opts.dataset, transform=wav2stft(logpower=True))
    va_dset = WavPairDataset(opts.dataset, split='valid',
                             transform=wav2stft(logpower=True))
    collater = SeqLPSCollater(maxlen=opts.maxlen)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, num_workers=opts.num_workers,
                         collate_fn=collater)
    va_dloader = DataLoader(va_dset, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers,
                            collate_fn=collater)
    timings = []
    global_step = 0
    patience = opts.patience
    min_va_loss = np.inf
    for epoch in range(opts.epoch):
        model.train()
        beg_t = timeit.default_timer()
        for bidx, batch in enumerate(dloader, start=1):
            # split into (X, Y) pairs
            lps_x, lps_y = batch
            lps_x, lps_x_pha = torch.chunk(lps_x, 2, dim=3)
            lps_x = lps_x.squeeze(3)
            lps_y, lps_y_pha = torch.chunk(lps_y, 2, dim=3)
            lps_y = lps_y.squeeze(3)
            lps_x = lps_x.to(device)
            lps_y = lps_y.to(device)
            opt.zero_grad()
            y_, state = model(lps_x)
            loss = criterion(y_, lps_y)
            loss.backward()        
            opt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if bidx % opts.save_freq == 0 or bidx >= len(dloader):
                print('Batch {}/{} (epoch {}) loss: {:.3f} '
                      'btime: {:.3f} s, mbtime: {:.3f}'
                      ''.format(bidx, len(dloader), epoch, loss.item(),
                                timings[-1], np.mean(timings)))
                writer.add_scalar('training/loss', loss.item(), global_step)
                writer.add_histogram('training/lps_x', lps_x.cpu().data,
                                     global_step, bins='sturges')
                writer.add_histogram('training/lps_y', lps_y.cpu().data,
                                     global_step, bins='sturges')
                writer.add_histogram('training/pred_y', y_.cpu().data,
                                     global_step, bins='sturges')
            global_step += 1
        va_losses = eval_epoch(va_dloader, model, criterion, epoch, writer, 
                               opts.save_freq, device)
        mva_loss = np.mean(va_losses)
        if min_va_loss > mva_loss:
            print('Val loss improved {:.3f} --> {:.3f}'.format(min_va_loss,
                                                               mva_loss))
            min_va_loss = mva_loss
            torch.save(model.state_dict(), 
                       os.path.join(opts.save_path,
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
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='lpsrnn_ckpt')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--maxlen', type=int, default=None)
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


    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))

    train(opts)
