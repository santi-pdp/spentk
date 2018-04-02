import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import argparse
from spentk.models.baseline1 import DNN
from spentk.datasets.swin_dataset import SWinDataset
from spentk.datasets.transforms import *
import random
import os
import json
import librosa
import timeit


def train(opts):
    print('WARNING: Only baseline1 at the moment')
    model = DNN()
    if opts.cuda:
        model.cuda()
    model.load_state_dict(torch.load(opts.ckpt_file,
                                     map_location=lambda storage, loc:storage))
    model.eval()
    criterion = nn.MSELoss()
    assert os.path.exists(opts.test_file)
    wav, rate = librosa.load(opts.test_file, 16000)
    print('wav.shape: ', wav.shape)
    y = model.clean_wav(wav)

        

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
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
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
