import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import soundfile as sf
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
    for t_i, test_file in enumerate(opts.test_files, start=1):
        basename = os.path.basename(test_file)
        wav, rate = librosa.load(test_file, 16000)
        out_file = os.path.join(opts.save_path, basename)
        print('Cleaning wav {}/{}: {} -> {}'.format(t_i, len(opts.test_files),
                                                    test_file, 
                                                    out_file))
        y = model.clean_wav(wav)
        sf.write(out_file, y, 16000,'PCM_16')
        #librosa.output.write_wav(out_file, y, 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_freq', type=int, default=100, help='Num of '
                        'batches to save/log (Def: 10)')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='baseline1_clean_utts')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--test_files', type=str, nargs='+', default=None)
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

    train(opts)
