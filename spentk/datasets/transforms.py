from scipy.fftpack import fft
import numpy as np
import torch

class wav2fft(object):

    def __init__(self, n_fft=512, logpower=False):
        self.n_fft = n_fft
        self.logpower = logpower

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        X = fft(x, self.n_fft)[:self.n_fft//2 + 1]
        X_mag = np.abs(X)[:, None]
        if self.logpower:
            X_mag = np.log((X_mag ** 2) + 1)
        X_pha = np.angle(X)[:, None]
        return np.concatenate((X_mag, X_pha), axis=1)
