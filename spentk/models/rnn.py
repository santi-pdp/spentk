import torch
import numpy as np
import librosa
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LPSRNN(nn.Module):
    """ RNN model to deal with Log Power Spectrum bins. """
    def __init__(self, num_inputs=257, rnn_size=512,
                 hidden_size=1024, 
                 dropout=0., no_bnorm=False):
        super().__init__()
        self.num_inputs = num_inputs
        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        # Build RNN layer
        self.rnn = nn.GRU(num_inputs, rnn_size, batch_first=True,
                         bidirectional=True)
        # Build MLP on top of it
        self.mlp = nn.Sequential(
            nn.Linear(2 * rnn_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_inputs)
        )

    def forward(self, x, state=None):
        """ Forward LPS frames through RNN + MLP
            #Arguments
                x: batch of LPS frame of dim [B, T, self.num_inputs]
        """
        ht, state = self.rnn(x, state)
        return self.mlp(ht), state

    def clean_wav(self, x, wsize=512, stride=None,
                  n_fft=512, cuda=False):
        """ Clean an input waveform, adapting
            it to forward through model and then
            connecting the chunks again
        """
        from scipy.fftpack import fft
        from scipy.signal import istft
        assert isinstance(x, np.ndarray), type(x)
        if stride is None:
            stride = wsize
        phases = []
        mags = []
        X_ = librosa.stft(x, n_fft=n_fft)
        X_mag = np.log(np.abs(X_) ** 2).T
        X_pha = np.angle(X_)
        X_mag = torch.FloatTensor(X_mag)
        if cuda:
            X_mag = X_mag.cuda()
        X_mag = X_mag.unsqueeze(0)
        pred_mag, state = self.forward(X_mag)
        pred_mag = pred_mag.squeeze(0).cpu().data.numpy()
        pred_mag = np.exp(pred_mag)
        pred_mag = np.sqrt(pred_mag).T
        X_back = pred_mag * np.exp(1j * X_pha)
        Y = librosa.istft(X_back)
        return Y

if __name__ == '__main__':
    rnn = LPSRNN()
    x = torch.randn(1, 100, 257)
    y, _ = rnn(x)
    print('y size: ', y.size())
    x = np.ones((16000))
    xc = rnn.clean_wav(x)
    print('xc shape: ', xc.shape)
