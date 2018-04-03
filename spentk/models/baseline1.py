import torch
import numpy as np
import librosa
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DNN(nn.Module):
    """DNN model to deal with Log Power Spectrum bins with
        fully connected topology end-to-end, thus each
        frame is mapped independently
    """
    def __init__(self, num_inputs=257, hidden_size=1024, num_layers=4,
                 in_frames=1):
        """ Default parameters stand for the model in the paper """
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # build model structure with odict and then stack sequential
        dnn_d = OrderedDict()
        for nl in range(num_layers):
            if nl == 0:
                ninputs = self.num_inputs * in_frames
            else:
                ninputs = hidden_size
            dnn_d['fc_%d'%nl] = nn.Linear(ninputs, hidden_size)
            dnn_d['prelu_%d'%nl] = nn.PReLU(hidden_size)
            dnn_d['bn_%d'%nl] = nn.BatchNorm1d(hidden_size)
        dnn_d['out_fc'] = nn.Linear(hidden_size, self.num_inputs)
        self.dnn = nn.Sequential(dnn_d)

    def forward(self, x):
        """ Forward LPS frames through DNN
            #Arguments
                x: batch of LPS frame of dim [B, self.num_inputs]
        """
        return self.dnn(x)

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
        X_ = librosa.stft(x, n_fft=n_fft, win_length=wsize, window='boxcar')
        X_mag = np.log(np.abs(X_) ** 2 + 1)
        X_pha = np.angle(X_)
        X_mag = Variable(torch.FloatTensor(X_mag)).t()
        if cuda:
            X_mag = X_mag.cuda()
        pred_mag = self.dnn(X_mag)
        pred_mag = pred_mag.cpu().data.numpy()
        pred_mag = np.exp(pred_mag) - 1
        # trim negative if available
        pred_mag[np.where(pred_mag < 0)] = 0
        pred_mag = np.sqrt(pred_mag).T
        X_back = pred_mag * np.exp(1j * X_pha)
        Y = librosa.istft(X_back, win_length=wsize, window='boxcar')
        return Y
            



if __name__ == '__main__':
    dnn = DNN()
    #dnn.eval()
    print(dnn)
    x = Variable(torch.randn(2, 257))
    y = dnn(x)
    print('y size: ', y.size())
