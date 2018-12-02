import torch
import numpy as np
import librosa
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DNN(nn.Module):
    """DNN model to deal with Log Power Spectrum bins with
        fully connected topology end-to-end. Each
        frame is mapped independently of time axis. We
        can include some input context frames to predict current
        output specifying more in_frames.
    """
    def __init__(self, num_inputs=257, hidden_size=1024, num_layers=4,
                 in_frames=1, dropout=0., no_bnorm=False):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_frames = in_frames
        # build model structure with odict and then stack sequential
        dnn_d = OrderedDict()
        for nl in range(num_layers):
            if nl == 0:
                ninputs = self.num_inputs * in_frames
            else:
                ninputs = hidden_size
            dnn_d['fc_%d'%nl] = nn.Linear(ninputs, hidden_size)
            dnn_d['prelu_%d'%nl] = nn.PReLU(hidden_size)
            if not no_bnorm:
                dnn_d['bn_%d'%nl] = nn.BatchNorm1d(hidden_size)
            if dropout > 0:
                dnn_d['dout_%d'%nl] = nn.Dropout(dropout)
        dnn_d['out_fc'] = nn.Linear(hidden_size, self.num_inputs)
        # No enforce since no +1 in normalization now
        # enforce strictly positive outputs
        #dnn_d['out_softplus'] = nn.Softplus()
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
        X_ = librosa.stft(x, n_fft=n_fft)
        #X_mag = np.log(np.abs(X_) ** 2 + 1)
        X_mag = np.log(np.abs(X_) ** 2)
        X_pha = np.angle(X_)
        X_mag = Variable(torch.FloatTensor(X_mag)).t()
        if cuda:
            X_mag = X_mag.cuda()
        if self.in_frames > 1:
            expns = []
            F_dim = int(X_mag.size(1))
            r = self.in_frames
            z_t = torch.zeros(r // 2, F_dim)
            if cuda:
                z_t = z_t.cuda()
            z_t = Variable(z_t)
            p_mag = torch.cat((z_t, X_mag, z_t), dim=0)
            for n in range(0, p_mag.size(0) - (r - 1)):
                mag_expn = p_mag[n:n+r, :].contiguous().view(1,
                                                             -1)
                expns.append(mag_expn)
            expns = torch.cat(expns, dim=0)
            X_mag = expns
        pred_mag = self.dnn(X_mag)
        pred_mag = pred_mag.cpu().data.numpy()
        #pred_mag = np.exp(pred_mag) - 1
        pred_mag = np.exp(pred_mag)
        # trim negative if available
        #pred_mag[np.where(pred_mag < 0)] = 0
        pred_mag = np.sqrt(pred_mag).T
        X_back = pred_mag * np.exp(1j * X_pha)
        Y = librosa.istft(X_back)
        return Y

if __name__ == '__main__':
    dnn = DNN()
    #dnn.eval()
    print(dnn)
    x = Variable(torch.randn(2, 257))
    y = dnn(x)
    print('y size: ', y.size())
