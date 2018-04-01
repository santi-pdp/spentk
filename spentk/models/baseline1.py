import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DNN(nn.Module):
    """DNN model to deal with Log Power Spectrum bins with
        fully connected topology end-to-end, thus each
        frame is mapped independently
    """
    def __init__(self, num_inputs=257, hidden_size=1024, num_layers=4):
        """ Default parameters stand for the model in the paper """
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # build model structure with odict and then stack sequential
        dnn_d = OrderedDict()
        for nl in range(num_layers):
            if nl == 0:
                ninputs = self.num_inputs
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
                  n_fft=512):
        """ Clean an input waveform, adapting
            it to forward through model and then
            connecting the chunks again
        """
        from scipy.fftpack import fft
        assert isinstance(x, np.ndarray), type(x)
        if stride is None:
            stride = wsize
        phases = []
        mags = []
        for beg_i in range(0, x.shape[0], stride):
            x_ = x[beg_i:beg_i + wsize]
            X_ = fft(x_, n_fft)[n_fft // 2 + 1]
            X_mag = np.log(np.abs(X_) ** 2 + 1)
            X_pha = np.angle(X_)
            print('X_mag: ', X_mag.shape)
            print('X_pha: ', X_pha.shape)
            phases.append(X_pha)
            mags.append(X_mag)



if __name__ == '__main__':
    dnn = DNN()
    #dnn.eval()
    print(dnn)
    x = Variable(torch.randn(2, 257))
    y = dnn(x)
    print('y size: ', y.size())
