from torch.utils.data import Dataset
import numpy as np
import timeit
import glob
import pickle
import librosa
import os

def slice_signal_index(signal, window_size, stride):
    """ Slice input signal into indexes (beg, end) each
        # Arguments
            window_size: size of each slice
            stride: fraction of sliding window per window size
        # Returns
            A list of tuples (beg, end) sample indexes
    """
    assert stride > 0, stride
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    #for beg_i in range(0, n_samples - window_size, stride):	
    for beg_i in range(0, n_samples, stride):	
        if n_samples - beg_i < window_size:
            left = n_samples - beg_i
        else:
            left = window_size
        end_i = beg_i + left
        #print('slicing {} -> {} from {} total'.format(beg_i, end_i, n_samples))
        slice_ = (beg_i, end_i)
        slices.append(slice_)
    return slices

class SWinDataset(Dataset):
    """ Strided Window dataset, consuming pairs of 
        speech chunks (clean, noisy) in certain strides 
        with certain window size.
    """

    def __init__(self, data_root, split='train',
                 window=512, stride=None, rate=16000,
                 cache_path=None, transform=None):
        """
        # Arguments
            window: num of samples of waveform
            stride: num of samples to shift b/w windows. If None
                    defaults to window size.
            cache_path: if not None, a cache file will be stored
                        in this directory with record of slices.
        """
        super().__init__()
        self.split = split
        self.window = window
        self.stride = stride
        self.transform = transform
        self.rate = 16000
        if stride is None:
            self.stride = window
        if split == 'train':
            clean_path = 'clean_trainset'
            noisy_path = 'noisy_trainset'
        elif split == 'valid':
            clean_path = 'clean_valset'
            noisy_path = 'noisy_valset'
        elif split == 'test':
            clean_path = 'clean_testset'
            noisy_path = 'noisy_testset'
        else:
            raise ValueError('Unrecognized split type: ', split)
        clean_path = os.path.join(data_root, clean_path)
        noisy_path = os.path.join(data_root, noisy_path)
        self.samples = []
        # read list of clean files and map to noise files
        cwavs = glob.glob(os.path.join(clean_path, '*.wav'))
        for ci, cwav in enumerate(cwavs):
            # check the noisy counterpart exists
            bname = os.path.basename(cwav)
            nwav = os.path.join(noisy_path, bname)
            if not os.path.exists(nwav):
                raise FileNotFoundError('Noisy file {} not '
                                        'found'.format(bname))
            self.samples.append({'cpath':cwav,
                                 'npath':nwav})
        if cache_path is not None:
            cache_file = os.path.join(cache_path, 
                                      '{}_wav_slices.cache'.format(split))

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as cache_f:
                    self.samples = pickle.load(cache_f)
            else:
                self.make_slicings()
                # store the slicings
                with open(cache_file, 'wb') as cache_f:
                    pickle.dump(self.samples, cache_f)
        else:
            self.make_slicings()
        if len(self.samples) == 0:
            raise ValueError('No samples found in SWin Dataset')


    def make_slicings(self):
        """ Make wav slice indexes, to reduce samples to windows """
        samples = []
        timings = []
        beg_t = timeit.default_timer()
        for w_i, sample in enumerate(self.samples):
            cpath = sample['cpath']
            npath = sample['npath']
            cwav, rate = librosa.load(cpath, self.rate)
            nwav, rate = librosa.load(npath, self.rate)
            cslicings = slice_signal_index(cwav, self.window, self.stride)
            nslicings = slice_signal_index(cwav, self.window, self.stride)
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            for (csl, nsl) in zip(cslicings, nslicings):
                samples.append({'cpath':cpath, 'npath':npath,
                                'cslice':csl, 'nslice':nsl})
            if (w_i + 1) % 100 == 0 or \
               (w_i + 1) >= len(self.samples):
                print('Sliced {:5d}/{:5d} pairs w/ {:.2f} s/sample'
                      ''.format(w_i+1, len(self.samples),
                                np.mean(timings)))
        # replace samples with new samples sliced
        self.samples = samples
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        cslice = self.samples[index]['cslice']
        nslice = self.samples[index]['nslice']
        cpath = self.samples[index]['cpath']
        npath = self.samples[index]['npath']
        cwav, rate = librosa.load(cpath, self.rate)
        nwav, rate = librosa.load(npath, self.rate)
        beg_c, end_c = cslice
        beg_n, end_n = nslice
        csl = cwav[beg_c:end_c]
        nsl = nwav[beg_n:end_n]
        if self.transform is not None:
            return self.transform(csl), self.transform(nsl)
        else:
            return csl, nsl
            

if __name__ == '__main__':
    from transforms import *
    dset = SWinDataset('/veu/spascual/git/segan_pytorch/data/expanded_segan1_additive', 
                       cache_path='.', transform=wav2fft(logpower=True))
    print(len(dset))
    for n in range(30, 40):
        cslice, nslice = dset.__getitem__(n)
        print('cslice shape: ', cslice.shape)
        print('nslice shape: ', nslice.shape)
        print('cslice min:{}, max:{}'.format(cslice[:, 0].min(),
                                             cslice[:, 0].max()))
    

