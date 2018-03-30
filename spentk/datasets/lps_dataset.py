from torch.utils.data import Dataset
import librosa
import os

class LPSDataset(Dataset):

    def __init__(self, data_root, n_fft=257):
        super().__init__()
        raise NotImplementedError

