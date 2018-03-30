import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self):
        super().__init__()

