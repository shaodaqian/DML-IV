import os
import sys
import re
import six
import math
import lmdb
import torch
import pandas as pd

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ResponseDataset(Dataset):
    def __init__(self,train_data):
        z, x, t, y=train_data
        self.nSamples = len(z)
        print('number of data: ',self.nSamples)
        self.inst=z
        self.feat=x
        self.treat=t
        self.resp=y


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        return (self.inst[index],self.feat[index], self.treat[index],self.resp[index])

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
