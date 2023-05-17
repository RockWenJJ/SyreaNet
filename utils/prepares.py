import torch
# from utils.utils import scale
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class RealTestPrepare:
    def __init__(self, mode='test'):
        self.input_data = None
        self.target_data = None
        self.pred_data = None
        self.batch_size = 1
        self.num = 1
        self.target = {}
        self.pred = {}
        self.mode = mode
        self.name = None
        
    def prepare(self, data):
        if self.mode == 'test':
            self.input_data, self.name = data
            self.input_data = self.input_data.cuda()
        else:
            raise NotImplementedError