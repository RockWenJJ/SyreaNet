import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from glob import glob
from PIL import Image


class RealTestDataset(Dataset):
    def __init__(self, data_path, img_fmt='png', max_size=1024):
        super(RealTestDataset, self).__init__()
        self.data_path = data_path
        self.img_fmt = img_fmt
        self.max_size = max_size
        
        self.images = glob(os.path.join(self.data_path, "*."+img_fmt))
        self.length = len(self.images)
        
        self.transforms = T.Compose([
            # T.Resize((256, 256)),
            T.ToTensor()])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        h, w = img.size
        h = int(h // 32) * 32
        w = int(w // 32) * 32
        img = self.transforms(T.Resize((w, h))(img))
        
        name = os.path.basename(self.images[idx])[:-4]
        
        return img, name