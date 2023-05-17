import json
import torch
import numpy as np

from utils import Config
from tqdm import tqdm
from helper import *
from PIL import Image

def main():
    # 0. load configurations
    args = argument_parser()
    cfg = Config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. init dataloaders
    _, _, test_loader = init_dataloader(cfg.dataset)
    
    # 2. init model
    net = init_network(cfg.network)
    net.to_cuda()
    if args.load_path is not None:
        net.load(args.load_path)
        
    # 3. init data preparer
    _, _, test_prep = init_preparer(cfg.prepare)
    
    for idx, data in tqdm(enumerate(test_loader)):
        try:
            test_prep.prepare(data)
            pred_data = net.forward(test_prep.input_data)
            pred_img = (pred_data[0].clamp(0, 1) * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            img = Image.fromarray(pred_img)
            img.save(os.path.join(args.output_dir, test_prep.name[0]+'.png'), format='png')
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()