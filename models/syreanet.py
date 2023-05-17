import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
# from .unet import UNet
# from .pix2pix_model import NLayerDiscriminator
# from .style_transferor import WCT2

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def load(self, load_path, optim=None):
        if load_path.endswith(".pth"):
            self.load_state_dict(torch.load(load_path))
            print("Loading net from %s ..." % load_path)
            self.start_ep, self.total_setps = 0, 0
            return None
        elif load_path.endswith(
                ".tar"):  # load pth.tar which contains model state_dict, optimizer state_dict, epoch and steps
            ckpt = torch.load(load_path)
            self.load_state_dict(ckpt["state_dict"])
            if optim is not None:
                optim.load_state_dict(ckpt["optimizer"])
            self.start_ep = ckpt["epoch"]
            self.total_steps = ckpt["total_steps"]
            print("Loading from %s ..." % load_path)
            print("Current Epoch %d, steps: %d." % (self.start_ep, self.total_steps))
            return optim
        else:
            print("Fail to load from %s." % load_path)

    def to_cuda(self):
        self.cuda()

    def get_parameters(self):
        return self.parameters()

    def get_state_dict(self):
        return self.state_dict()

    def to_eval(self):
        self.eval()  # switch to eval mode

    def to_train(self):
        self.train()  # switch to train mode
        

class SyreaNet(BaseNet):
    def __init__(self, model_cfg):
        super().__init__()
        self.wave_encoder = WaveEncoder(**model_cfg["layers"]["WaveEncoder"])
        self.wave_decoder = WaveDecoder(**model_cfg["layers"]["WaveDecoder"])
        self.decoder_b = WaveDecoder(**model_cfg["layers"]["Decoder_B"])
        self.decoder_t = WaveDecoder(**model_cfg["layers"]["Decoder_T"])
        
        # decoder for white point
        self.decoder_w1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_w2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_w3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.decoder_w4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, bias=True)
        
        self.mode = model_cfg["mode"]
        if self.mode == 'train':
            self.netG = UNet(model_cfg['Generator'])
            self.netD = NLayerDiscriminator(**model_cfg['Discriminator'])
            self.style_transferor = WCT2(model_path='./model_checkpoints',
                                        transfer_at={'decoder', 'encoder', 'skip'},
                                        option_unpool='cat5', verbose=False)
            
            self.set_requires_grad(self.style_transferor.encoder, False)
            self.set_requires_grad(self.style_transferor.decoder, False)
    
    def forward(self, x):
        x, skips = self.wave_encoder(x)
        x_b, outs_b = self.decoder_b(x, skips)
        x_t, outs_t = self.decoder_t(x, skips)
        x_w1 = self.decoder_w1(x)
        x_w2 = self.decoder_w2(x_w1)
        x_w3 = self.decoder_w3(x_w2)
        x_w = self.decoder_w4(self.avg_pool(x_w3))
        # x_w = self.decoder_w3(self.avg_pool(self.decoder_w2(self.decoder_w1(x))))
        skips['conv3'] = (skips['conv3'] - outs_b[3]) / (torch.maximum(outs_t[3] * self.avg_pool(x_w1),
                                                                                  torch.ones_like(outs_t[3] * 1e-3)))
        skips['conv2'] = (skips['conv2'] - outs_b[2]) / (torch.maximum(outs_t[2] * self.avg_pool(x_w2),
                                                                                  torch.ones_like(outs_t[2] * 1e-3)))
        skips['conv1'] = (skips['conv1'] - outs_b[1]) / (torch.maximum(outs_t[1] * self.avg_pool(x_w3),
                                                                       torch.ones_like(outs_t[1] * 1e-3)))
        x_cl, _ = self.wave_decoder(x, skips)
        
        return x_cl, x_b, x_t, x_w