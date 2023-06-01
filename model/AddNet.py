import torch.nn as nn
from .BaseBlock import *
import pickle
import torch
torch.cuda.current_device()
import torch.nn.functional as F

def denorm(x):
    # from [-1, 1] to [0, 255]
    return (x + 1.0) * 255.0 / 2.0

def norm(x):
    # from [0, 255] to [-1, 1]
    return x * 2.0 / 255 - 1


class AddNet(nn.Module):    
    def __init__(self, stacksize):
        super(AddNet, self).__init__()
        self.StackSize = stacksize

    # conv-img:
        self.conv_1 = nn.Sequential(
            Conv2D(3, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )
    # conv-events:
        self.conv_e1 = nn.Sequential(
            Conv2D(self.StackSize, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_e2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_e3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )

    # deconv 
        self.deconv_2 = DeConv2D(512*2, 256)

        self.conv_5 = nn.Sequential(
            Conv2D(256*3, 128, 1, padding=0),
            Dense_block(128, 32)
        )

        self.deconv_1 = DeConv2D(256, 128)

        self.conv_6 = nn.Sequential(
            Conv2D(128*3, 64, 1, padding=0)
        )

    # prediction
        self.predConv = nn.Sequential(
            ResidualBlock(channel_num=64),
            ResidualBlock(channel_num=64),
            Conv2D(64, 3, 1, padding=0)
        )



    def forward(self, begin_img, events):

        # stack.shape = [batch_size, self.StackSize, crop_size, crop_size]

        begin_img_log = torch.where(begin_img == 0, -10, torch.log(begin_img))

        c1 = self.conv_1(begin_img_log)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)

        ce1 = self.conv_e1(events)
        ce2 = self.conv_e2(ce1)
        ce3 = self.conv_e3(ce2)

        m3 = torch.cat([c3, ce3], dim=1)        
        dc2 = self.deconv_2(m3)  
        m2 = torch.cat([c2, ce2, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, ce1, dc1], dim=1)
        c6 = self.conv_6(m1)

        pred = self.predConv(c6)

        log_sum = begin_img_log + pred
        
        lin_sum = torch.exp(log_sum)
        out = torch.clamp(lin_sum, min=0.0, max=255.0)
        #print(out.shape)
        return out
