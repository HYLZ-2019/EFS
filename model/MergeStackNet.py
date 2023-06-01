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
    return x * 2 / 255 - 1

# Y: [16, 235] Cr,Cb: [16, 240]
def denorm_ycrcb(x):
    # from [-1, 1] to range
    y = (x + 1.0) * 219.0 / 2.0
    y[:, 1, :, :] = (x[:, 1, :, :] + 1.0) * 224.0 / 2.0
    y[:, 2, :, :] = (x[:, 2, :, :] + 1.0) * 224.0 / 2.0
    return y

def norm_ycrcb(x):
    # to [-1, 1]
    y = (x - 16.0) / 219.0 * 2 - 1
    y[:, :, 1, :, :] = (x[:, :, 1, :, :] - 16.0) / 224.0 * 2 - 1
    y[:, :, 2, :, :] = (x[:, :, 2, :, :] - 16.0) / 224.0 * 2 - 1
    return y


class MergeStackNet(nn.Module):    
    def __init__(self, stacksize):
        super(MergeStackNet, self).__init__()
        self.StackSize = stacksize

    # conv-img:
        self.conv_1 = nn.Sequential(
            Conv2D(self.StackSize*3, 64, 3),
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
            Conv2D(128*3, self.StackSize, 1, padding=0)
        )

    # prediction
        self.predWeight = nn.Sequential(
            ResidualBlock(channel_num=self.StackSize),
            ResidualBlock(channel_num=self.StackSize)
        )

        self.weightsToDepth = nn.Sequential(
            ResidualBlock(channel_num=self.StackSize*3),
            Conv2D(self.StackSize*3, 1, 1, stride=1, padding=0)
        )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.merge = nn.Sequential(
            Conv2D(self.StackSize*3, 32, 1, stride=1, padding=0),
            Conv2D(32, 3, 1, stride=1, padding=0),
        )

    def forward(self, stack, evstack):

        # stack.shape = [batch_size, self.StackSize, 3, crop_size, crop_size]        
        bs, ss, ch, h, w = stack.shape
        stack = norm(stack)
        stack_fold = torch.reshape(stack, (bs, ss*ch, h, w))

        c1 = self.conv_1(stack_fold)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)

        ce1 = self.conv_e1(evstack)
        ce2 = self.conv_e2(ce1)
        ce3 = self.conv_e3(ce2)

        m3 = torch.cat([c3, ce3], dim=1)        
        dc2 = self.deconv_2(m3)  
        m2 = torch.cat([c2, ce2, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, ce1, dc1], dim=1)
        c6 = self.conv_6(m1)

        weights = self.softmax(self.predWeight(c6))
        # weights: [bs, ss, h, w]
        more_weights = torch.unsqueeze(weights, dim=2)

        attentioned_images = more_weights * stack

        #results = self.merge(attentioned_images)

        # softmax 产生的权重之和是1
        #out = denorm(self.tanh(results))
        #print(attentioned_images.shape)
        #attentioned_stack = torch.reshape(attentioned_images, (bs, ss, ch, h, w))
        out = denorm(torch.sum(attentioned_images, axis=1))

        # 要求weights里蕴含深度信息
        #depth = self.weightsToDepth(weights)
        #depth_norm = torch.sigmoid(depth)
        depth_norm = None#torch.zeros(out.shape)
                
        return out, weights, depth_norm
