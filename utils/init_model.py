import torch
import math


def init_model(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)  # conv init
            m.bias.data.fill_(0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_()  # linear init
        elif isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight.data)  # deconv init
            m.bias.data.fill_(0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)  # batch init
            torch.nn.init.constant_(m.bias.data, 0.0)
    return model
