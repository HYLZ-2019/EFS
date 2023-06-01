import torch
import torchvision.models as models


class myVGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(myVGG19, self).__init__()
        vgg_pretrained_features = models.vgg19( weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 24):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 33):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_conv3_3 = self.slice3(self.slice2(self.slice1(X)))
        h_conv5_3 = self.slice5(self.slice4(h_conv3_3))
        return h_conv3_3, h_conv5_3