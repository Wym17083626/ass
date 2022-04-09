import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetEncoder(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20):
        super(FlowNetEncoder,self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow      # A coefficient to obtain small output value for easy training, ignore it
        self.net = nn.Sequential(
             nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
             nn.LeakyReLU(),
             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
             nn.LeakyReLU(),
             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
             nn.LeakyReLU(),
             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(),
             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
             nn.LeakyReLU(),
             nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
             nn.Upsample(scale_factor=16)
        )

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        flow4=self.net(x)

        if self.training:
            return flow4
        else:
            return flow4 * self.div_flow

