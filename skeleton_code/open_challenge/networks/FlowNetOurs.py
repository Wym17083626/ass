import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetOurs(nn.Module):
    def __init__(self, args, input_channels = 12,batchNorm=True, div_flow=20):
        super(FlowNetOurs, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(770, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv7 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv8 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4)
        )

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        flow5 = self.conv6(conv5)
        deconv1 = torch.cat((conv4, self.deconv1(conv5), self.upsample1(flow5)), dim=1)
        flow4 = self.conv7(deconv1)
        deconv2 = torch.cat((conv3, self.deconv2(deconv1), self.upsample2(flow4)), dim=1)
        flow3 = self.conv8(deconv2)
        deconv3 = torch.cat((conv2,self.deconv3(deconv2),self.upsample3(flow3)), dim=1)
        flow2 = self.conv9(deconv3)

        if self.training:
            return flow2,flow3,flow4,flow5
        else:
            return flow2*self.div_flow

