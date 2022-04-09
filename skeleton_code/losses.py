import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EPELoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(EPELoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['EPE'],

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        assert output.shape == target.shape, (output.shape, target.shape)
        epevalue = torch.norm(target-output,p=2,dim=1)
        epevalue = torch.mean(epevalue)
        return [epevalue]


class MultiscaleLoss(nn.Module):
    def __init__(self, args):
        super(MultiscaleLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Multiscale'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            if(i==0):
                epevalue += torch.mean(torch.norm(target_-output_,p=2,dim=1))
            elif(i==1):
                epevalue += 0.5*torch.mean(torch.norm(target_-output_,p=2,dim=1))
            elif(i==2):
                epevalue += 0.25*torch.mean(torch.norm(target_-output_,p=2,dim=1))		
        return [epevalue]
