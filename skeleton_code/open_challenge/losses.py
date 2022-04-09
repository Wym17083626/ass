import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OursLoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(OursLoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['Ours'],

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        for i,output_ in enumerate(output):
            target_=F.interpolate(target,output_.shape[2:],mode='bilinear',align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            if(i==0):
                epevalue += torch.mean(torch.norm(target_-output_,p=2,dim=1))
            elif(i==1):
                epevalue += torch.mean(torch.norm(target_-output_,p=2,dim=1))
            elif(i==2):
                epevalue += torch.mean(torch.norm(target_-output_,p=2,dim=1))
            elif(i==3):
                epevalue += torch.mean(torch.norm(target_-output_,p=2,dim=1))
        return [epevalue]
