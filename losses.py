'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import transforms
import numpy as np
from PIL import Image


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class PhotoL1(nn.Module):
    def __init__(self):
        super(PhotoL1, self).__init__()
        self.loss = L1()

    def forward(self, outputs, inputs):
        prev_images = inputs[:, 0:3, :, :]
        next_images = inputs[:, 3:6, :, :]

        _, _, height, width = outputs.size()
        prev_images_resize = F.interpolate(prev_images, size=(height, width), mode='nearest')
        next_images_resize = F.interpolate(next_images, size=(height, width), mode='nearest')

        next_images_warped = self.warp(next_images_resize, outputs)
        
        diff = next_images_warped - prev_images_resize

        # Result should be a batch size by one tensor
        photometric_loss = self.loss(next_images_warped, prev_images_resize)

        return photometric_loss

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target, inputs):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target, inputs):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class PhotoL1Loss(nn.Module):
    def __init__(self, args):
        super(PhotoL1Loss, self).__init__()
        self.args = args
        self.loss = PhotoL1()
        self.loss_labels = ['PhotoL1', 'EPE']

    def forward(self, output, target, inputs):
        lossvalue = self.loss(output, inputs)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 20.0 #0.05

        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        elif self.l_type == 'L2':
            self.loss = L2()
        elif self.l_type == 'PhotoL1':
            self.loss = PhotoL1()
        else:
            raise ValueError('Unrecognized loss passed to Multiscale loss')

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE']

    def forward(self, output, target, inputs):
        # If output is a tuple then this is a multiscale loss where each element
        # of the tuple is one batch worth of flow at a certain scale
        # If output is not a tuple then why is multiscale loss being used?
        assert type(output) is tuple

        lossvalue = 0
        epevalue = 0

        # TODO we currently have hacked div_flow to 20 for training SD network
        # However for all other networks this would be 0.05, should be a parameter
        target_scaled = self.div_flow * target

        # Each member of output is a batch worth of flow at a certain scale
        for i, output_ in enumerate(output):
            target_ = self.multiScales[i](target_scaled)

            epe = EPE(output_, target_) * self.loss_weights[i]

            if self.l_type == 'L1' or self.l_type == 'L2':
                loss = self.loss(output_, target_)
            elif self.l_type == 'PhotoL1':
                scaled_inputs = self.multiScales[i](inputs)
                loss = self.loss(output_, scaled_inputs)
            else:
                raise ValueError('Unrecognized loss passed to Multiscale loss')

            loss*=self.loss_weights[i]

            #print('output {} {} {} {}'.format(i, loss, epe, self.loss_weights[i]))

            # lburner - fix memory usage bug according to GitHub issue
            # https://github.com/NVIDIA/flownet2-pytorch/issues/146#issuecomment-491171289
            # Requires setting the epevalue and lossvalue functions on first iteration
            # to make sure everything is in the correct devices memory
            # Accumulate weighted loss from each scale
            if i == 0:
                epevalue = epe
                lossvalue = loss
            else:
                epevalue += epe
                lossvalue += loss

        return [lossvalue, epevalue]
