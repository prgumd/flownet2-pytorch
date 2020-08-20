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

def L1_loss(delta):
    lossvalue = torch.abs(delta).mean()
    return lossvalue

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

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
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
        self.div_flow = 0.05
        self.rgb_max = args.rgb_max

        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target, inputs):
        lossvalue = 0
        epevalue = 0

        # lburner - fix memory usage bug according to GitHub issue
        # https://github.com/NVIDIA/flownet2-pytorch/issues/146#issuecomment-491171289

        # if type(output) is tuple:
        #     target = self.div_flow * target
        #     for i, output_ in enumerate(output):
        #         target_ = self.multiScales[i](target)
        #         epevalue += self.loss_weights[i]*EPE(output_, target_)
        #         lossvalue += self.loss_weights[i]*self.loss(output_, target_)
        #     return [lossvalue, epevalue]
        # else:
        #     epevalue += EPE(output, target)
        #     lossvalue += self.loss(output, target)
        #     return  [lossvalue, epevalue]

        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        prev_images = x[:,:,0,:,:]
        next_images = x[:,:,1,:,:]

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                if i == 0:
                    epevalue = EPE(output_, target_) * self.loss_weights[i]
                    lossvalue = self.loss(output_, target_) * self.loss_weights[i]
                else:
                    epevalue += EPE(output_, target_) * self.loss_weights[i]
                    lossvalue += self.loss(output_, target_) * self.loss_weights[i]

            photo_loss, _,_ = self.compute_photometric_loss_batch(prev_images, next_images, output)
            return [photo_loss, epevalue]
        else:
            epevalue = EPE(output, target)
            lossvalue = self.loss(output, target)
            return  [lossvalue, epevalue]





    def compute_photometric_loss_batch(self, prev_images, next_images, flow_dict):

        total_photometric_loss = 0.
        loss_weight_sum = 0.

        warped_imgs = []
        diff_maps = []
        
        for flow in flow_dict:
            _, _, height, width = flow.size()
            prev_images_resize = F.interpolate(prev_images, size=(height, width), mode='nearest')
            next_images_resize = F.interpolate(next_images, size=(height, width), mode='nearest')

            next_images_warped = self.warp(next_images_resize, flow)
            
            diff = next_images_warped - prev_images_resize

            photometric_loss = L1_loss(next_images_warped - prev_images_resize)
            total_photometric_loss += photometric_loss
            loss_weight_sum += 1.
            warped_imgs.append(next_images_warped[0])
            diff_maps.append(diff[0])

        total_photometric_loss /= loss_weight_sum

        return total_photometric_loss, warped_imgs, diff_maps


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
