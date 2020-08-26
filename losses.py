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
    per_pixel_norm = torch.norm(target_flow-input_flow,p=2,dim=1)

    elements_per_sample = int(per_pixel_norm.numel() / per_pixel_norm.shape[0])
    per_sample_mean = per_pixel_norm.view((per_pixel_norm.shape[0], elements_per_sample)).mean(dim=1)
    return per_sample_mean

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        per_pixel_norm = torch.abs(output-target).sum(dim=1)

        elements_per_sample = int(per_pixel_norm.numel() / per_pixel_norm.shape[0])
        per_sample_mean = per_pixel_norm.view((per_pixel_norm.shape[0], elements_per_sample)).mean(dim=1)
        return per_sample_mean

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        per_pixel_norm = torch.norm(output-target,p=2,dim=1)

        elements_per_sample = int(per_pixel_norm.numel() / per_pixel_norm.shape[0])
        per_sample_mean = per_pixel_norm.view((per_pixel_norm.shape[0], elements_per_sample)).mean(dim=1)
        return per_sample_mean

class PhotoL1(nn.Module):
    def __init__(self):
        super(PhotoL1, self).__init__()
        self.loss = L1()

    def forward(self, outputs, inputs):
        prev_images = inputs[:, 0:3, :, :]
        next_images = inputs[:, 3:6, :, :]

        next_images_warped = self.warp(next_images, outputs)

        # Result should be a batch size by one tensor
        photometric_loss = self.loss(next_images_warped, prev_images)

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

import cv2
import motion_illusions.utils.flow_plot as flow_plot
class BrightnessConstancyL1(nn.Module):
    def __init__(self):
        super(BrightnessConstancyL1, self).__init__()
        self.loss = L1()

        # Pytorch does not accelerate torchvision with GPU so do our own operations
        # Convert RGB to Gray the same way OpenCV does
        # https://docs.opencv.org/master/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
        self.rgb_to_gray = nn.Conv2d(3, 1, kernel_size=(1, 1), bias=False)
        self.rgb_to_gray.weight = torch.nn.Parameter(
            torch.FloatTensor((0.299, 0.587, 0.114)).reshape(self.rgb_to_gray.weight.shape), requires_grad=False)

        self.sobel = nn.Conv2d(2, 1, kernel_size=(3, 3), bias=False, padding=1)
        self.sobel.weight = torch.nn.Parameter(
            (1.0/8.0) * torch.FloatTensor((
                (-1, 0, 1),
                (-2, 0, 2),
                (-1, 0, 1),
                (-1, -2, -1),
                ( 0,  0,  0),
                ( 1,  2,  1)
            )).reshape((2, 1, 3, 3)),
            requires_grad=False)

    def forward(self, output, inputs):
        prev_images = inputs[:, 0:3, :, :]
        next_images = inputs[:, 3:6, :, :]

        prev_intensity = self.rgb_to_gray(prev_images)
        next_intensity = self.rgb_to_gray(next_images)

        dI = next_intensity - prev_intensity
        grad = self.sobel(next_intensity)

        expected_brightness = (grad*output).sum(axis=1, keepdim=True)
        loss_values = self.loss(expected_brightness, -dI)

        # grad_bgr = flow_plot.visualize_optical_flow_bgr(grad[0, :, :, :].abs().cpu().numpy().transpose(1, 2, 0))

        #dI_numpy = (255*(dI[0, :, :, :].abs())).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        #dI_numpy = np.concatenate((dI_numpy, dI_numpy, dI_numpy), axis=2)
        #cv2.imshow('test', np.concatenate((dI_numpy, grad_bgr), axis=1))
        # brightness_error = (expected_brightness + dI).abs()[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        # cv2.imshow('test', brightness_error)
        # cv2.waitKey(1000)

        return loss_values

class SmoothFirstOrderGradAware(nn.Module):
    def __init__(self, gamma=150):
        super(SmoothFirstOrderGradAware, self).__init__()
        self.gamma = 150
        self.sobel = nn.Conv2d(2, 1, kernel_size=(3, 3), bias=False, padding=1)
        self.sobel.weight = torch.nn.Parameter(
            (1.0/8.0) * torch.FloatTensor((
                (-1, 0, 1),
                (-2, 0, 2),
                (-1, 0, 1),
                (-1, -2, -1),
                ( 0,  0,  0),
                ( 1,  2,  1)
            )).reshape((2, 1, 3, 3)),
            requires_grad=False)

    def forward(self, output, inputs):
        prev_images = inputs[:, 0:3, :, :]
        #next_images = inputs[:, 3:6, :, :]

        single_channel_image_shape = (prev_images.shape[0], 1, prev_images.shape[2], prev_images.shape[3])
        r_grad = torch.abs(self.sobel(prev_images[:, 0, :, :].view(single_channel_image_shape)))
        g_grad = torch.abs(self.sobel(prev_images[:, 1, :, :].view(single_channel_image_shape)))
        b_grad = torch.abs(self.sobel(prev_images[:, 2, :, :].view(single_channel_image_shape)))
        grad_sum_x = -self.gamma*(r_grad[:, 0, :, :] + g_grad[:, 0, :, :] + b_grad[:, 0, :, :]) / 3.0
        grad_sum_y = -self.gamma*(r_grad[:, 1, :, :] + g_grad[:, 1, :, :] + b_grad[:, 1, :, :]) / 3.0

        grad_sum_x_exp = torch.exp(grad_sum_x)
        grad_sum_y_exp = torch.exp(grad_sum_y)

        flow_grad_x_sum = torch.abs(self.sobel(output[:, 0, :, :].view(single_channel_image_shape))).sum(dim=1)
        flow_grad_y_sum = torch.abs(self.sobel(output[:, 1, :, :].view(single_channel_image_shape))).sum(dim=1)

        per_pixel_loss = grad_sum_x_exp * flow_grad_x_sum + grad_sum_y_exp * flow_grad_y_sum

        elements_per_sample = int(per_pixel_loss.numel() / per_pixel_loss.shape[0])
        per_sample_mean = per_pixel_loss.view((per_pixel_loss.shape[0], elements_per_sample)).mean(dim=1)
        return per_sample_mean

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
        self.loss_labels = ['Photo-L1', 'EPE']
    def forward(self, output, target, inputs):
        lossvalue = self.loss(output, inputs)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class BrightnessConstancyL1Loss(nn.Module):
    def __init__ (self, args):
        super(BrightnessConstancyL1Loss, self).__init__()
        self.args = args
        self.loss = BrightnessConstancyL1()
        self.loss_labels = ['BrightnessConstancy-L1', 'EPE']
    def forward(self, output, target, inputs):
        lossvalue = self.loss(output, inputs)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class PhotoSmoothFirstGradAwareLoss(nn.Module):
    def __init__ (self, args):
        super(PhotoSmoothFirstGradAwareLoss, self).__init__()
        self.args = args
        self.w_photo = 1.0
        self.w_smooth = 4.0
        self.loss_photo  = PhotoL1()
        self.loss_smooth = SmoothFirstOrderGradAware()
        self.loss_labels = ['PhotoSmoothFirstGradAware', 'Photo-L1', 'SmoothFirstGradAware', 'EPE']
    def forward(self, output, target, inputs):
        loss_photo = self.loss_photo(output, inputs)
        loss_smooth = self.loss_smooth(output, inputs)

        lossvalue = self.w_photo * loss_photo + self.w_smooth * loss_smooth
        epevalue = EPE(output, target)
        return [lossvalue, loss_photo, loss_smooth, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm

        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        elif self.l_type == 'L2':
            self.loss = L2()
        elif self.l_type == 'PhotoL1':
            self.loss = PhotoL1()
        elif self.l_type == 'BrightnessConstancyL1':
            self.loss = BrightnessConstancyL1()
        elif self.l_type == 'PhotoSmoothFirstGradAwareLoss':
            self.loss = PhotoSmoothFirstGradAwareLoss(args)
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

        # Each member of output is a batch worth of flow at a certain scale
        for i, output_ in enumerate(output):
            target_ = self.multiScales[i](target)

            epe = EPE(output_, target_) * self.loss_weights[i]

            if self.l_type == 'L1' or self.l_type == 'L2':
                loss = self.loss(output_, target_)
            elif self.l_type == 'PhotoL1' or self.l_type == 'BrightnessConstancyL1':
                scaled_inputs = self.multiScales[i](inputs)
                loss = self.loss(output_, scaled_inputs)
            elif self.l_type == 'PhotoSmoothFirstGradAwareLoss':
                scaled_inputs = self.multiScales[i](inputs)
                loss = self.loss(output_, target_, scaled_inputs)[0]
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
