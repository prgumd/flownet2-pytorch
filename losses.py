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

# For debbuging
import cv2
import motion_illusions.utils.flow_plot as flow_plot
from motion_illusions.utils.image_tile import ImageTile

def integrate_flow_to_future(flo):
    """
    Backward warp a list of flows into a list of flow fields
    that all represent a warp to the endpoints of the last flow field
    flo: [B, 2, N, H, W] flow, N is the number of flow fields
    """
    N = flo.shape[2]
    integrated_flow = flo[:, :, N-1, :, :].clone()
    forward_flows = [integrated_flow.clone()]
    for i in range(N-2, -1, -1):
        integrated_flow_backwarped = backward_warp(integrated_flow,
                                                   flo[:, :, i,   :, :])
        integrated_flow = flo[:, :, i, :, :] + integrated_flow_backwarped
        forward_flows.insert(0, integrated_flow.clone())

    forward_flows_reshaped = torch.stack(forward_flows, dim=0).permute(1, 2, 0, 3, 4)
    return forward_flows_reshaped

def backward_warp(x, flo):
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

        next_images_warped = backward_warp(next_images, outputs)

        # Result should be a batch size by one tensor
        photometric_loss = self.loss(next_images_warped, prev_images)

        return photometric_loss

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

        expected_brightness = (grad*output).sum(dim=1, keepdim=True)
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

class SmoothFirstOrder(nn.Module):
    def __init__(self, gamma=150):
        super(SmoothFirstOrder, self).__init__()
        # self.gamma = 150
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
        # r_grad = torch.abs(self.sobel(prev_images[:, 0, :, :].view(single_channel_image_shape)))
        # g_grad = torch.abs(self.sobel(prev_images[:, 1, :, :].view(single_channel_image_shape)))
        # b_grad = torch.abs(self.sobel(prev_images[:, 2, :, :].view(single_channel_image_shape)))
        # grad_sum_x = -self.gamma*(r_grad[:, 0, :, :] + g_grad[:, 0, :, :] + b_grad[:, 0, :, :]) / 3.0
        # grad_sum_y = -self.gamma*(r_grad[:, 1, :, :] + g_grad[:, 1, :, :] + b_grad[:, 1, :, :]) / 3.0

        # grad_sum_x_exp = torch.exp(grad_sum_x)
        # grad_sum_y_exp = torch.exp(grad_sum_y)

        flow_grad_x_sum = torch.abs(self.sobel(output[:, 0, :, :].view(single_channel_image_shape))).sum(dim=1)
        flow_grad_y_sum = torch.abs(self.sobel(output[:, 1, :, :].view(single_channel_image_shape))).sum(dim=1)

        per_pixel_loss = flow_grad_x_sum + flow_grad_y_sum

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
        self.w_smooth = 0.1
        self.loss_photo  = PhotoL1()
        self.loss_smooth = SmoothFirstOrderGradAware()
        self.loss_labels = ['PhotoSmoothFirstGradAware', 'Photo-L1', 'SmoothFirstGradAware', 'EPE']
    def forward(self, output, target, inputs):
        loss_photo = self.loss_photo(output, inputs)
        loss_smooth = self.loss_smooth(output, inputs)

        lossvalue = self.w_photo * loss_photo + self.w_smooth * loss_smooth
        epevalue = EPE(output, target)
        return [lossvalue, loss_photo, loss_smooth, epevalue]

class PhotoSmoothFirstGradAwareLossMultiLoss(nn.Module):
    def __init__ (self, args):
        super(PhotoSmoothFirstGradAwareLossMultiLoss, self).__init__()
        self.args = args
        self.loss = PhotoSmoothFirstGradAwareLoss(args)
        self.loss_labels = self.loss.loss_labels
        self.frame_weights = [1, 1, 1]

    def forward(self, output, target, inputs):
        num_outputs = int(output.shape[1] / 2)
        loss = torch.zeros(len((self.loss_labels, )))

        losses = []
        for i in range(num_outputs):
            losses.append(self.frame_weights[i] * self.loss(output[:,  2*i:2*(i+1),:,:],
                                                            target[:,:,i,:,:],
                                                            inputs[:,  3*i:3*(i+2),:,:]))
        loss_sum = torch.tensor(losses).sum(dim=0)
        return loss_sum

class PhotoSmoothFirstLoss(nn.Module):
    def __init__ (self, args):
        super(PhotoSmoothFirstLoss, self).__init__()
        self.args = args
        self.w_photo = 1.0
        self.w_smooth = 0.1
        self.loss_photo  = PhotoL1()
        self.loss_smooth = SmoothFirstOrder()
        self.loss_labels = ['PhotoSmoothFirst', 'Photo-L1', 'SmoothFirst', 'EPE']
    def forward(self, output, target, inputs):
        loss_photo = self.loss_photo(output, inputs)
        loss_smooth = self.loss_smooth(output, inputs)

        lossvalue = self.w_photo * loss_photo + self.w_smooth * loss_smooth
        epevalue = EPE(output, target)
        return [lossvalue, loss_photo, loss_smooth, epevalue]

class SupervisedBrightnessConstancyLoss(nn.Module):
    def __init__ (self, args):
        super(SupervisedBrightnessConstancyLoss, self).__init__()
        self.args = args
        self.w_supervised = 1.0
        self.w_brightness = 1.0

        self.loss_supervised = L1()
        self.loss_brightness = BrightnessConstancyL1()

        self.loss_labels = ['SupervisedBrightnessConstancy', 'L1', 'Brightness', 'EPE']

    def forward(self, output, target, inputs, scale_factor=None):
        if scale_factor is not None:
            output_scaled = output / scale_factor
        else:
            output_scaled = output

        loss_supervised = self.loss_supervised(output, target)
        loss_brightness = self.loss_brightness(output_scaled, inputs)

        lossvalue = self.w_supervised * loss_supervised + self.w_brightness * loss_brightness
        epevalue = EPE(output, target)
        return [lossvalue, loss_supervised, loss_brightness, epevalue]

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
        elif self.l_type == 'PhotoSmoothFirstLoss':
            self.loss = PhotoSmoothFirstLoss(args)
        elif self.l_type == 'SupervisedBrightnessConstancyLoss':
            self.loss = SupervisedBrightnessConstancyLoss(args)
        else:
            raise ValueError('Unrecognized loss passed to Multiscale loss')

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.multiScaleFactors = [self.startScale * (2**scale) for scale in range(self.numScales)]
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
                output_scaled_ = output_ / self.multiScaleFactors[i]
                loss = self.loss(output_scaled_, scaled_inputs)
            elif self.l_type == 'PhotoSmoothFirstGradAwareLoss':
                scaled_inputs = self.multiScales[i](inputs)
                output_scaled_ = output_ / self.multiScaleFactors[i]
                loss = self.loss(output_scaled_, target_, scaled_inputs)[0]
            elif self.l_type == 'PhotoSmoothFirstLoss':
                scaled_inputs = self.multiScales[i](inputs)
                output_scaled_ = output_ / self.multiScaleFactors[i]
                loss = self.loss(output_scaled_, target_, scaled_inputs)[0]
            elif self.l_type == 'SupervisedBrightnessConstancyLoss':
                scaled_inputs = self.multiScales[i](inputs)
                loss = self.loss(output_, target_, scaled_inputs,
                                 scale_factor=self.multiScaleFactors[i])[0]
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

class MultiScaleMultiFrame(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1', warp_to_future=False):
        super(MultiScaleMultiFrame,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.warp_to_future = warp_to_future

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
        elif self.l_type == 'PhotoSmoothFirstLoss':
            self.loss = PhotoSmoothFirstLoss(args)
        elif self.l_type == 'SupervisedBrightnessConstancyLoss':
            self.loss = SupervisedBrightnessConstancyLoss(args)
        else:
            raise ValueError('Unrecognized loss passed to Multiscale loss')

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.multiScaleFactors = [self.startScale * (2**scale) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE']
        self.frame_weights = args.frame_weights
        self.num_outputs = len(self.frame_weights)

        if self.num_outputs == 3: pass
        elif self.num_outputs == 2 and (self.l_type == 'L1' or self.l_type == 'L2'): pass
        else:
            raise ValueError('Unsupported configuration of num_outputs and l_type')

        self.tiler = ImageTile.get_instance(session='losses_test', max_width=1024*3, scale_factor=4.0)

    def forward(self, output, target, inputs, raw_data=None):
        # If output is a tuple then this is a multiscale loss where each element
        # of the tuple is one batch worth of flow at a certain scale
        # If output is not a tuple then why is multiscale loss being used?
        assert type(output) is tuple
        lossvalue = 0
        epevalue = 0

        if self.warp_to_future:
            output_maybe_warped = []
            # Warp each scale
            for output_ in output:
                output_shaped_for_warp = torch.stack(
                    [output_[:, i:i+2, :, :] for i in range(0, self.num_outputs*2, 2)], dim=0)
                output_shaped_for_warp = output_shaped_for_warp.permute(1, 2, 0, 3, 4)
                output_warped_ = integrate_flow_to_future(output_shaped_for_warp)
                output_warped_ = torch.cat([output_warped_[:, :, i, :, :] for i in range(0, self.num_outputs)], dim=1)
                output_maybe_warped.append(output_warped_)

            # Reorder output_shaped_for_warp into the list representation used
            target_maybe_warped = integrate_flow_to_future(target)

            last_input = inputs[:, 9:12, :, :]
            inputs_1 = torch.cat((inputs[:, 0:3, :, :], last_input), dim=1)
            inputs_2 = torch.cat((inputs[:, 3:6, :, :], last_input), dim=1)
            inputs_3 = torch.cat((inputs[:, 6:9, :, :], last_input), dim=1)

            # Let's make sure all of the results make sense
            # Visualize the forward warped flow fields
            # def flow_vis(flow, image=None):
            #     flow_1 = flow.detach().cpu().numpy().transpose(1, 2, 0)
            #     if image is None:
            #         image = flow_plot.visualize_optical_flow_bgr(flow_1)
            #     flow_bgr = flow_plot.dense_flow_as_quiver_plot(flow_1, image=image)
            #     return flow_bgr

            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, :2, :, :]/4))
            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, 2:4, :, :]/4))
            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, 4:6, :, :]/4))

            # # Visualize the inputs (these will look weird since the normalization is not reversed)
            # scaled_inputs_1 = self.multiScales[0](inputs_1)
            # scaled_inputs_2 = self.multiScales[0](inputs_2)
            # scaled_inputs_3 = self.multiScales[0](inputs_3)

            # def add_normalized_image(image):
            #     image_numpy = image.detach().cpu().numpy().transpose(1, 2, 0)
            #     image_unscaled = (image_numpy * 255)

            #     image_unscaled[:, :, 0] += np.min(image_unscaled[:, :, 0])
            #     image_unscaled[:, :, 1] += np.min(image_unscaled[:, :, 1])
            #     image_unscaled[:, :, 2] += np.min(image_unscaled[:, :, 2])
            #     image_unnormalized = image_unscaled.astype(np.uint8)
            #     return image_unscaled

            # # unnorm1 = add_normalized_image(scaled_inputs_1[0, :3, :, :])
            # # unnorm2 = add_normalized_image(scaled_inputs_2[0, :3, :, :])
            # # unnorm3 = add_normalized_image(scaled_inputs_3[0, :3, :, :])
            # # unnorm4 = add_normalized_image(scaled_inputs_3[0, 3:6, :, :])

            # # self.tiler.add_image(unnorm1.astype(np.uint8))
            # # self.tiler.add_image(unnorm2.astype(np.uint8))
            # # self.tiler.add_image(unnorm3.astype(np.uint8))
            # # self.tiler.add_image(unnorm4.astype(np.uint8))

            # # Add the unnormalized inputs
            # scaled_raw_data_1 = self.multiScales[0](raw_data[:, :, 0, :, :])[0]
            # scaled_raw_data_2 = self.multiScales[0](raw_data[:, :, 1, :, :])[0]
            # scaled_raw_data_3 = self.multiScales[0](raw_data[:, :, 2, :, :])[0]
            # scaled_raw_data_4 = self.multiScales[0](raw_data[:, :, 3, :, :])[0]

            # unnorm1 = scaled_raw_data_1.detach().cpu().numpy().transpose(1, 2, 0)
            # unnorm2 = scaled_raw_data_2.detach().cpu().numpy().transpose(1, 2, 0)
            # unnorm3 = scaled_raw_data_3.detach().cpu().numpy().transpose(1, 2, 0)
            # unnorm4 = scaled_raw_data_4.detach().cpu().numpy().transpose(1, 2, 0)

            # # Create averaged versions and overlay the arrows
            # unnnorm1_4 = ((unnorm1 + unnorm4) / 2).astype(np.uint8)
            # unnnorm2_4 = ((unnorm2 + unnorm4) / 2).astype(np.uint8)
            # unnnorm3_4 = ((unnorm3 + unnorm4) / 2).astype(np.uint8)

            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, :2, :, :]/4, image=unnnorm1_4.copy()))
            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, 2:4, :, :]/4, image=unnnorm2_4.copy()))
            # self.tiler.add_image(flow_vis(output_maybe_warped[0][0, 4:6, :, :]/4, image=unnnorm3_4.copy()))

            # # Do the same for target flow fields
            # target_maybe_warped_scaled_1 = self.multiScales[0](target_maybe_warped[0, :, 0, :, :])
            # target_maybe_warped_scaled_2 = self.multiScales[0](target_maybe_warped[0, :, 1, :, :])
            # target_maybe_warped_scaled_3 = self.multiScales[0](target_maybe_warped[0, :, 2, :, :])

            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_1/4))
            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_2/4))
            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_3/4))
            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_1/4, image=unnnorm1_4.copy()))
            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_2/4, image=unnnorm2_4.copy()))
            # self.tiler.add_image(flow_vis(target_maybe_warped_scaled_3/4, image=unnnorm3_4.copy()))

            # cv2.imshow('losses_test', self.tiler.compose())
            # self.tiler.clear_scene()
            # cv2.waitKey(10000)

        else:
            output_maybe_warped = output
            target_maybe_warped = target
            inputs_1 = inputs[:,:6,:,:]
            inputs_2 = inputs[:, 3:9, :, :]
            inputs_3 = inputs[:, 6:12, :, :]

        # Each member of output is a batch worth of flow at a certain scale
        for i, output_ in enumerate(output_maybe_warped):
            scaled_targets = [self.multiScales[i](target_maybe_warped[:, :, j, :, :]) for j in range(0, self.num_outputs)]

            scaled_inputs_1 = self.multiScales[i](inputs_1)
            scaled_inputs_2 = self.multiScales[i](inputs_2)
            scaled_inputs_3 = self.multiScales[i](inputs_3)

            epe = EPE(output_[:,:2,:,:], scaled_targets[0]) * self.loss_weights[i]
            for j in range(1, self.num_outputs):
                epe += EPE(output_[:,2*j:2*(j+1),:,:], scaled_targets[j]) * self.loss_weights[i]

            if self.l_type == 'L1' or self.l_type == 'L2':
                loss = self.frame_weights[0] * self.loss(output_[:,:2,:,:], scaled_targets[0])
                for j in range(1, self.num_outputs):
                    loss +=  self.frame_weights[j] * self.loss(output_[:,2*j:2*(j+1),:,:], scaled_targets[j])

            elif self.l_type == 'PhotoL1' or self.l_type == 'BrightnessConstancyL1':
                raise ValueError('PhotoL1 and BrightnessConstancyL1 not supported in MultiScaleMultiFrame')

            elif self.l_type == 'PhotoSmoothFirstGradAwareLoss':
                output_scaled_ = output_ / self.multiScaleFactors[i]
                loss =  self.frame_weights[0] * self.loss(output_scaled_[:,:2,:,:], scaled_targets[0], scaled_inputs_1)[0] + \
                        self.frame_weights[1] * self.loss(output_scaled_[:,2:4,:,:], scaled_targets[1], scaled_inputs_2)[0] + \
                        self.frame_weights[2] * self.loss(output_scaled_[:,4:,:,:], scaled_targets[2], scaled_inputs_3)[0]

            elif self.l_type == 'PhotoSmoothFirstLoss':
                raise ValueError('PhotoSmoothFirstLoss not supported in MultiScaleMultiFrame')

            elif self.l_type == 'SupervisedBrightnessConstancyLoss':
                loss = self.frame_weights[0] * self.loss(output_[:,:2,:,:], scaled_targets[0], scaled_inputs_1,
                                                         scale_factor=self.multiScaleFactors[i])[0] + \
                       self.frame_weights[1] * self.loss(output_[:,2:4,:,:], scaled_targets[1], scaled_inputs_2,
                                                         scale_factor=self.multiScaleFactors[i])[0] + \
                       self.frame_weights[2] * self.loss(output_[:,4:,:,:], scaled_targets[2], scaled_inputs_3,
                                                         scale_factor=self.multiScaleFactors[i])[0]
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
