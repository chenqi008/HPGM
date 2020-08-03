import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.hooks as hooks
# import visdom 
from PIL import Image
import torchvision.transforms as transforms

class MaskedGradient(nn.Module):
    """docstring for MaskedGradient"""
    def __init__(self, opt):
        super(MaskedGradient, self).__init__()
        self.opt = opt
        # filter of image gradient
        self.dx = nn.Conv2d(in_channels=opt.nc, out_channels=opt.nc, \
                            kernel_size=(1, 3), stride=1, padding=0, bias=False, groups=3)
        self.dy = nn.Conv2d(in_channels=opt.nc, out_channels=opt.nc, \
                            kernel_size=(3, 1), stride=1, padding=0, bias=False, groups=3)

        # stable the parameter of dx and dy
        self.dx.weight.requires_grad = False
        self.dy.weight.requires_grad = False

        # initialize the weights of the kernels
        self._init_weights()

        # self.criterion = nn.L1Loss(reduction=self.opt.l1_reduction)
        self.criterion = nn.L1Loss()
        # self.criterion = nn.BCELoss(reduction='sum')
        self.gamma = 1

    def _init_weights(self):
        weights_dx = torch.FloatTensor([1, 0, -1])
        weights_dy = torch.FloatTensor([[1], [0], [-1]])
        # if not self.opt.ycbcr:
        for i in range(self.dx.weight.size(0)):
            for j in range(self.dx.weight.size(1)):
                self.dx.weight.data[i][j].copy_(weights_dx)

        for i in range(self.dy.weight.size(0)):
            for j in range(self.dy.weight.size(1)):
                self.dy.weight.data[i][j].copy_(weights_dy)

    def _normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        min_element, _ = torch.min(inputs_view, 1)
        min_element = min_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        max_element, _ = torch.max(inputs_view, 1)
        max_element = max_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = (inputs - min_element) / (max_element - min_element + eps)
        return outputs

    def _abs_normalize(self, inputs):
        eps = 1e-5
        inputs_view = inputs.view(inputs.size(0), -1)
        f_norm = torch.norm(inputs_view, 2, 1)
        f_norm = f_norm.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = inputs / (f_norm + eps)
        return outputs

    def _combine_gradient_xy(self, gradient_x, gradient_y):
        eps = 1e-4
        return torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + eps)

    def _padding_gradient(self, gradient_x, gradient_y):
        # padding with 0 to ensure the same size of the masks and images
        output_x = F.pad(gradient_x, (1, 1, 0, 0), "constant", 0)
        output_y = F.pad(gradient_y, (0, 0, 1, 1), "constant", 0)
        return output_x, output_y

    def forward(self, inputs, targets):
        # # input gradient
        inputs_grad_x = self.dx(inputs)
        inputs_grad_y = self.dy(inputs)
        
        # target gradient
        targets_grad_x = self.dx(targets.detach())
        targets_grad_y = self.dy(targets.detach())

        # padding with 0 to ensure the same size of the masks and images
        inputs_grad_x, inputs_grad_y = self._padding_gradient(inputs_grad_x, inputs_grad_y)
        targets_grad_x, targets_grad_y = self._padding_gradient(targets_grad_x, targets_grad_y)
        
        inputs_grad = self._combine_gradient_xy(inputs_grad_x, inputs_grad_y)
        targets_grad = self._combine_gradient_xy(targets_grad_x, targets_grad_y)

        # inputs_mask = self._normalize(inputs_grad)
        # targets_mask = self._normalize(targets_grad)
        targets_mask = targets_grad
        # P = mask**self.gamma
        grad_loss = self.criterion(inputs_grad, targets_mask.detach())
        
        loss = grad_loss * 1

        return loss, inputs_grad, targets_mask
