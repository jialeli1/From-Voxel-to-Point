

# import logging
# from collections import defaultdict
# from enum import Enum

import numpy as np
import torch
# from det3d.core import box_torch_ops
# from det3d.models.builder import build_loss
# from det3d.models.losses import metrics
# from det3d.torchie.cnn import constant_init, kaiming_init
# from det3d.torchie.trainer import load_checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
# from det3d.models.losses.centernet_loss import FocalLoss, SmoothRegLoss, RegLoss, RegClsLoss, FastFocalLoss
# from det3d.core.utils.center_utils import ddd_decode
# from det3d.models.utils import Sequential
# from .. import builder
# from ..losses import accuracy
# from ..registry import HEADS
# import copy 
try:
    # from pcdet.ops.dcn.deform_conv import DeformConv, ModulatedDeformConvPack
    # from pcdet.ops.DeformableConvolutionV2PyTorch.modules.deform_conv import DeformConv
    from pcdet.ops.DeformableConvolutionV2PyTorch.modules.modulated_deform_conv import ModulatedDeformConv 
except:
    print("Deformable Convolution not built!")


class MdeformConvBlock(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4, 
                 ):
        super(MdeformConvBlock, self).__init__()
        offset_mask_channels = kernel_size * kernel_size * (2+1)
        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            deformable_groups * offset_mask_channels, 
            kernel_size=kernel_size,
            stride=1, 
            padding=(kernel_size-1) // 2,
            bias=True)
        self.conv_adaption = ModulatedDeformConv(
            in_channels,
            out_channels,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups,
            bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def init_weights(self):
        pass 
        """normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)
        """
    
    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        
        offset = torch.cat((o1, o2), dim=1)
        # offset = torch.cat((o1, o2), dim=1).detach()
        
        mask = torch.sigmoid(mask)

        # print('==> offset.size(): ', offset.size()) #torch.Size([1, 72, 200, 176])
        # print(offset[0, 0:18, :, :])

        # just dcn without actfunc
        x = self.conv_adaption(x, offset, mask)
            
        return x