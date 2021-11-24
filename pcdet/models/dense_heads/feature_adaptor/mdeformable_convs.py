import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

try:
    # from pcdet.ops.DeformableConvolutionV2PyTorch.modules.deform_conv import DeformConv
    from pcdet.ops.DeformableConvolutionV2PyTorch.modules.modulated_deform_conv import ModulatedDeformConv 
except:
    print("Deformable Convolution not built!")


class FeatureAdaption(nn.Module):
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
                 activate_out=True):
        super(FeatureAdaption, self).__init__()
        offset_mask_channels = kernel_size * kernel_size * (2+1)
        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            deformable_groups * offset_mask_channels, 
            kernel_size=kernel_size,
            padding=(kernel_size-1) // 2,
            stride=1, 
            bias=True)
        self.conv_adaption = ModulatedDeformConv(
            in_channels,
            out_channels,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups,
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()
        self.activate_out = activate_out

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
        if self.activate_out:
            x = self.relu(self.conv_adaption(x, offset, mask))
        else:
            x = self.conv_adaption(x, offset, mask)
            
        return x