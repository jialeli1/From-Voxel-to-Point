import torch
import torch.nn as nn
import torch.nn.functional as F


class CornerGeometryEncodeModule(nn.Module):
    def __init__(self, up_filters, interact_filters):
        super().__init__()
        """
        up_filters: a list
        interact_filters: a list
        """
        # define corners_up_layers, (Nbox, 3, 8, 1) -> (Nbox, C, 8, 1)
        up_filters_list = [3] + up_filters
        corners_up_layers = []
        for i in range(len(up_filters_list) - 1 ):
            corners_up_layers.extend([
                nn.Conv2d(up_filters_list[i], up_filters_list[i+1], kernel_size=1, bias=False),
                nn.BatchNorm2d(up_filters_list[i+1]),
                nn.ReLU()
            ])
        self.corners_up_layer = nn.Sequential(*corners_up_layers)

        # define corner interaction, (Nbox, C, 8) -> (Nbox, C1, 1)
        corners_inter_layers = []
        inter_filters_list = [ up_filters[-1] ] + interact_filters
        for k in range(len(inter_filters_list) - 1 ):
            corners_inter_layers.extend([
                nn.Conv1d(inter_filters_list[k], inter_filters_list[k+1], kernel_size=8, bias=False),
                nn.BatchNorm1d(inter_filters_list[k+1]),
                nn.ReLU()
            ])
        self.corners_inter_layer = nn.Sequential(*corners_inter_layers)

    def forward(self, box_corners):
        """
        Args: 
            box_corners: (Nbox, 8, 3)

        Return:
            cge_features: (Nbox, C, 1)
        """
        # 1. (Nbox, 8, 3) -> (Nbox, 3, 8, 1) for conv2d
        box_corners_trans = box_corners.transpose(1,2).unsqueeze(dim=3).contiguous()
        
        # 2. up, (Nbox, 3, 8, 1) -> (Nbox, 64, 8, 1) -> (Nbox, 64, 8)
        corners_up_features = self.corners_up_layer(box_corners_trans).squeeze(dim=-1)

        # 3. interaction, (Nbox, 64, 8) -> (Nbox, 128, 1)
        corners_inter_features = self.corners_inter_layer(corners_up_features)

        # 4. output
        cge_features = corners_inter_features

        return cge_features


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, fuse_filters):
        super().__init__()
        """
        in_channels: an int
        fuse_filters: a list
        """
        fuse_filters_list = [in_channels] + fuse_filters
        fuse_layers = []
        for i in range(len(fuse_filters_list) - 1 ):
            fuse_layers.extend([
                nn.Conv1d(fuse_filters_list[i], fuse_filters_list[i+1], kernel_size=1, bias=False),
                nn.BatchNorm1d(fuse_filters_list[i+1]),
                nn.ReLU()
            ])
        self.fuse_layer = nn.Sequential(*fuse_layers)

    def forward(self, feature_list):
        # 1. cat, (B, C1+C2, 1)
        cated_features = torch.cat(feature_list, dim=1)

        # 2. fuse, (B, C3, 1)
        fused_features = self.fuse_layer(cated_features)

        # 3. output
        return fused_features