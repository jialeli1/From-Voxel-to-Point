import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
# from ....utils import pc_utils # 这种knn实现太占显存了
from ....ops.pointnet2.pointnet2_batch import pointnet2_utils as pointnet2_batch_utils


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class BEVGridPooling(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # channel compression for point bevfeature
        in_bev_features = self.model_cfg.IN_CHANNELS
        compressed_bev_features = self.model_cfg.OUT_CHANNELS

        if in_bev_features != compressed_bev_features:
            self.point_bev_feature_compress = nn.Sequential(
                nn.Linear(in_bev_features, compressed_bev_features, bias=False),
                nn.BatchNorm1d(compressed_bev_features, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        else:
            self.point_bev_feature_compress = nn.Sequential()

        self.num_point_bev_features = compressed_bev_features
        

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0).contiguous()  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features


    def forward(self, batch_dict, keypoints):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            out_point_bev_features: (B, num_keypoints, c)
        """
        # 先产生点
        # keypoints = self.get_sampled_points(batch_dict)
        batch_size = batch_dict['batch_size']
        batch_size, num_keypoints, _ = keypoints.shape
        
        # print('==> keypoints.shape: ', keypoints.shape)
        # torch.Size([2, 27648, 3]), 2768 = 128*216
        # print('==> spatial_features_before_head.shape: ', batch_dict['spatial_features_before_head'].shape)
        # torch.Size([B, 128, 200, 176])

        # 先pool bev上的特征用于后面的roi pooling, 这里先不使用
        point_bev_features = self.interpolate_from_bev_features(
            keypoints=keypoints, 
            bev_features=batch_dict['spatial_features_before_head'], 
            batch_size=batch_dict['batch_size'],
            bev_stride=batch_dict['spatial_features_stride']
        ) 
        # print('==> point_bev_features.shape: ', point_bev_features.shape)
        # torch.Size([2, 27648, 128]), 27648 = 128 * 216

        # 完成通道压缩
        # (batch_size, num_keypoints, c_bev) -> (batch_size*num_keypoints, c_bev) -> (batch_size*num_keypoints, c_bev')
        out_point_bev_features = self.point_bev_feature_compress(
            point_bev_features.view(-1, point_bev_features.shape[-1])
        )
        # print('==> out_point_bev_features.shape: ', out_point_bev_features.shape)

        out_point_bev_features = out_point_bev_features.view(batch_size, num_keypoints, out_point_bev_features.shape[-1])
        # print('==> out_point_bev_features.shape: ', out_point_bev_features.shape)
        # assert torch.equal(point_bev_features, out_point_bev_features) == True

        return out_point_bev_features
