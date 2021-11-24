import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
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


class LateralBottomResBlock(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, nsample,
        lateral_downsample_times, lateral_channels, 
        bottom_downsample_times, bottom_channels,
        out_channels):
        super().__init__()

        self.voxel_size=voxel_size
        self.lateral_downsample_times = lateral_downsample_times
        self.bottom_downsample_times = bottom_downsample_times
        self.point_cloud_range = point_cloud_range
        self.nsample = nsample

        # interpolation only when bottom_channels = -1, and bottom_point_feats is None
        if bottom_channels != -1:
            self.net = nn.Sequential(
                nn.Linear(lateral_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            )
            self.downsample = nn.Sequential(
                nn.Linear(bottom_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            ) 
            self.relu = nn.ReLU()

    def forward(self, lateral_sp_voxel, bottom_point_feats, bottom_point_coords, batch_size):
        """
        # 输出是以bottom对齐的，插值也是从lateral往bottom上插值
        Args:
            lateral_sp_voxel: SPTensor
                indices: (VN1+VN2+..., 4), [bs_idx, idxz, idxy, idxx]
                features: (VN1+VN2+..., VC), VC channel voxel features
            bottom_point_feats: (PN1+PN2+..., PC), PC channel point features
            bottom_point_coords: (PN1+PN2+..., 4), [bs_idx, x, y, z]
        Returns:
            x_out: (PN1+PN2+..., PC_{out}), new_point_feats, aligned to bottom
            x_coords: (PN1+PN2+..., 4), [bs_idx, x, y, z]
        """
        # 1. get lateral_point_feats
        lateral_voxel_feats = lateral_sp_voxel.features
        lateral_voxel_coords = lateral_sp_voxel.indices[:, 1:4]
        lateral_voxel_bs_idx = lateral_sp_voxel.indices[:, 0].long()
        bottom_point_xyz = bottom_point_coords[:, 1:4] # (PN1+PN2+..., 3)
        bottom_point_bs_idx = bottom_point_coords[:, 0].long()

        # 2. get lateral_point_xyz
        lateral_voxel_xyz = common_utils.get_voxel_centers(
            voxel_coords=lateral_voxel_coords,
            downsample_times=self.lateral_downsample_times,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        ) # (VN1 + VN2+..., 3)


        # 3. interpolation in each img
        # bottom(xyz) -> lateral(new_xyz): x_identity
        # (PN1+PN2+..., PC) -> (VN1+VN2+..., PC)
        new_feats_list = []
        for bs_idx in range(batch_size):
            voxel_bs_mask = (lateral_voxel_bs_idx == bs_idx)
            point_bs_mask = (bottom_point_bs_idx == bs_idx)

            single_interpolate_feats = pointnet2_batch_utils.top3_interpolate(
                xyz=lateral_voxel_xyz[voxel_bs_mask], 
                new_xyz=bottom_point_xyz[point_bs_mask], 
                feats=lateral_voxel_feats[voxel_bs_mask],
                nsamples=self.nsample)
            # shape should be (VN1 / VN2 / /..., PC)
            new_feats_list.append(single_interpolate_feats)
        lateral_new_feats = torch.cat(new_feats_list, dim=0)

        # 4. get x_dentity, x_residual, x_out_feats
        if bottom_point_feats is not None:
            x_identity = bottom_point_feats
            x_residual = lateral_new_feats

            # 5. residual block, post act
            x_out_feats = self.relu(  self.net(x_residual) + self.downsample(x_identity) )
        else:
            x_out_feats = lateral_new_feats
            
        x_out_coords = torch.cat( (bottom_point_bs_idx.unsqueeze(dim=-1).float(), bottom_point_xyz), dim=1 )

        # check.
        # assert x_out_feats.shape[0] == x_out_coords.shape[0]
        # assert (x_out_coords - bottom_point_coords).sum() == 0

        return x_out_coords, x_out_feats


class ResidualVoxelToPointDecoder(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        # init_block
        InitBlock_cfg = self.model_cfg.INIT_BLOCK
        self.init_x_source = InitBlock_cfg.SOURCE
        self.decode_block_init = LateralBottomResBlock( 
                voxel_size=self.voxel_size, 
                point_cloud_range=self.point_cloud_range, 
                nsample=InitBlock_cfg.NSAMPLE,
                lateral_downsample_times=InitBlock_cfg.LATERAL_DOWNSAMPLE_FACTOR, 
                lateral_channels=InitBlock_cfg.LATERAL_CHANNELS, 
                bottom_downsample_times=InitBlock_cfg.BOTTOM_DOWNSAMPLE_FACTOR, 
                bottom_channels= -1, # interpolation only
                out_channels=InitBlock_cfg.OUT_CHANNELS
            )
        prefix_bottom_channels = InitBlock_cfg.OUT_CHANNELS

        # other blocks
        DeBlocks_cfg = self.model_cfg.DECODE_BLOCKS
        self.downsample_times_map = {}
        self.decode_blocks_map = nn.ModuleDict()
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            cur_cfg = DeBlocks_cfg[src_name]
            self.downsample_times_map[src_name] = cur_cfg.LATERAL_DOWNSAMPLE_FACTOR
            self.decode_blocks_map[src_name] = LateralBottomResBlock( 
                voxel_size=self.voxel_size, 
                point_cloud_range=self.point_cloud_range, 
                nsample=cur_cfg.NSAMPLE,
                lateral_downsample_times=cur_cfg.LATERAL_DOWNSAMPLE_FACTOR, 
                lateral_channels=cur_cfg.LATERAL_CHANNELS, 
                bottom_downsample_times=cur_cfg.BOTTOM_DOWNSAMPLE_FACTOR, 
                bottom_channels=prefix_bottom_channels,
                out_channels=cur_cfg.OUT_CHANNELS
            )
            prefix_bottom_channels = cur_cfg.OUT_CHANNELS


        # out block
        # no interpolation
        OutBlock_cfg = self.model_cfg.OUT_BLOCK
        self.decode_block_out = nn.Sequential(
            nn.Linear(prefix_bottom_channels, OutBlock_cfg.OUT_CHANNELS, bias=False),
            nn.BatchNorm1d(OutBlock_cfg.OUT_CHANNELS, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        prefix_bottom_channels = OutBlock_cfg.OUT_CHANNELS


        self.num_point_features = prefix_bottom_channels
        self.num_point_features_before_fusion = prefix_bottom_channels
       

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints


    def forward(self, batch_dict):
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
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)
        batch_size, num_keypoints, _ = keypoints.shape
        
        # batch format to stack format
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        out_point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)
        

        # init
        point_x_init_coords, point_x_init_features = self.decode_block_init(
            lateral_sp_voxel=batch_dict['multi_scale_3d_features'][self.init_x_source], 
            bottom_point_feats=None, # set None for interpolation only
            bottom_point_coords=out_point_coords, 
            batch_size=batch_size
        )


        point_z4_coords, point_z4_features = self.decode_blocks_map['x_conv4'](
            lateral_sp_voxel=batch_dict['multi_scale_3d_features']['x_conv4'], 
            bottom_point_feats=point_x_init_features, 
            bottom_point_coords=point_x_init_coords, 
            batch_size=batch_size
        )


        point_z3_coords, point_z3_features = self.decode_blocks_map['x_conv3'](
            lateral_sp_voxel=batch_dict['multi_scale_3d_features']['x_conv3'], 
            bottom_point_feats=point_z4_features, 
            bottom_point_coords=point_z4_coords, 
            batch_size=batch_size
        )


        point_z2_coords, point_z2_features = self.decode_blocks_map['x_conv2'](
            lateral_sp_voxel=batch_dict['multi_scale_3d_features']['x_conv2'], 
            bottom_point_feats=point_z3_features, 
            bottom_point_coords=point_z3_coords, 
            batch_size=batch_size
        )


        point_z1_coords, point_z1_features = self.decode_blocks_map['x_conv1'](
            lateral_sp_voxel=batch_dict['multi_scale_3d_features']['x_conv1'], 
            bottom_point_feats=point_z2_features, 
            bottom_point_coords=point_z2_coords, 
            batch_size=batch_size
        )


        point_z0_coords = point_z1_coords
        point_z0_features = self.decode_block_out(point_z1_features)


        # assert (point_z0_coords - out_point_coords).sum() == 0.

        batch_dict['point_features'] = point_z0_features # (BN, C)
        batch_dict['point_coords'] = point_z0_coords  # (BxN, 4)

        return batch_dict
