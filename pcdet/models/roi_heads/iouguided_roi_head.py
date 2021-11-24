import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils, box_utils
from .roi_withiou_head_template import RoIWithIoUHeadTemplate
from .feature_adaptor.nn_modules import CornerGeometryEncodeModule, FeatureFusionModule
from pcdet.models.backbones_3d.pfe.bev_grid_pooling import BEVGridPooling

class IoUGuidedRoIHead(RoIWithIoUHeadTemplate):
    """
    multi-steam roi-align & iou-alignment 
    """
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # bev steam 
        self.bev_grid_pool_layer = BEVGridPooling(
            model_cfg=self.model_cfg.BEV_GRID_POOL, 
            point_cloud_range=point_cloud_range, 
            voxel_size=voxel_size)
        channel_point_bev_feats = self.bev_grid_pool_layer.num_point_bev_features


        # point steam 
        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )

        use_bn = self.model_cfg.USE_BN

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out*2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        self.SA_modules = nn.ModuleList()
        channel_in = c_out
        for k in range(self.model_cfg.ROI_GRID_POOL.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.ROI_GRID_POOL.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.ROI_GRID_POOL.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.ROI_GRID_POOL.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.ROI_GRID_POOL.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = channel_out
        

        # compress the bev&point steams 
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE # an int number
        # pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * channel_in
        # point + bev
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * (channel_in + channel_point_bev_feats)
        grid_interact_fc_list = []
        
        for k in range(0, self.model_cfg.GRID_INTERACT.INTERACT_FILTERS.__len__()):
            grid_interact_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.GRID_INTERACT.INTERACT_FILTERS[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_INTERACT.INTERACT_FILTERS[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.GRID_INTERACT.INTERACT_FILTERS[k]

            if k != self.model_cfg.GRID_INTERACT.INTERACT_FILTERS.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                grid_interact_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.grid_interact_fc_layer = nn.Sequential(*grid_interact_fc_list)


        # corner geometry embeddings steam
        self.CGE_module = CornerGeometryEncodeModule(
            up_filters=self.model_cfg.CGE_MODULE.UP_FILTERS, 
            interact_filters=self.model_cfg.CGE_MODULE.INTERACT_FILTERS
        )


        # Combine the grid point features and corner geometry features.
        num_features_to_fuse = self.model_cfg.GRID_INTERACT.INTERACT_FILTERS[-1] + self.model_cfg.CGE_MODULE.INTERACT_FILTERS[-1]
        self.feature_fusion = FeatureFusionModule(
            in_channels=num_features_to_fuse,
            fuse_filters=self.model_cfg.FUSE_FILTERS
        )

        num_shared_features = self.model_cfg.FUSE_FILTERS[-1]
        self.cls_layers = self.make_fc_layers(
            input_channels=num_shared_features, 
            output_channels=self.num_class, 
            fc_list=self.model_cfg.CLS_FC
        )
        # an additional channel for iou reg
        self.reg_layers = self.make_fc_layers(
            input_channels=num_shared_features,
            output_channels=(1 + self.box_coder.code_size) * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.init_weights(weight_init='xavier')
        self.predict_boxes_when_training = self.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'rcnn_iou'

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict, batch_rois):
        """
        roipool3d_gpu performs like pointrcnn.
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']

        rois = batch_rois  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)

        batch_points = point_coords.view(batch_size, -1, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0

        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        
        # (N, 3) [x_idx, y_idx, z_idx]
        # dense_idx = faked_features.nonzero()  
        dense_idx = torch.nonzero( faked_features, as_tuple=False )  # torch 1.5+

        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points


    def forward_single_loop(self, batch_dict, batch_rois):
        """
        batch_rois: (B, roi_perimg, 7), the batch_rois is used to do feature pooling and iou alignment.
        """
        batch_size = batch_dict['batch_size']
        pooled_features = self.roipool3d_gpu(
            batch_dict=batch_dict, batch_rois=batch_rois)  # (total_rois, num_sampled_points, 3 + C)

        # pooled_features: [x,y,z, score, depth, c-dim_features]
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3).contiguous()
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]


        # get grid points
        roi_CTcorners = box_utils.boxes_to_CTcorners_3d( batch_rois[..., :7].view(-1, 7) ).contiguous()
        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois=batch_rois[..., :7], grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE)# (total_rois, 6x6x6, 3)
        
        
        # ---------------------BEV Steam----------------------
        # bev pooling by global_roi_grid_points
        # (total_rois, 6x6x6, 3) -> (B, roi_perimg, 6x6x6, 3) -> (B, roi_perimgx6x6x6, 3)
        grid_shape = global_roi_grid_points.shape
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, grid_shape[1], grid_shape[2])
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, grid_shape[2])
        # (B, roi_perimgx6x6x6, C)
        grid_bev_features = self.bev_grid_pool_layer(batch_dict, global_roi_grid_points)
        # (B, roi_perimgx6x6x6, C) -> (B, roi_perimg, 6x6x6, C) -> (Bxroi_perimg, 6x6x6, C)
        grid_shape1 = grid_bev_features.shape
        grid_bev_features = grid_bev_features.view(batch_size, -1, grid_shape[1], grid_shape1[2])
        grid_bev_features = grid_bev_features.view(grid_shape[0], grid_shape[1], grid_shape1[2])
        # (Bxroi_perimg, 6x6x6, C) -> (total_rois, C, 6x6x6)
        grid_bev_features = grid_bev_features.permute(0,2,1).contiguous()


        # ---------------------Point Steam----------------------
        # point pooling by local_roi_grid_points
        for i in range(len(self.SA_modules)):
            if i < len(self.SA_modules)-1:
                # random point aggregation by FPS in pointrcnn
                assert False 
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)
            elif i == len(self.SA_modules)-1:
                # grid point aggregation
                aggregation_center = local_roi_grid_points
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], aggregation_center)
                l_xyz.append(li_xyz)
                l_features.append(li_features)


        grid_point_features = l_features[-1] # (total_rois, C, 6x6x6)
        grid_features = torch.cat([grid_point_features, grid_bev_features] , dim=1)

        # (total_rois, C, 6x6x6) -> (total_rois, Cx6x6x6, 1) -> (total_rois, C', 1)
        batch_size_rcnn = grid_features.shape[0]
        pc_features = self.grid_interact_fc_layer(grid_features.view(batch_size_rcnn, -1, 1))


        # ---------------------CGE Steam----------------------
        cge_features = self.CGE_module(roi_CTcorners) # (bs, c, 1)


        # ---------------------Fusion----------------------
        total_features = [pc_features, cge_features]
        shared_features = self.feature_fusion(total_features)


        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_regwithiou = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1+C)
        rcnn_reg = rcnn_regwithiou[:, 1:] # (B, C)
        rcnn_iouscore = rcnn_regwithiou[:, :1] # (B, 1)

        return rcnn_cls, rcnn_reg, rcnn_iouscore


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """

        # ---------------------RoI Samping for Stage2----------------------
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )



        # ---------------------Target Assignment----------------------
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        # ---------------------First Inference----------------------
        rcnn_cls_0, rcnn_reg_0, rcnn_iouscore_0 = self.forward_single_loop(
            batch_dict=batch_dict, batch_rois=batch_dict['rois'])

        if self.training:
            targets_dict['rcnn_cls'] = rcnn_cls_0
            targets_dict['rcnn_reg'] = rcnn_reg_0
            targets_dict['rcnn_iouscore'] = rcnn_iouscore_0
            targets_dict['batch_size'] = batch_dict['batch_size']
            self.forward_ret_dict = targets_dict
        else:
            # ---------------------IoU-Alignment----------------------
            # get the un-aligned iou prediction 
            batch_cls_preds_0, batch_box_preds_0, batch_iouscore_pred_0 = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], 
                    rois=batch_dict['rois'], 
                    cls_preds=rcnn_cls_0, 
                    box_preds=rcnn_reg_0,
                    iouscore_preds=rcnn_iouscore_0,
                )

            # perform the second inference 
            rcnn_cls_1, rcnn_reg_1, rcnn_iouscore_1 = self.forward_single_loop(
                batch_dict=batch_dict, batch_rois=batch_box_preds_0.clone()
            )

            # get the aligned iou prediction 
            batch_cls_preds_1, batch_box_preds_1, batch_iouscore_preds_1 = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], 
                    rois=batch_box_preds_0.clone(), 
                    cls_preds=rcnn_cls_1, 
                    box_preds=rcnn_reg_1,
                    iouscore_preds=rcnn_iouscore_1,
                )

            # assembly of the final detection
            # You can try different combination of boxs & iouscores from the First Inference and Second Inference. 
            batch_cls_preds_final = batch_cls_preds_0
            batch_box_preds_final = batch_box_preds_0
            
            batch_iouscore_preds_1_renorm = batch_iouscore_preds_1 * 0.5 + 0.5
            batch_iouscore_preds_1_clamped = torch.clamp(batch_iouscore_preds_1_renorm, min=1e-3, max=1.0)

            # Option1: clsscore0 * iouscore1
            batch_iouscore_pred_final = torch.sigmoid(batch_cls_preds_0) * batch_iouscore_preds_1_clamped

            # Option2: clsscore0 * iouscore1.pow(6), CIA-SSD
            # batch_iouscore_preds_1_powered = batch_iouscore_preds_1_clamped.pow(6)
            # batch_iouscore_pred_final = torch.sigmoid(batch_cls_preds_0) * batch_iouscore_preds_1_powered

            # Option3: ... 


            # update batch_dict
            batch_dict['batch_cls_preds'] = batch_cls_preds_final
            batch_dict['batch_box_preds'] = batch_box_preds_final
            batch_dict['batch_iouscore_preds'] = batch_iouscore_pred_final
            
            # use roi_labels as the detection category
            # use batch_cls_preds to filter out negative results
            # use batch_iouscore_preds to perform NMS.
            batch_dict['has_class_labels'] = True 
            batch_dict['cls_preds_normalized'] = False
            

        return batch_dict
