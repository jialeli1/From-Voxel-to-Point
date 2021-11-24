import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from .roi_head_template import RoIHeadTemplate

class PointRCNNIoUHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

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
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]
        # 这个cls_layer是否还需要分类别呢？
        # CLASS_AGNOSTIC = True，所以self.num_class应该一直是1的
        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        # 这个reg_layer好像是分类别进行的？
        # CLASS_AGNOSTIC = True, 所以还是不分类别的
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
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

    def roipool3d_gpu(self, batch_dict):
        """
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
        # print('==> 1. point_features.shape: ', point_features.shape) # (32768, 128), bs=2
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
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
        # print('==> batch_point_features.shape: ', batch_point_features.shape) # (2, 16384, 130), 因为增加了scores和depth
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
        # print('==> pooled_features.shape: ', pooled_features.shape) # (256, 512, 133), 因为增加了xyz到前面吗
        return pooled_features

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment

    @torch.no_grad()
    def generate_rcnn_iouscore_label(self, rcnn_cls, rcnn_reg, batch_dict):
        """
        Args：
            rcnn_cls: (BN, num_class)
            rcnn_reg: (BN, code_size)
            batch_dict:
                roi_labels: (B, N), 这个一定要用更新后的
        return：
            rcnn_cls_labels: (B, N)
        """
        batch_size = batch_dict['batch_size']

        # 1. 先解出预测得rcnn box, 一定要先clone().detach()
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=batch_size, rois=batch_dict['rois'], 
            cls_preds=rcnn_cls.clone().detach(), box_preds=rcnn_reg.clone().detach()
        )
        # print('==> 0.batch_box_preds.shape: ', batch_box_preds.shape) #(B, N, 7)

        # 3. 分batch，分类别的计算3D iou
        batch_gt_boxes = batch_dict['gt_boxes'] # (B, N, c)
        batch_roi_labels = batch_dict['roi_labels'] # (B, N) # 这个一定要用更新后的
        # print('==> 1.batch_gt_boxes.shape: ', batch_gt_boxes.shape)
        # print('==> 2.batch_roi_labels.shape: ', batch_roi_labels.shape)

        rcnn_iou3d_list = []
        for bs_idx in range(batch_size):
            cur_box_preds = batch_box_preds[bs_idx]
            cur_gt_boxes = batch_gt_boxes[bs_idx]
            cur_roi_labels = batch_roi_labels[bs_idx]
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_box_preds, roi_labels=cur_roi_labels, 
                gt_boxes=cur_gt_boxes[:, 0:7], gt_labels=cur_gt_boxes[:, -1].long()
            )
            rcnn_iou3d_list.append(max_overlaps)
            # print('==> max_overlaps.shape: ', max_overlaps.shape) #(N, )
        
        batch_rcnn_ious = torch.stack(rcnn_iou3d_list, dim=0) #(B, N)
        
        # 4. 然后需要直接在这对iou划分，制作cls_label
        iou_bg_thresh = self.model_cfg.TARGET_CONFIG.CLS_BG_THRESH
        iou_fg_thresh = self.model_cfg.TARGET_CONFIG.CLS_FG_THRESH
        fg_mask = batch_rcnn_ious > iou_fg_thresh
        bg_mask = batch_rcnn_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        batch_cls_labels = (fg_mask > 0).float()
        batch_cls_labels[interval_mask] = \
            (batch_rcnn_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        
        # 5. 再记录一下样本，观察训练过程
        distribution_dict = {}
        num_sample_fg = fg_mask.float().sum()
        num_sample_bg = bg_mask.float().sum()
        num_sample_inter = interval_mask.float().sum()
        distribution_dict['num_sample_fg'] = num_sample_fg / batch_size
        distribution_dict['num_sample_bg'] = num_sample_bg / batch_size
        distribution_dict['num_sample_inter'] = num_sample_inter / batch_size
        
        # 输出
        rcnn_cls_labels = batch_cls_labels
        
        return rcnn_cls_labels, distribution_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # 这里重新赋值是因为assign_targets里面会对 rois 进行重新采样，此时rois已经发生改变, 故更新之.
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)

        # 难道这里的 point_features 是包含了xyz的吗？
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3).contiguous()
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if self.training and self.predict_boxes_when_training:
            new_rcnn_cls_labels, new_distribution_dict = self.generate_rcnn_iouscore_label(
                rcnn_cls=rcnn_cls, rcnn_reg=rcnn_reg, batch_dict=batch_dict
            )
            # 先检查一下之前的target分配，再更新为rcnn_iouscore
            assert targets_dict['rcnn_cls_labels'] == None
            targets_dict['rcnn_cls_labels'] = new_rcnn_cls_labels
            targets_dict['distribution_dict'].update(new_distribution_dict)

        # 这里始终希望使用roi_labels作为最终的类别标签，
        # 使用rcnn_cls_labels作为检测置信度，以及negative result 过滤
        # 强置设置为True
        batch_dict['has_class_labels'] = True 

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        return batch_dict
