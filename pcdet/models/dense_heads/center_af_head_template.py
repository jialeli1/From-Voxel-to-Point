import numpy as np
import torch
import torch.nn as nn

from ...utils import loss_utils, box_utils
from .keypoint_assigner.centertarget_assigner import CenterTargetAssigner

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import center_utils


class CenterAFHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, voxel_size, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training

        # actually, anchor point
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.target_assigner = self.get_target_assigner(anchor_target_cfg, voxel_size, point_cloud_range)

        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)



    def make_conv_layers(self, input_channels, output_channels, fc_list, ks_list):
        """
        Make conv2d with kernal_size 1x1 as the fc.
        Args:

        """
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv2d(pre_channel, fc_list[k], kernel_size=ks_list[k], padding=(ks_list[k]-1)//2, bias=False),
                nn.BatchNorm2d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO > 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv2d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def make_fc_head(self, input_channels, head_cfg):
        """
        Args:
            head_cfg:
                name:
        Return:
            head_fc
        """
        fc_layers = []
        pre_channel = input_channels
        # 1. conv
        fc_layers.extend([
            nn.Conv2d(in_channels=pre_channel, out_channels=head_cfg['head_conv'], kernel_size=3, padding=1, bias=False)
        ])
        pre_channel = head_cfg['head_conv']
        # 2. bn or not
        if head_cfg['use_bn']:
            if head_cfg['mod_bn']:
                fc_layers.extend([
                    nn.BatchNorm2d(head_cfg['head_conv'], eps=1e-3, momentum=0.01) 
                ])
            else:
                fc_layers.extend([
                    nn.BatchNorm2d(head_cfg['head_conv']) 
                ]) 
        # 3. activate
        fc_layers.extend([
            nn.ReLU()
        ])
        # 4. drop_out or not 
        if head_cfg['dp_ratio'] > 0:
            fc_layers.extend([
                nn.Dropout(head_cfg['dp_ratio'])
            ])
        # 5. conv
        fc_layers.extend([
            nn.Conv2d(in_channels=pre_channel, out_channels=head_cfg['out_channel'], \
                    kernel_size=head_cfg['final_kernel'], padding=(head_cfg['final_kernel']-1)//2, bias=True)
        ])
        # 6. 
        fc_layers = nn.Sequential(*fc_layers)

        # 7. init
        if 'hm' in head_cfg['name']:
            bias_value = -2.19
            self.init_weights(fc_layers, init_bias=bias_value)
            # print('==> head: %s with init_bias: %f' %(head_cfg['name'], bias_value))
        elif 'segm' in head_cfg['name']:
            pi = 0.01
            bias_value = -np.log((1 - pi) / pi) # -4.5951
            self.init_weights(fc_layers, init_bias=bias_value)
            # print('==> head: %s with init_bias: %f' %(head_cfg['name'], bias_value))
        else: 
            self.init_weights(fc_layers)
            # print('==> head: %s with init_bias: %f' %(head_cfg['name'], 0))


        return fc_layers

    def init_weights(self, modules, weight_init='xavier', init_bias=0):
        # pi = 0.01
        # init_bias = -2.19
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
                    # print('==> ini_bias: %f' %init_bias)
                    m.bias.data.fill_(init_bias)

    def get_target_assigner(self, anchor_target_cfg, voxel_size, point_cloud_range):
        if anchor_target_cfg.NAME == 'CenterTargetAssigner':
            target_assigner = CenterTargetAssigner(
                model_cfg = self.model_cfg,
                class_names=self.class_names,
                voxel_size=voxel_size,
                point_cloud_range= point_cloud_range
            )          
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'hm_loss_func',
            loss_utils.CenterNetFocalLoss()
        )
        self.add_module(
            'offset_loss_func',
            loss_utils.CenterNetResLoss(cfg=losses_cfg.OFFSET_LOSS_CONFIG)
        )
        self.add_module(
            'height_loss_func',
            loss_utils.CenterNetResLoss(cfg=losses_cfg.HEIGHT_LOSS_CONFIG)
        )
        self.add_module(
            'dim_loss_func',
            loss_utils.CenterNetResLoss(cfg=losses_cfg.DIM_LOSS_CONFIG)
        )
        self.add_module(
            'rot_loss_func',
            loss_utils.CenterNetRotBinResLoss(cfg=losses_cfg.ROT_LOSS_CONFIG)
        )
        self.add_module(
            'segm_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
            targets_dict: 
                center_map: 
                corners_map:
                boxreg_map:
                # iou_map:

        """
        targets_dict = self.target_assigner.assign_targets(
            gt_boxes
        )
        return targets_dict



    def get_loss(self):
        rpn_loss = 0

        hm_loss, tb_dict = self.get_hm_loss()
        rpn_loss += hm_loss
        
        offset_loss, offset_tb_dict = self.get_offset_loss()
        rpn_loss += offset_loss
        tb_dict.update(offset_tb_dict)

        height_loss, height_tb_dict = self.get_height_loss()
        rpn_loss += height_loss
        tb_dict.update(height_tb_dict)

        dim_loss, dim_tb_dict = self.get_dim_loss()
        rpn_loss += dim_loss
        tb_dict.update(dim_tb_dict)

        rot_loss, rot_tb_dict = self.get_rot_loss()
        rpn_loss += rot_loss
        tb_dict.update(rot_tb_dict)

        segm_loss, segm_tb_dict = self.get_segm_loss()
        rpn_loss += segm_loss
        tb_dict.update(segm_tb_dict)

        corner_loss, corner_tb_dict = self.get_corner_loss()
        rpn_loss += corner_loss
        tb_dict.update(corner_tb_dict)

        iouscore_loss, iouscore_tb_dict = self.get_iouscore_loss()
        rpn_loss += iouscore_loss
        tb_dict.update(iouscore_tb_dict)

        tb_dict['rpn_loss'] = rpn_loss.item()
        
        return rpn_loss, tb_dict

    def get_hm_loss(self):
        """
        #([bs, num_cls, 188, 188]) for waymo
        """
        tb_dict = {}
        hm_pred = self.forward_ret_dict['hm_pred'] # (B, c(=num_class), sizey, sizex)
        hm_target = self.forward_ret_dict['hm_target'] # (B, c(=num_class), sizey, sizex)
        hm_loss_weight=self.model_cfg.LOSS_CONFIG.HM_LOSS_CONFIG['weight']

        hm_loss = self.hm_loss_func(hm_pred, hm_target)
        hm_loss = hm_loss_weight * hm_loss

        tb_dict['rpn_hm_loss'] = hm_loss.item()
        return hm_loss, tb_dict

    def get_offset_loss(self):
        tb_dict={}
        offset_pred = self.forward_ret_dict['offset_pred']
        offset_target = self.forward_ret_dict['anno_box_target'][:,:,0:2]
        mask = self.forward_ret_dict['mask_target']
        ind = self.forward_ret_dict['ind_target']
        offset_loss = self.offset_loss_func(output=offset_pred,
                                            mask=mask,
                                            ind=ind,
                                            target=offset_target)

        offset_loss_weight=self.model_cfg.LOSS_CONFIG.OFFSET_LOSS_CONFIG['weight']
        offset_loss = offset_loss_weight * offset_loss

        tb_dict['rpn_offset_loss'] = offset_loss.item()
        return offset_loss, tb_dict

    def get_height_loss(self):
        tb_dict={}
        height_pred = self.forward_ret_dict['height_pred']
        height_target = self.forward_ret_dict['anno_box_target'][:,:,2].unsqueeze(dim=-1)
        mask = self.forward_ret_dict['mask_target']
        ind = self.forward_ret_dict['ind_target']
        height_loss = self.height_loss_func(output=height_pred,
                                            mask=mask,
                                            ind=ind,
                                            target=height_target)

        height_loss_weight=self.model_cfg.LOSS_CONFIG.HEIGHT_LOSS_CONFIG['weight']
        height_loss = height_loss_weight * height_loss

        tb_dict['rpn_height_loss'] = height_loss.item()
        return height_loss, tb_dict

    def get_dim_loss(self):
        tb_dict={}
        dim_pred = self.forward_ret_dict['dim_pred']
        dim_target = self.forward_ret_dict['anno_box_target'][:,:,3:6]
        mask = self.forward_ret_dict['mask_target']
        ind = self.forward_ret_dict['ind_target']
        dim_loss = self.dim_loss_func(output=dim_pred,
                                      mask=mask,
                                      ind=ind,
                                      target=dim_target)

        dim_loss_weight=self.model_cfg.LOSS_CONFIG.DIM_LOSS_CONFIG['weight']
        dim_loss = dim_loss_weight * dim_loss

        tb_dict['rpn_dim_loss'] = dim_loss.item()
        return dim_loss, tb_dict

    def get_rot_loss(self):
        tb_dict={}
        rot_pred =  self.forward_ret_dict['rot_pred']
        rot_target = self.forward_ret_dict['anno_box_target'][:,:,6].unsqueeze(dim=-1)
        mask = self.forward_ret_dict['mask_target']
        ind = self.forward_ret_dict['ind_target']
        rot_loss = self.rot_loss_func(output=rot_pred,
                                    mask=mask,
                                    ind=ind,
                                    target=rot_target)

        rot_loss_weight=self.model_cfg.LOSS_CONFIG.ROT_LOSS_CONFIG['weight']
        rot_loss = rot_loss_weight * rot_loss

        tb_dict['rpn_rot_loss'] = rot_loss.item()
        return rot_loss, tb_dict

    def get_segm_loss(self):
        tb_dict={}
        segm_pred = self.forward_ret_dict['segm_pred'] # (B, C(=1), sizey, sizex)
        segm_target = self.forward_ret_dict['segm_target'] # (B, 1, sizey, sizex)
        
        batch_size, nchs, sizey, sizex = segm_pred.shape

        # flatten as (B, N, C=1)
        segm_pred_flat = segm_pred.permute(0,2,3,1).contiguous().view(batch_size, sizey*sizex, nchs)
        segm_target_flat = segm_target.permute(0,2,3,1).contiguous().view(batch_size, sizey*sizex, nchs)

        positives = segm_target_flat > 0
        negatives = segm_target_flat == 0

        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_loss_src = self.segm_loss_func(segm_pred_flat, segm_target_flat, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        segm_loss_weight=self.model_cfg.LOSS_CONFIG.SEGM_LOSS_CONFIG['weight']
        segm_loss = segm_loss_weight * cls_loss
        tb_dict = {
            'rpn_segm_loss': segm_loss.item()
        }

        return segm_loss, tb_dict

    def get_corner_loss(self):
        """
        Args:
        Returns:
        """
        tb_dict = {}

        src_box_pred = self.forward_ret_dict['gthm_box_preds'] # (B, max_objs, 7)
        src_box_target = self.forward_ret_dict['src_box_target'] # (B, max_objs, 7)
        mask_target = self.forward_ret_dict['mask_target'] # (B, max_objs)
        # flatten
        src_box_pred_flat = src_box_pred.view(-1, src_box_pred.shape[-1]) # (B*max_objs, 7)
        src_box_target_flat = src_box_target.view(-1, src_box_target.shape[-1]) # (B*max_objs, 7)
        mask_target_flat = mask_target.view(-1).bool() # (B*max_objs)

        corner_loss = loss_utils.get_corner_loss_mse(pred_bbox3d=src_box_pred_flat[mask_target_flat],
                                                    gt_bbox3d=src_box_target_flat[mask_target_flat])
        corner_loss_weight = self.model_cfg.LOSS_CONFIG.CORNER_LOSS_CONFIG['weight']
        corner_loss = corner_loss_weight * corner_loss / 3.0

        tb_dict['rpn_corner_loss'] = corner_loss.item()
        return corner_loss, tb_dict

    def get_iouscore_loss(self):
        """
        bug-fixed version.
        """
        tb_dict = {}
        iouscore_pred = self.forward_ret_dict['batch_iouscore_preds'] # (B, max_objs, 1) 
        src_box_pred = self.forward_ret_dict['batch_box_preds'].detach() # (B, max_objs, 7)
        _ , cls_ind_pred = torch.max(self.forward_ret_dict['batch_cls_preds'], dim=-1)
        cls_label_pred = cls_ind_pred + 1 # (B, max_objs, )

        src_box_target = self.forward_ret_dict['batch_gtboxes_src'][..., 0:7] # (B, M, 7)
        cls_label_target = self.forward_ret_dict['batch_gtboxes_src'][..., 7].long() # (B, M, )
        batch_size = src_box_pred.shape[0]

        # for each batch
        roi_iou3d_list = []
        for i in range(batch_size):
            src_box_pred_single = src_box_pred[i, ...] # (max_objs, 7)
            src_cls_pred_single = cls_label_pred[i, ...]
            src_box_target_single = src_box_target[i, ...] # (M, 7)
            src_cls_target_single = cls_label_target[i, ...]

            valid = ( torch.sum(src_box_target_single, dim=1) != 0 ) # (M) 
            src_box_target_single = src_box_target_single[valid]
            src_cls_target_single = src_cls_target_single[valid]

            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=src_box_pred_single, roi_labels=src_cls_pred_single, 
                gt_boxes=src_box_target_single, gt_labels=src_cls_target_single)
            roi_iou3d_list.append( max_overlaps.view(-1, 1) )
        
        # stack for batch axis, batch (max_objs, 1) -> (batch, max_objs, 1)
        roi_iou3d = torch.stack(roi_iou3d_list, dim=0) 

        # flatten & loss, (batch, max_objs, 1) -> (batch*max_objs, )
        fg_thresh=self.model_cfg.LOSS_CONFIG.IOUSCORE_LOSS_CONFIG['iou_fg_thresh']
        bg_thresh=self.model_cfg.LOSS_CONFIG.IOUSCORE_LOSS_CONFIG['iou_bg_thresh']

        iouscore_loss = loss_utils.get_iouscore_loss_bce(iou_preds=iouscore_pred.view(-1),
                                                         iou_gts=roi_iou3d.view(-1),
                                                         iou_fg_thresh=fg_thresh, 
                                                         iou_bg_thresh=bg_thresh)
                                                         
        iouscore_loss_weight = self.model_cfg.LOSS_CONFIG.IOUSCORE_LOSS_CONFIG['weight']
        iouscore_loss = iouscore_loss_weight * iouscore_loss

        tb_dict['rpn_iouscore_loss'] = iouscore_loss.item()
        
        roi_iou3d_flat = roi_iou3d.view(-1)

        fg_mask = (roi_iou3d_flat > fg_thresh)
        bg_mask = (roi_iou3d_flat < bg_thresh)
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        # add to tensorboard
        num_sample_fg = fg_mask.float().sum()
        num_sample_bg = bg_mask.float().sum()
        num_sample_inter = interval_mask.float().sum()
        tb_dict['num_sample_fg'] = num_sample_fg / batch_size
        tb_dict['num_sample_bg'] = num_sample_bg / batch_size
        tb_dict['num_sample_inter'] = num_sample_inter / batch_size

        return iouscore_loss, tb_dict

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (M, )
            gt_labels:(M)

        Returns:

        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                # original_gt_assignment = gt_mask.nonzero().view(-1)
                original_gt_assignment = torch.nonzero( gt_mask, as_tuple=False ).view(-1) 

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment



    def gthm_based_predicted_boxes_generation(self):
        """
        Prepare the data as (B, max_objs, C)
        """
        # hm_pred = self.forward_ret_dict['hm_pred']
        offset_pred = self.forward_ret_dict['offset_pred']
        height_pred = self.forward_ret_dict['height_pred']
        dim_pred = self.forward_ret_dict['dim_pred']
        rot_pred = self.forward_ret_dict['rot_pred']
        ind_target = self.forward_ret_dict['ind_target']
        mask_target = self.forward_ret_dict['mask_target']
        xsys_target = self.forward_ret_dict['xsys_target'] # (B, max_objs, 2)

        # batch, num_class, _, _ = hm_pred.size()
        # assert self.num_class == num_class
        batch, num_objs = ind_target.shape

        # select by ind_target, (B, max_objs, C)
        # offset
        offset = center_utils._transpose_and_gather_feat(offset_pred, ind_target)
        offset = offset.view(batch, num_objs, 2)
        xs = xsys_target[:,:,0:1] + offset[:,:,0:1] # x_ind + x_res
        ys = xsys_target[:,:,1:2] + offset[:,:,1:2] # y_ind + y_res
        
        # height
        height = center_utils._transpose_and_gather_feat(height_pred, ind_target)
        height = height.view(batch, num_objs, 1)

        # dims
        dim = center_utils._transpose_and_gather_feat(dim_pred, ind_target)
        dim = dim.view(batch, num_objs, 3)

        # rot
        num_bins = rot_pred.shape[1] // 2
        rot = center_utils._transpose_and_gather_feat(rot_pred, ind_target) # (B, max_objs, C(=2*num_bins))
        rot = rot.view(batch, num_objs, rot.shape[-1]) 
        rot_to_decode = rot.view(-1, rot.shape[-1])
        rot_decoded = box_utils.decode_rot_binres(rot_to_decode, num_head_bin=num_bins)
        rot = rot_decoded.view(batch, num_objs, 1)

        # scaling for center location 
        xs = xs.view(batch, num_objs, 1) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = ys.view(batch, num_objs, 1) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        # cat together, (B, max_objs, 7)
        final_box_pred = torch.cat([xs, ys, height, dim, rot], dim=2) 
        
        # mask or not? CANNT, num_objs varies in batches.

        rtn_dict = {
            'gthm_box_preds': final_box_pred
        }
        return rtn_dict        
        
    def predhm_based_predicted_boxes_generation_ssd(self, K=50):
        """
        Args:
            hm_pred: (B, C(=num_class), sizey, sizex)
            offset_pred: (B, C(=2), sizey, sizex)
            heigt_pred: (B, C(=1), sizey, sizex)
            dim_pred: (B, C(=3), sizey, sizex)
            rot_pred: (B, C(=24), sizey, sizex)
        Return:
            batch_box_preds: (B, K, 7)
            batch_cls_preds: (B, K, C(=num_class))
            batch_score_preds: (B, K, 1)
        """
        hm_pred = self.forward_ret_dict['hm_pred']
        iouscore_pred = self.forward_ret_dict['iouscore_pred']

        offset_pred = self.forward_ret_dict['offset_pred']
        height_pred = self.forward_ret_dict['height_pred']
        dim_pred = self.forward_ret_dict['dim_pred']
        rot_pred = self.forward_ret_dict['rot_pred']
        
        batch, num_class, _, _ = hm_pred.size()
        assert self.num_class == num_class
        
        # NMS-1: max-pooling based nms
        heat = center_utils._nms(hm_pred)
        
        # NMS-2: The following line can used for normal NMS. 
        # heat = hm_pred

        # select topk ind
        scores, inds, clses, ys, xs = center_utils._topk(heat, K=K) #(B, K)
        
        # offset for xy
        offset = center_utils._transpose_and_gather_feat(offset_pred, inds)
        offset = offset.view(batch, K, 2)
        # add the offset 
        xs = xs.view(batch, K, 1) + offset[:, :, 0:1] # x_ind + x_res
        ys = ys.view(batch, K, 1) + offset[:, :, 1:2] # y_ind + y_res

        # height for z
        height = center_utils._transpose_and_gather_feat(height_pred, inds)
        height = height.view(batch, K, 1)

        # dim 
        dim = center_utils._transpose_and_gather_feat(dim_pred, inds)
        dim = dim.view(batch, K, 3)

        # rot 
        num_bins = rot_pred.shape[1] // 2 
        rot = center_utils._transpose_and_gather_feat(rot_pred, inds) # (B, K, C(=2*num_bins))
        rot = rot.view(batch, K, rot.shape[-1]) 
        rot_to_decode = rot.view(-1, rot.shape[-1])
        rot_decoded = box_utils.decode_rot_binres(rot_to_decode, num_head_bin=num_bins)
        rot = rot_decoded.view(batch, K, 1)

        # hm for classes (batch_cls_preds)
        classes = center_utils._transpose_and_gather_feat(heat, inds)
        final_classes = classes.view(batch, K, self.num_class)

        # score for batch_score_preds
        iouscore = center_utils._transpose_and_gather_feat(iouscore_pred, inds)
        final_scores = iouscore.view(batch, K, 1)


        # scaling for center location
        xs = xs.view(batch, K, 1) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = ys.view(batch, K, 1) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]


        # TODO: center range checking
        final_box_pred = torch.cat([xs, ys, height, dim, rot], dim=2)


        rtn_dict = {
            'batch_box_preds': final_box_pred,
            'batch_cls_preds': final_classes,
            'batch_iouscore_preds': final_scores
        }
        
        return rtn_dict

    def predhm_based_predicted_boxes_generation_nomaxpooling(self, K=50):
        """
        Args:
            hm_pred: (B, C(=num_class), sizey, sizex)
            offset_pred: (B, C(=2), sizey, sizex)
            heigt_pred: (B, C(=1), sizey, sizex)
            dim_pred: (B, C(=3), sizey, sizex)
            rot_pred: (B, C(=24), sizey, sizex)
        Return:
            batch_box_preds: (B, K, 7)
            batch_cls_preds: (B, K, C(=num_class))
            batch_score_preds: (B, K, 1)
        """
        hm_pred = self.forward_ret_dict['hm_pred']
        iouscore_pred = self.forward_ret_dict['iouscore_pred']

        offset_pred = self.forward_ret_dict['offset_pred']
        height_pred = self.forward_ret_dict['height_pred']
        dim_pred = self.forward_ret_dict['dim_pred']
        rot_pred = self.forward_ret_dict['rot_pred']
        
        batch, num_class, _, _ = hm_pred.size()
        assert self.num_class == num_class
        

        # NMS-1: max-pooling based nms
        # heat = center_utils._nms(hm_pred)
        
        # NMS-2: The following line can used for normal NMS. 
        heat = hm_pred


        # select topk ind
        scores, inds, clses, ys, xs = center_utils._topk(heat, K=K) #(B, K)
        
        # offset for xy
        offset = center_utils._transpose_and_gather_feat(offset_pred, inds)
        offset = offset.view(batch, K, 2)
        # add the offset 
        xs = xs.view(batch, K, 1) + offset[:, :, 0:1] # x_ind + x_res
        ys = ys.view(batch, K, 1) + offset[:, :, 1:2] # y_ind + y_res

        # height for z
        height = center_utils._transpose_and_gather_feat(height_pred, inds)
        height = height.view(batch, K, 1)

        # dim 
        dim = center_utils._transpose_and_gather_feat(dim_pred, inds)
        dim = dim.view(batch, K, 3)

        # rot 
        num_bins = rot_pred.shape[1] // 2 
        rot = center_utils._transpose_and_gather_feat(rot_pred, inds) # (B, K, C(=2*num_bins))
        rot = rot.view(batch, K, rot.shape[-1]) 
        rot_to_decode = rot.view(-1, rot.shape[-1])
        rot_decoded = box_utils.decode_rot_binres(rot_to_decode, num_head_bin=num_bins)
        rot = rot_decoded.view(batch, K, 1)

        # hm for classes (batch_cls_preds)
        classes = center_utils._transpose_and_gather_feat(heat, inds)
        final_classes = classes.view(batch, K, self.num_class)

        # score for batch_score_preds
        iouscore = center_utils._transpose_and_gather_feat(iouscore_pred, inds)
        final_scores = iouscore.view(batch, K, 1)


        # scaling for center location
        xs = xs.view(batch, K, 1) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = ys.view(batch, K, 1) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        # TODO: center range checking
        final_box_pred = torch.cat([xs, ys, height, dim, rot], dim=2)

        rtn_dict = {
            'batch_box_preds': final_box_pred,
            'batch_cls_preds': final_classes,
            'batch_score_preds': final_scores
        }
        
        return rtn_dict



    def forward(self, **kwargs):
        raise NotImplementedError

