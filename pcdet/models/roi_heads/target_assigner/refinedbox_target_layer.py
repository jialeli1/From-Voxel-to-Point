import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class RefinedBoxTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_size = batch_dict['batch_size']
        distribution_dict = {}

        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()
        num_sample_regvalid = reg_valid_mask.float().sum()
        distribution_dict['num_sample_regvalid'] = num_sample_regvalid / batch_size # 新增加除以batch_size

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1

            # 记录一下样本分布, norm 到每个batch上去
            num_sample_fg = batch_cls_labels.float().sum()
            num_sample_inter = ignore_mask.float().sum()
            num_sample_bg = batch_cls_labels.shape[0] - num_sample_inter
            distribution_dict['num_sample_fg'] = num_sample_fg / batch_size
            distribution_dict['num_sample_bg'] = num_sample_bg / batch_size
            distribution_dict['num_sample_inter'] = num_sample_inter / batch_size

        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            # 这里的目的是把中间的iou归一化到两个端点(1/0)上，使得数值在0-1上连续
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
            
            # 记录一下样本分布
            num_sample_fg = fg_mask.float().sum()
            num_sample_bg = bg_mask.float().sum()
            num_sample_inter = interval_mask.float().sum()
            distribution_dict['num_sample_fg'] = num_sample_fg / batch_size
            distribution_dict['num_sample_bg'] = num_sample_bg / batch_size
            distribution_dict['num_sample_inter'] = num_sample_inter / batch_size

        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'rcnn_iou':
            # 这里先不进行label制作，后面根据预测的rcnn_reg来制作，并更新它.
            batch_cls_labels = None
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'distribution_dict': distribution_dict}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        # 实际上这里没有进行sample，只是按类别进行iou计算
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        _, ROI_PER_IMAGE, code_size = rois.shape
        assert _ == batch_size

        batch_rois = rois.new_zeros(batch_size, ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            # cur_roi, cur_gt, cur_roi_labels = rois[index], gt_boxes[index], roi_labels[index]

            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            # 这里不再进行box的sample
            batch_rois[index] = cur_roi
            batch_roi_labels[index] = cur_roi_labels
            batch_roi_ious[index] = max_overlaps
            batch_roi_scores[index] = cur_roi_scores
            batch_gt_of_rois[index] = cur_gt[gt_assignment]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels


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
                # original_gt_assignment = gt_mask.nonzero().view(-1)
                # print('==> 1. original_gt_assignment.shape: ', original_gt_assignment.shape)
                # print(original_gt_assignment)
                original_gt_assignment = torch.nonzero( gt_mask, as_tuple=False ).view(-1) # for torch 1.5+
                # 但是他们实际上也有同样的输出
                # print('==> 2. original_gt_assignment.shape: ', original_gt_assignment.shape)
                # print(original_gt_assignment)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
