import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import box_utils
from . import center_utils

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse



class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    # 这里flip的目的应该是忽略朝向，但实际上呢把朝向也纳入整体更好还是说它会造成不稳定呢？
    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)




def get_corner_loss_mse(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (1,) float scaler
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    # (N, 8, 3)
    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)
    # print('==> pred_box_corners[0, :, :]')
    # print(pred_box_corners[0,:,:])
    # print('==> gt_box_corners[0, :, :]')
    # print(gt_box_corners[0,:,:])
    # print('==> pred_box_corners[10, :, :]')
    # print(pred_box_corners[10,:,:])
    # print('==> gt_box_corners[10, :, :]')
    # print(gt_box_corners[10,:,:])
    # print('==> pred_box_corners[100, :, :]')
    # print(pred_box_corners[100,:,:])
    # print('==> gt_box_corners[100, :, :]')
    # print(gt_box_corners[100,:,:])

    # for each box, mean by 8 corners.
    corner_loss_x = F.mse_loss(input=pred_box_corners[:,:,0], target=gt_box_corners[:,:,0]) # (N, 8) -> (N)
    corner_loss_y = F.mse_loss(input=pred_box_corners[:,:,1], target=gt_box_corners[:,:,1]) # (N, 8) -> (N)
    corner_loss_z = F.mse_loss(input=pred_box_corners[:,:,2], target=gt_box_corners[:,:,2]) # (N, 8) -> (N)

    # xyz之间求和
    corner_loss = corner_loss_x + corner_loss_y + corner_loss_z

    return corner_loss    


def get_iouscore_loss_bce(iou_preds, iou_gts, iou_fg_thresh=0.75, iou_bg_thresh=0.25):
    """
    Args:
        iou_preds: (N,)
        iou_gts: (N, )
    Returns:
        loss_iouscore:
    """
    # prepare the labels
    # now only for car class, 08132020

    # iou_preds = iou_preds.view(-1)
    # iou_gts = iou_gts.view(-1)

    # print('==> iou_preds.size()')
    # print(iou_preds.size())
    # print(torch.sigmoid(iou_preds))
    # print('==> iou_gts.size()')
    # print(iou_gts.size())
    # print(iou_gts)

    # CLS_FG_THRESH: 0.75
    # CLS_BG_THRESH: 0.25
    # iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
    # iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
    # iou_bg_thresh = 0.25
    # iou_fg_thresh = 0.75

    fg_mask = iou_gts > iou_fg_thresh
    bg_mask = iou_gts < iou_bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)
    
    iou_cls_labels = (fg_mask > 0).float()
    iou_cls_labels[interval_mask] = \
        (iou_gts[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

    # print('==> iou_cls_labels')
    # print(iou_cls_labels.size())
    # print(iou_cls_labels[:50])
    
    # 这里CE是计算的整个范围的iou，但是最后求和的时候只计算了iou>=0这部分的。
    # 条件 iou_cls_labels >= 0 选出来了那些iou >= 0 的候选框。
    loss_ioucls = F.binary_cross_entropy(torch.sigmoid(iou_preds), iou_cls_labels.float(), reduction='none')
    cls_valid_mask = (iou_cls_labels >= 0).float()
    loss_iouscore = (loss_ioucls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

    return loss_iouscore



def get_rot_binres_loss(pred_reg, reg_label, num_head_bin, get_ry_fine=False):
    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
    
    :param pred_reg: (N, C)
    :param reg_label: (N, 1), ry
    :param num_head_bin: constant
    :param get_ry_fine: False
    :return:
    """
    # print('==> pred_reg.size()')
    # print(pred_reg.size()) # should be (N, 24)

    reg_loss_dict = {}
    # angle loss
    start_offset = 0
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    start_offset = ry_res_r
    ry_label = reg_label.squeeze(dim=-1)
    # print('==> reg_label[] in encode')
    # print(reg_label.size()) # should be (N, C)
    # print(reg_label[100:150])
    # print('==> ry_label[] in encode')
    # print(ry_label.size()) # should be (N,)
    # print(ry_label[100:150])
    if get_ry_fine:
        assert False, "one-stage should not get_ry_fine."

        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi
        # print('==> heading_angle[] in encode')
        # print(heading_angle.size())
        # print(heading_angle[100:150])

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)
        # print('==> ry_bin_label in encode')
        # print(ry_bin_label.size())
        # print(ry_bin_label[100:150])


    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res
    # Total regression loss
    reg_loss_dict['loss_angle'] = angle_loss

    return angle_loss, reg_loss_dict



class CenterNetFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, gamma=4, alpha=2):
        super(CenterNetFocalLoss, self).__init__()
        # self.neg_loss = _neg_loss
        self.gamma = gamma
        self.alpha = alpha

    def _sigmoid(self, x):
        # y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        # dnnt use the replace version!
        y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)

        # too small will cause loss nan.
        # y = torch.clamp(x.sigmoid_(), min=1e-12, max=1 - 1e-12)
        return y

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred: (batch x c x h x w), do some clamp or not?. should be clampped already.
            gt: (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # neg_weights = torch.pow(1 - gt, 4)
        neg_weights = torch.pow(1 - gt, self.gamma)

        loss = 0

        # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


    def forward(self, out, target):
        out_norm = self._sigmoid(out)

        return self._neg_loss(out_norm, target)


class CenterNetResLoss(nn.Module):
    def __init__(self, cfg):
        super(CenterNetResLoss, self).__init__()
        self.res_func_type = cfg['res_func']

    def forward(self, output, mask, ind, target):
        """
        Args:
            output: torch.Size([B, C, 152, 152])
            mask: torch.Size([B, max_objs])
            ind: torch.Size([B, max_objs])
            target: torch.Size([B, max_objs, C])
        Returns:
            reduced and weighted loss term.
        """
        pred = center_utils._transpose_and_gather_feat(output, ind)  # (B, max_objs, C)

        # print('==> (ind != 0).float().sum(): ', (ind != 0).float().sum() )
        # print('==> mask.sum(): ', mask.sum() )

        if mask.sum():
            # 1. flatten.
            pred_flat = pred.view(-1, pred.shape[-1]) #(B*max_objs, C)
            target_flat = target.view(-1, target.shape[-1]) #(B*max_objs, C)
            mask_flat = mask.view(-1).bool() #(B*max_objs)
            # 2. valid select
            pred_valid = pred_flat[mask_flat] #(num_valid, C)
            target_valid = target_flat[mask_flat] #(num_valid, C)
            # 3. un-reduced loss term
            if self.res_func_type == 'smooth-l1':
                loss = F.smooth_l1_loss(pred_valid, target_valid, reduction='none')
            elif self.res_func_type == 'l1':
                loss = F.l1_loss(pred_valid, target_valid, reduction='none') 
            elif self.res_func_type == 'balanced_l1':
                loss = get_balanced_l1_loss(pred_valid, target_valid)
            else:
                raise NotImplementedError                                        

            # mean for num_obj_dims, sum for channel_dims
            # (num_valid, C) -> (C) -> ()
            loss = loss.mean(dim=0).sum() 
        else:
            loss = 0.

        return loss

class CenterNetRotBinResLoss(nn.Module):
    def __init__(self, cfg):
        super(CenterNetRotBinResLoss, self).__init__()

        self.num_head_bin = cfg['num_bins']

    def forward(self, output, mask, ind, target):
        """
        Args:
            output: torch.Size([B, C, 152, 152])
            mask: torch.Size([B, max_objs])
            ind: torch.Size([B, max_objs])
            target: torch.Size([B, max_objs, C])
        Returns:
            reduced and weighted loss term.
        """
        pred = center_utils._transpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])

        if mask.sum():
            # 1. flatten
            pred_flat = pred.view(-1, pred.shape[-1]) # (B*max_objs, C)
            target_flat = target.view(-1, target.shape[-1]) # (B*max_objs, 1)
            mask_flat = mask.view(-1).bool() # (B*max_objs)
            # 2. valid select
            pred_valid = pred_flat[mask_flat] # (num_valid, C)
            target_valid = target_flat[mask_flat] # (num_valid, 1)

            # 3. return the reduced rot loss term.
            loss, _ = get_rot_binres_loss(pred_valid, target_valid, num_head_bin=self.num_head_bin)
            
        else:
            loss = 0.

        # print('==> loss in rot')
        # print(loss)
        return loss




def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      NOTE probas should be applied with softmax.
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    # print('==> lovasz_softmax, classes: ', classes)
    # print('==> lovasz_softmax, per_image: ', per_image)
    # print('==> lovasz_softmax, ignore: ', ignore)

    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss



def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 2: 
        # do nothing, 3D segmentation for sparse tensor
        pass
    elif probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    elif probas.dim() == 5:
        # 3D segmentation for dense tensor
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C


    labels = labels.view(-1)
    if ignore is not None:
        valid = (labels != ignore)
        # vprobas = probas[valid.nonzero().squeeze()]
        # for newer pytorch
        vprobas = probas[torch.nonzero(valid, as_tuple=False).squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    else: 
        return probas, labels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

