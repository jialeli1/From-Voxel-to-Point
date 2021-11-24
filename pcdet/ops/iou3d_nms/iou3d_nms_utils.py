"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda

import numpy as np
from shapely.geometry import Polygon
from pcdet.utils.box_utils import boxes_to_corners_3d


def soft_nms_torch(dets, box_scores, iou_thresh=0.1, sigma=0.5, thresh=0.001):
    """
    # 0.001
    # 应该要使用bev_iou 还是 iou3d ？
    reference https://github.com/DocF/Soft-NMS.git
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        3D boxes coordinate tensor
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        iou_thresh   iou_thresh if use method2 else 0
        thresh:      score thresh       
    # Return
        the sorted index of the selected boxes
    """
    # print('==> input dets.shape: ', dets.shape)
    # print('==> input box_scores: ', box_scores)

    N = dets.shape[0]  # the number of boxes

    # Indexes concatenate boxes with the last column
    indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1) 
    dets = torch.cat((dets, indexes), dim=1)
    # print('==> cated dets.shape: ', dets.shape)
    # Sort the scores of the boxes from largest to smallest
    box_scores, conf_sort_index = torch.sort(box_scores, descending=True)
    # 这里顺序都改变了？外面的顺序是否还能保持一致呢？
    # print('==> sorted box_scores: ', box_scores)
    dets = dets[conf_sort_index]

    for i in range(N):
    # for i in range(N-1):

        pos=i+1

        #iou calculate
        # ious = box_iou(dets[i][0:4].view(-1,4), dets[pos:,:4])
        # TODO: 选择3Diou还是beviou？
        ious = boxes_iou_bev(dets[i][0:7].view(-1, 7), dets[pos:, :7])
        # ious = boxes_iou3d_gpu(dets[i][0:7].view(-1, 7), dets[pos:, :7])

        # method1
        # Gaussian decay 
        box_scores[pos:] = torch.exp(-(ious * ious) / sigma) * box_scores[pos:]

        # method2
        # zero = torch.zeros_like(ious)
        # ious  = torch.where(ious < iou_thresh, zero , ious)
        # box_scores[pos:] = torch.exp(-(ious * ious) / sigma) * box_scores[pos:]

        # method3 original nms
        # weight = torch.ones(ious.shape)
        # weight[ious > iou_thresh] = 0
        # box_scores[pos:] = weight * box_scores[pos:]

        # box_scores[pos:] = box_scores[pos:]
        # box_scores[pos:], arg_sort = torch.sort(box_scores[pos:], descending=True)
        # print('==> box_scores[pos:].shape: ', box_scores[pos:].shape)
        box_scores_sorted, arg_sort = torch.sort(box_scores[pos:], descending=True)
        box_scores[pos:] = box_scores_sorted
        # print('==> in loop, sorted box_scores: ', box_scores)

        a=dets[pos:]
        
        dets[pos:] = a[arg_sort]

    # select the boxes and keep the corresponding indexes
    # 这里是因为之前把index cat到了box的最后一维度上去，所以这里应该是选择box的index
    # keep = dets[:,4][box_scores>thresh].long()
    # print('==> final dets.shape: ', dets.shape)
    # print('==> final box_scores.shape: ', box_scores.shape)
    # print('==> final box_scores: ', box_scores)
    valid = box_scores>thresh
    # print('==> valid.shape: ', valid.shape)
    det_indexes = dets[:,7]
    # print('==> det_indexes.shape: ', det_indexes.shape)
    print('==> det_indexes: ', det_indexes)

    # keep = dets[:,7][box_scores>thresh].long()
    keep = (det_indexes[valid]).long()

    # print('==> output keep.shape: ', keep.shape)
    return keep

def soft_nms_torch_1(dets, box_scores, iou_thresh=0.1, sigma=0.009, thresh=0.001, cuda=1):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """
    # print('0. ==> dets.shape: ', dets.shape)
    # print('1. ==> box_scores.shape: ', box_scores.shape)

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)
    # print('2. ==> dets.shape: ', dets.shape)

    scores = box_scores
    for i in range(N):
        # print('==> i: ', i)
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()

        # IoU calculate
        if dets[pos:, 0].shape[0] != 0:
            ovr = boxes_iou_bev(dets[i, :7].view(-1, 7), dets[pos:, :7].view(-1, 7))
            # ovr = boxes_iou3d_gpu(dets[i, :7].view(-1, 7), dets[pos:, :7].view(-1, 7))
            # 这里是计算BEV IoU还是3D IoU
            ovr = ovr.squeeze(dim=0)
            # print('==> ovr.shape: ', ovr.shape)
            # print('==> ovr', ovr)
        else:
            ovr = torch.tensor([]).to(dets.device)
            # print('==> ovr.shape: ', ovr.shape)
            # print('==> ovr', ovr)
        # print('==> ovr: ', ovr)
        # Gaussian decay
        # '''
        weight = torch.exp(-(ovr * ovr) / sigma)
        # print('==> weight.shape: ', weight.shape)
        # print('==> weight: ', weight)
        scores[pos:] = weight * scores[pos:]
        # print('==> scores.shape: ', scores.shape)
        # '''

        # original nms
        '''
        weight = torch.ones(ovr.shape).to(dets.device)
        weight[ovr > iou_thresh] = 0
        scores[pos:] = weight * scores[pos:]
        '''

    # select the boxes and keep the corresponding indexes
    # print('==> final scores.shape: ', scores.shape)
    # keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, -1][scores > thresh].long()
    # print('==> keep: ', keep)
    return keep

def iou_weighted_nms_cpu(
                        boxes, scores, iou_preds, labels, anchors,
                        suppressed_thresh = 0.3, # 0.3 in CIA-SSD OR 0.1?
                        cnt_thresh = 0.5, #2.6, # from CIA-SSD
                        match_thresh = 0.3,
                        nms_sigma_dist_interval=[0, 20, 40, 60],
                        nms_sigma_square = [0.0009, 0.009, 0.1, 1]):
    """
    必须是单张点云内的数据
    thresh: 是suppressed thresh
    boxes: (N, 7), 
    scores: (N, )
    iou_preds: (N, )
    labels: (N, )
    anchors: (N, 7), 与boxes对齐的
    """
    scores_ret = []
    boxes_ret = []
    labels_ret = []

    boxes_r = boxes.clone().cpu().numpy()
    scores_r = scores.clone().cpu().numpy()
    IOU_preds_r = iou_preds.clone().cpu().numpy()
    labels_r = labels.clone().cpu().numpy()
    anchors_r = anchors.clone().cpu().numpy()
    box_corners_r = boxes_to_corners_3d(boxes).clone().cpu().numpy() # (N, 8, 3)
    nms_sigma_dist_interval_r = nms_sigma_dist_interval
    nms_sigma_square_r = nms_sigma_square
    # print('==> boxes_r.shape: ', boxes_r.shape)
    # print('==> scores_r.shape: ', scores_r.shape)
    # print('==> IOU_preds_r.shape: ', IOU_preds_r.shape)
    # print('==> labels_r.shape: ', labels_r.shape)
    # print('==> anchors_r.shape: ', anchors_r.shape)
    # print('==> box_corners_r.shape: ', box_corners_r.shape)


    ndets = boxes.shape[0]
    suppressed_rw = np.zeros((ndets))
    # weight_pos = np.zeros((7))
    # avg_pos = np.zeros((7))

    standup_iou_r = boxes_iou3d_gpu(boxes, boxes).clone().cpu().numpy() # (N, N)

    # 假设用bev_dist(box, anchor)
    # TODO 这里是否需要根据这个来调制呢？
    # dist = torch.pow((boxes[:,0]-anchors[:,0]), 2) + torch.pow( (boxes[:,1]-anchors[:,1]), 2)
    # dist_nrom = (1 - torch.softmax(dist, dim=0)).cpu().numpy()
    # 对score进行调制
    # scores_rw = scores_r * dist_nrom

    scores_rw = scores_r * 1

    # 对调制后的score_rw取最大值，然后归一化
    score_max4norm = np.max(scores_rw)
    scores_rw /= (score_max4norm + 1e-6)

    assert not np.array_equal(scores_rw, scores_r)
    
    while (True):
        score_max = -1
        idx_max = -1
        flag_all_checked = True
        # find out the box with the maximum score
        for i in range(ndets):
            if suppressed_rw[i] == 1:
                continue
            flag_all_checked = False
            # 找到原始分数最大的那一个
            if scores_r[i] > score_max:
                score_max = scores_r[i]
                idx_max = i
        if flag_all_checked:
            break
        # 计算当前box到原点的距离, bev距离上的
        dist2origin = np.linalg.norm(boxes_r[idx_max, 0:2], ord=2)
        suppressed_rw[idx_max] = 1
        
        # 对weight_pos avg_pos重新置零
        weight_pos = np.zeros((1))
        avg_pos = np.zeros((7))
        score_box = -1
        cnt = 0
        recover_list = []
        merge_cnt = 0
        merge_box_list = []
        for j in range(ndets):
            # 当前idx_max的box去和所有的box计算bev_iou
            poly = Polygon(box_corners_r[idx_max, 0:4, 0:2])
            qpoly = Polygon(box_corners_r[j, 0:4, 0:2])
            if poly.intersects(qpoly):
                inter_area = poly.intersection(qpoly).area
                # 计算unions
                union_area = poly.union(qpoly).area
                overlap = inter_area / union_area
                # 计算重叠的数量，用于过滤false_positives
                if (overlap > 0) and (labels_r[j] == labels_r[idx_max]):
                    # TODO: 选择？
                    # cnt += overlap * IOU_preds_r[j]
                    cnt += overlap * scores_r[j]
                # 选择和当前box的iou>0.3的这些辅助box来重新加权
                if (overlap >= match_thresh) and (labels_r[j] == labels_r[idx_max]):
                    if score_box < scores_rw[j]:
                        # 始终保留分数最大的一个
                        # TODO: 这里box是否能够加权呢？
                        score_box = scores_rw[j] 
                    IOU_weight = 0.
                    # 和当前box的iou越大的辅助box的weight应该更大
                    # 根据到原点的距离分段进行
                    for k in range( len(nms_sigma_dist_interval_r) - 1):
                        dist_l = nms_sigma_dist_interval_r[k]
                        dist_r = nms_sigma_dist_interval_r[k+1]
                        # print('==> dist_l: ', dist_l)
                        # print('==> dist_r: ', dist_r)
                        if (dist2origin >= dist_l) and (dist2origin < dist_r):
                            tmp = -np.power( (1-overlap), 2 ) / nms_sigma_square_r[k]
                            IOU_weight = np.exp(tmp)
                            # print('==> dist2origin: ', dist2origin)

                    # 使用IOU_weight对box进行加权
                    # avg_pos += IOU_weight * IOU_preds_r[j] * boxes_r[j]
                    # weight_pos += IOU_weight * IOU_preds_r[j]
                    avg_pos += IOU_weight * scores_r[j] * boxes_r[j]
                    weight_pos += IOU_weight * scores_r[j]
                    assert avg_pos.shape[0] == 7
                    merge_cnt += 1
                    merge_box_list.append(boxes_r[j])

                # suppress the box whose IOU with box[idx_max] > suppressed_thresh
                if (suppressed_rw[j] != 1) and (standup_iou_r[idx_max, j] > 0):
                    if (overlap > suppressed_thresh):
                        suppressed_rw[j] = 1
                        recover_list.append(j)
        
        # print('==> merge_cnt: ', merge_cnt)
        # print('==> merge_box_list: ', merge_box_list)
        # print('==> cnt: ', cnt)
        if (cnt > cnt_thresh):
            scores_ret.append(score_box * score_max4norm)
            avg_pos /= (weight_pos + 1e-6)
            boxes_ret.append(avg_pos)
            labels_ret.append(labels_r[idx_max])
            # print('==> merge valid.')
            if avg_pos[3] + avg_pos[4] + avg_pos[5] == 0:
                scores_ret.pop()
                boxes_ret.pop()
                labels_ret.pop()
                for k in range(len(recover_list)):
                    suppressed_rw[recover_list[k]] = 0
        else:
            # 如果当前这个box是一个false pos，那么就恢复(释放)之前被它抑制掉的其它box
            for k in range(len(recover_list)):
                suppressed_rw[recover_list[k]] = 0

    if len(boxes_ret):
        # not empty:
        ret_boxes_np = np.stack(boxes_ret, axis=0)
        ret_scores_np = np.stack(scores_ret, axis=0)
        ret_labels_np = np.stack(labels_ret, axis=0)
    else:
        ret_boxes_np = np.zeros( [0, 7] )
        ret_scores_np = np.zeros( [0, 1] )
        ret_labels_np = np.zeros( [0, 1] )

    ret_boxes = torch.from_numpy(ret_boxes_np).to(boxes.device)
    ret_scores = torch.from_numpy(ret_scores_np).to(boxes.device)
    ret_labels = torch.from_numpy(ret_labels_np).to(boxes.device)
    # boxes, scores, iou_preds, labels
    '''
    print('==> input boxes: ', boxes)
    print('==> input scores: ', scores)
    print('==> input iou_preds: ', iou_preds)
    print('==> input labels: ', labels)
    print('==> ret_boxes.shape: ', ret_boxes)
    print('==> ret_scores.shape: ', ret_scores)
    print('==> ret_labels.shape: ', ret_labels)
    '''
    return ret_boxes, ret_scores, ret_labels



def matched_boxes_bevdist(boxes_a, boxes_b):
    """
    以bev上的中心点距离作为certainty
    """
    dist = torch.pow((boxes_a[:,0]-boxes_b[:,0]), 2) + torch.pow( (boxes_a[:,1]-boxes_b[:,1]), 2)
    dist_nrom = 1 - torch.softmax(dist, dim=0)

    return dist_nrom.view(-1, 1)


def matched_boxes_iou3d_cpu(boxes_a, boxes_b):
    """
    # 前提是一一对齐的两组box
    Args: 
        boxes_a: tensor, (N, 7) [x,y,z,dx,dy,dz,heading]
        boxes_b: tensor, (N, 7) [x,y,z,dx,dy,dz,heading]
    Returns:
        ans_iou: tensor, (N, 1)
    """
    assert boxes_a.shape == boxes_b.shape

    # 1. 计算高度方向上的overlap
    # height overlap, (N, )
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2)
    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 2. 转为corner, (N, 8, 3)
    corners_a = boxes_to_corners_3d(boxes_a)
    corners_b = boxes_to_corners_3d(boxes_b)

    # 3. 准备numpy格式的polygon 2D points
    np_2dcorners_a = corners_a.clone().cpu().numpy()
    np_2dcorners_b = corners_b.clone().cpu().numpy()
    np_overlaps_bev = np.zeros( (np_2dcorners_a.shape[0]) )
    for i in range(np_2dcorners_a.shape[0]):
        poly_points_a = np_2dcorners_a[i, 0:4, 0:2]
        poly_points_b = np_2dcorners_b[i, 0:4, 0:2]
        poly_a = Polygon(poly_points_a)
        poly_b = Polygon(poly_points_b)
        if poly_a.is_valid and poly_b.is_valid:
            # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
            overlap = poly_a.intersection(poly_b).area
        else:
            overlap = 0.
        np_overlaps_bev[i] = overlap
    # move back to torch
    overlaps_bev = torch.from_numpy(np_overlaps_bev).to(boxes_a.device)

    # 4. 计算3d overlaps
    overlaps_3d = overlaps_bev * overlaps_h
    assert overlaps_3d.shape == overlaps_bev.shape == overlaps_h.shape

    # 5. 计算3d iou
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d.view(-1, 1)
    

def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    # 还要进行限制一下，小于0或者大于1的都设置为0
    iou3d[iou3d < 0] = 0
    iou3d[iou3d > 1] = 1

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def batch_boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (B, N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (B, M, 7) [x, y, z, dx, dy, dz, heading], GT, 可能会补零，所以先去掉0

    Returns:
        ans_iou: (B, N, 1)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[-1] == boxes_b.shape[-1]

    # for each batch
    boxiou_a_list = []
    batch_size = boxes_a.shape[0]
    for i in range(batch_size):
        boxes_a_single = boxes_a[i, ...] # (max_objs, 7)
        boxes_b_single = boxes_b[i, ...] # (M, 7)
        # print('==> 0. src_box_target_single.size(): ', src_box_target_single.size()) # (M, 7)
        # valid = ( torch.sum(boxes_b_single, dim=1) != 0 )
        # print('==> valid.size(): ', valid.size()) # (M) 
        # boxes_b_single = boxes_b_single[valid]
        # print('==> 1. src_box_target_single.size(): ', src_box_target_single.size()) # (Nb(=M))
        iou3d = boxes_iou3d_gpu(boxes_a=boxes_a_single, 
                                boxes_b=boxes_b_single)  # (Na, Nb)
        # print('==> iou3d.size(): ', iou3d.size()) # (max_objs(=Na), Nb)
        max_overlaps, gt_assignment = torch.max(iou3d, dim=1) # roi side iou
        # print('==> max_overlaps.size(): ', max_overlaps.size()) # (max_objs(=Na))
        boxiou_a_list.append( max_overlaps.view(-1, 1) )
    
    # stack for batch axis, batch (max_objs, 1) -> (batch, max_objs, 1)
    boxiou_a = torch.stack(boxiou_a_list, dim=0)

    return  boxiou_a
