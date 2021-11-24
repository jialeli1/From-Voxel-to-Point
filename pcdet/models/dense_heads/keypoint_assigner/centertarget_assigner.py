import numpy as np
import torch
import math
import cv2

from ....utils import box_utils
from ....utils.center_utils import draw_umich_gaussian, gaussian_radius, draw_seg_mask

class CenterTargetAssigner(object):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, class_names):
        super().__init__()

        target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.class_names = np.array(class_names)
        self.gaussian_minoverlap = target_cfg.GAUSSIAN_MINOVERLAP
        self.gaussian_minradius = target_cfg.GAUSSIAN_MINRADIUS
        self.feature_map_stride = target_cfg.FEATURE_MAP_STRIDE
        self.max_objs = target_cfg.MAX_OBJS

        self.point_cloud_range = point_cloud_range
        self.voxel_size = np.array(voxel_size)


    def assign_targets(self, gt_boxes_with_classes):
        """
        Args: 
            gt_boxes_with_classes: (B, M, 8) [x,y,z,dimx(l),dimy(w),dimz(h),rot,cls]

        Return:
            hm_target: (B, n_dim(=num_class), mapsizey, mapsizex) 
            anno_box_target: (B, max_objs, 7)
            ind_target: (B, max_objs, )
            mask: (B, max_objs, )
            batch_gtboxes_src: (B, M, 8)
        """

        batch_gtboxes_src = gt_boxes_with_classes.clone()
        target_device = gt_boxes_with_classes.device

        # move to cpu
        gt_boxes_with_classes_np = gt_boxes_with_classes.cpu().numpy()

        batch_size = gt_boxes_with_classes_np.shape[0]
        gt_classes = gt_boxes_with_classes_np[:, :, -1]
        gt_boxes = gt_boxes_with_classes_np[:, :, :-1]

        target_list = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            # cur_gt_classes = gt_classes[k][:cnt + 1].int()
            cur_gt_classes = (gt_classes[k][:cnt + 1]).astype(np.int8)

            single_target = self.assign_target_maps_single(
                                gt_boxes=cur_gt, 
                                gt_classes=cur_gt_classes, 
                                num_classes=self.class_names.shape[0], 
                                max_objs=self.max_objs,  
                                gaussian_minoverlap=self.gaussian_minoverlap, 
                                gaussian_minradius=self.gaussian_minradius, 
                                point_cloud_range=self.point_cloud_range, 
                                feature_map_stride=self.feature_map_stride, 
                                voxel_size=self.voxel_size
            )
            target_list.append(single_target)

        # stack to batch format
        target_dict = { 
            'hm_target': torch.from_numpy( np.stack( [t['hm'] for t in target_list], axis=0) ).to(target_device),
            'anno_box_target': torch.from_numpy( np.stack( [t['anno_box'] for t in target_list], axis=0)  ).to(target_device),
            'ind_target': torch.from_numpy( np.stack( [t['ind'] for t in target_list], axis=0)  ).to(target_device),
            'mask_target': torch.from_numpy( np.stack( [t['mask'] for t in target_list], axis=0)  ).to(target_device),
            'segm_target': torch.from_numpy( np.stack( [t['segm'] for t in target_list], axis=0)  ).to(target_device),
            'height_target': torch.from_numpy( np.stack( [t['height'] for t in target_list], axis=0)  ).to(target_device),
            'src_box_target': torch.from_numpy( np.stack( [t['src_box'] for t in target_list], axis=0)  ).to(target_device),
            'xsys_target': torch.from_numpy( np.stack( [t['xsys'] for t in target_list], axis=0)  ).to(target_device),
            'batch_gtboxes_src': batch_gtboxes_src,
        }

        # move to gpu
        # target_dict['hm_target'] = torch.from_numpy(target_dict['hm_target'], device=target_device)
        # target_dict['anno_box_target'] = torch.from_numpy(target_dict['anno_box_target'], device=target_device)
        # target_dict['ind_target'] = torch.from_numpy(target_dict['ind_target'], device=target_device)
        # target_dict['mask_target'] = torch.from_numpy(target_dict['mask_target'], device=target_device)
        

        return target_dict


    def assign_target_maps_single(self, gt_boxes, gt_classes, 
                                num_classes, max_objs,  
                                gaussian_minoverlap, gaussian_minradius, 
                                point_cloud_range, feature_map_stride, voxel_size):
        '''
        Args:
            point_cloud_range: [  0.  -40.   -3.   70.4  40.    1. ], dtype是dtype=np.float32!!!
            mapsize: (200, 176) for kitti
        Return:
        '''
        # print('==> point_cloud_range: ', point_cloud_range)
        # print('==> voxel_size: ', voxel_size)

        feature_map_sizey = np.round( ((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1] / feature_map_stride) ).astype(np.int64) # size_y(img_h), should be 200
        feature_map_sizex = np.round( ((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0] / feature_map_stride) ).astype(np.int64) # size_x(img_w), should be 176

        # print('==> feature_map_sizey: ', feature_map_sizey)
        # print('==> feature_map_sizex: ', feature_map_sizex)
        hm = np.zeros((num_classes, feature_map_sizey, feature_map_sizex), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        mask = np.zeros((max_objs), dtype=np.uint8)
        anno_box = np.zeros((max_objs, 7), dtype=np.float32)
        segm = np.zeros((1, feature_map_sizey, feature_map_sizex), dtype=np.float32)
        height = np.zeros((1, feature_map_sizey, feature_map_sizex), dtype=np.float32)
        xsys = np.zeros((max_objs, 2), dtype=np.float32)
        src_box = np.zeros((max_objs, 7), dtype=np.float32)

        # boxes3d o corners, (N, 7) -> (N, 8, 3)
        gt_corners = box_utils.boxes_to_corners_3d(gt_boxes)

        num_objs = min(gt_boxes.shape[0], max_objs)
        for k in range(num_objs):
            cls_id = gt_classes[k] - 1 
            dimx, dimy, dimz = gt_boxes[k, 3], gt_boxes[k, 4], gt_boxes[k, 5]
            dimx, dimy = dimx / voxel_size[0] / feature_map_stride, dimy / voxel_size[1] / feature_map_stride
            fg_pixel_map = np.zeros((feature_map_sizey, feature_map_sizex), dtype=np.float32) # 每次清零
            if dimx>0 and dimy>0:
                radius = gaussian_radius((math.ceil(dimx), math.ceil(dimy)), min_overlap=gaussian_minoverlap)
                # print('==> min_overlap', gaussian_minoverlap)
                # print('==> float radius', radius)
                radius = max(int(radius), gaussian_minradius)
                # print('==> final radius')
                # print(radius)
                x, y, z = gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2]
                coor_x, coor_y = (x - point_cloud_range[0]) / voxel_size[0] / feature_map_stride, \
                                 (y - point_cloud_range[1]) / voxel_size[1] / feature_map_stride
                

                # for draw_umich_gaussian,
                # NOTE: CT IS [X,Y] OR [Y,X], [X,Y]
                # ct = np.array([coor_x, coor_y], dtype=np.float32)
                ct = np.array([coor_x, coor_y])
                # floor
                # ct_int = ct.astype(np.int32)

                # closet location
                # ct_int = (np.around(ct)).astype(np.int32)
                ct_int = (np.around(ct)).astype(np.int64)

                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < feature_map_sizex and 0 <= ct_int[1] < feature_map_sizey):
                    # print('==> ct_int[0]: %d, feature_map_sizex: %d, ct_int[1]: %d, feature_map_sizey: %d' \
                        # %(ct_int[0], feature_map_sizex, ct_int[1], feature_map_sizey))
                    continue
                
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                x_int, y_int = ct_int[0], ct_int[1]

                if not (y_int * feature_map_sizex + x_int < feature_map_sizey * feature_map_sizex):
                    # a double check, should never happen
                    print(y_int, x_int, y_int * feature_map_sizex + x_int)
                    assert False 
                    
                ind[k] = feature_map_sizex * y_int + x_int
                mask[k] = 1
                anno_box[k, :2] = np.array((ct[0] - x_int, ct[1] - y_int), dtype=np.float32)
                
                '''
                # floor() int
                if not (0 <= anno_box[k, 0] < 1 or 0 <= anno_box[k, 1] < 1 ):
                    print('==> ct[0]: , x_int: , ct[1]: , y_int: ')
                    print((ct[0], x_int, ct[1], y_int))
                    assert False
                '''
                # '''
                # around() int
                if not (-0.5 <= anno_box[k, 0] <= 0.5 or -0.5 <= anno_box[k, 1] <= 0.5 ):
                    print('==> ct[0]: , x_int: , ct[1]: , y_int: ')
                    print((ct[0], x_int, ct[1], y_int))
                    assert False
                # '''
                anno_box[k, 2:7] = gt_boxes[k, 2:7]

                xsys[k, :] = ct_int
                src_box[k, :] = gt_boxes[k, :]

                # 4 corners with xy
                cornersbev = gt_corners[k, 0:4, 0:2] # (4, 2)
                cornersbev[:,0] = np.clip(cornersbev[:,0], a_min=point_cloud_range[0], a_max=point_cloud_range[3]) #(4,)
                cornersbev[:,1] = np.clip(cornersbev[:,1], a_min=point_cloud_range[1], a_max=point_cloud_range[4]) #(4,)

                corner_coor_x = (cornersbev[:,0] - point_cloud_range[0]) / voxel_size[0] / feature_map_stride
                corner_coor_y = (cornersbev[:,1] - point_cloud_range[1]) / voxel_size[1] / feature_map_stride
                corner_coor = np.stack([corner_coor_x, corner_coor_y], axis=1) #(4,2)
                corner_coor_int = np.around(corner_coor).astype(np.int32) # (4, 2)
                # corner_coor_int = corner_coor.astype(np.int32) # (4, 2)
                draw_seg_mask(segm[0], corner_coor_int) # must be int type

                # fill box height
                draw_seg_mask(fg_pixel_map, corner_coor_int) 
                fg_pixel_ind = fg_pixel_map > 0
                height[0][fg_pixel_ind] = gt_boxes[k, 2]


        ret_dict = {
            'hm': hm,
            'anno_box': anno_box,
            'ind': ind,
            'mask': mask,
            'segm': segm,
            'height': height,
            'src_box': src_box,
            'xsys': xsys
        }

        return ret_dict


       