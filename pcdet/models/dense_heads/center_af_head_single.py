import torch
import torch.nn as nn

from .center_af_head_template import CenterAFHeadTemplate
from .feature_adaptor.deformable_convs import FeatureAdaption as FeatureAdaptionV1
from .feature_adaptor.mdeformable_convs import FeatureAdaption as FeatureAdaptionV2

class CenterAFHeadSingle(CenterAFHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, voxel_size, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, voxel_size=voxel_size, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.predict_boxes_when_training = True
        self.iouscore_training_samples = self.model_cfg.NUM_IOUSCORE_TRAINING_SAMPLES
        self.num_infernce_samples = self.model_cfg.NUM_INFERENCE_SAMPLES

        # shared conv
        pre_channel = input_channels
        shared_conv_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            # hiddens
            shared_conv_list.extend([
                nn.Conv2d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]
            # dropout
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_conv_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_conv_layer = nn.Sequential(*shared_conv_list)
        input_channels = pre_channel

        # adaptation with deformable convs
        if self.model_cfg.USE_DCN == 'DCN':
            self.feature_adapt = FeatureAdaptionV1(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                deformable_groups=4)
        elif self.model_cfg.USE_DCN == 'MDCN':
            self.feature_adapt = FeatureAdaptionV2(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                deformable_groups=4)
        
        self.num_spatial_features_before_head = input_channels


        # heads
        self.head_names = [ cfg['name'] for cfg in self.model_cfg.HEADS_CONFIG ]
        for head_cfg in self.model_cfg.HEADS_CONFIG:
            if head_cfg['name'] == 'hm': 
                head_cfg['out_channel'] = self.num_class
            cur_head = self.make_fc_head(input_channels=input_channels,
                                        head_cfg=head_cfg)
            self.__setattr__(head_cfg['name'], cur_head)


    def forward(self, data_dict):
        """
        'points', 'frame_id', 'calib', 'gt_boxes', 'road_plane', 'use_lead_xyz', 
        'voxels', 'voxel_coords', 'voxel_num_points', 
        'image_shape', 'batch_size', 
        'voxel_features', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 
        'multi_scale_3d_features', 'spatial_features', 'spatial_features_stride', 
        'spatial_features_2d'
        """
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv_layer(spatial_features_2d)

        # -------------Second Half of the ADFA Module--------------------
        # The second half of the ADFA module follows DCNBEVBackbone, including the final deformable conv layer and mask-guided attention enhancement.
        # For convenience, the segmentation sub-network in Figure 4 is implemented as a head parallel to the detection head, with the same operation according to the paper.
        if self.model_cfg.USE_DCN in ['DCN', 'MDCN']:
            spatial_features_2d = self.feature_adapt(spatial_features_2d)

        if 'segm' in self.head_names:
            # segm_pred
            segm_pred = self.__getattr__('segm')(spatial_features_2d) # (B, C, sizey, sizex)
            segm_pred_norm = torch.sigmoid(segm_pred.detach()) # (B, 1, sizey, sizex) 
            
            # res from adding
            spatial_weight = segm_pred_norm.expand_as(spatial_features_2d) # (B, C, sizey, sizex)
            spatial_features_2d_res = spatial_weight*spatial_features_2d
            spatial_features_2d_att = spatial_features_2d + spatial_features_2d_res 
            segm_preds_name = 'segm' + '_pred'
            self.forward_ret_dict.update({segm_preds_name: segm_pred})
            data_dict.update({'spatial_features_before_head': spatial_features_2d_att})


        # -------------Anchor-free Detection Head--------------------
        for head_name in self.head_names:
            if head_name != 'segm':
                cur_preds_name = head_name + '_pred'
                cur_preds = self.__getattr__(head_name)(spatial_features_2d_att) # (B, C, sizey, sizex)
                self.forward_ret_dict.update({cur_preds_name: cur_preds})


        # -------------Target Assignment--------------------
        if self.training:
            '''
            target_dict includs:
            heatmaps
            heightmaps
            '''
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            

        # -------------Decode predicted boxes for loss computation--------------------
        if self.training and self.predict_boxes_when_training:
            # for iouscore loss computation 
            # decode predicted boxes in an inference manner
            self.forward_ret_dict.update( self.predhm_based_predicted_boxes_generation_ssd(K=self.iouscore_training_samples) )
            
            # for corner loss computation
            # decode predicted boxes in a training manner
            self.forward_ret_dict.update( self.gthm_based_predicted_boxes_generation() )



        # -------------Decode detections for inference--------------------
        if not self.training:
            # You can compare a variety of NMS by setting different flags, which have slight differences.

            # NMS-1: max-pooling based nms (default)
            normal_infer_flag = True
            if normal_infer_flag:
                center3d_pred_dict = self.predhm_based_predicted_boxes_generation_ssd(K=self.num_infernce_samples)
                data_dict.update(center3d_pred_dict)
                data_dict['cls_preds_normalized'] = False
        
            # NMS-2: normal NMS. 
            normal_nms_flag = False
            # normal_nms_flag = True
            if normal_nms_flag:
                center3d_pred_dict = self.predhm_based_predicted_boxes_generation_nomaxpooling(K=100)
                data_dict.update(center3d_pred_dict)
                data_dict['cls_preds_normalized'] = False

        return data_dict