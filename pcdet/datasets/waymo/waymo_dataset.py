# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.include_waymo_data(self.mode)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file[:-9]) + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        # print('==================> multiprocessing.cpu_count(): ', multiprocessing.cpu_count())
        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))
        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    @staticmethod
    def generate_prediction_dicts_save_to_kitti_format(batch_dict, pred_dicts, class_names, output_path=None, save_to_kitti_format=True):
        """
        generate_prediction_dicts_save_to_kitti_format()
        在保存成kitti格式的时候不改变annos
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        # print('==> batch_dict.keys(): ', batch_dict.keys())
        # ['points', 'frame_id', 'gt_boxes', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'metadata', 'batch_size']

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        if (output_path is not None) and save_to_kitti_format:
            # 为转为KITTI格式准备一些函数
            def get_template_prediction_for_kitti(num_samples):
                ret_dict = {
                    'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                    'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                    'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                    'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                    'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
                }
                return ret_dict

            def generate_single_sample_dict_for_kitti(batch_index, box_dict):
                map_name_to_kitti = {
                    'Vehicle': 'Car',
                    'Pedestrian': 'Pedestrian',
                    'Cyclist': 'Cyclist',
                    'Sign': 'Sign',
                    'Car': 'Car'
                    }
                pred_scores = box_dict['pred_scores'].cpu().numpy()
                pred_boxes = box_dict['pred_boxes'].cpu().numpy()
                pred_labels = box_dict['pred_labels'].cpu().numpy()
                pred_dict = get_template_prediction_for_kitti(pred_scores.shape[0])
                if pred_scores.shape[0] == 0:
                    return pred_dict

                # 对pred_dict进行填充
                pred_dict['bbox'] = np.zeros((len(pred_dict['name']), 4))
                pred_dict['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                pred_dict['truncated'] = np.zeros(len(pred_dict['name']))
                pred_dict['occluded'] = np.zeros(len(pred_dict['name']))
                boxes_lidar = pred_boxes.copy()
                # 预测值不用gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                boxes_lidar[:, 2] -= boxes_lidar[:, 5] / 2
                pred_dict['location'] = np.zeros((boxes_lidar.shape[0], 3))
                pred_dict['location'][:, 0] = -boxes_lidar[:, 1]  # x = -y_lidar
                pred_dict['location'][:, 1] = -boxes_lidar[:, 2]  # y = -z_lidar
                pred_dict['location'][:, 2] = boxes_lidar[:, 0]  # z = x_lidar            
                dxdydz = boxes_lidar[:, 3:6]
                pred_dict['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                pred_dict['rotation_y'] = -boxes_lidar[:, 6] - np.pi / 2.0
                pred_dict['alpha'] = -np.arctan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + pred_dict['rotation_y']

                # 这个class_names是dataset的class_names(str)
                # 这一步是把数字的cls映射为str的cls
                pred_dict['name'] = np.array(class_names)[pred_labels - 1]
                pred_dict['score'] = pred_scores
                pred_dict['boxes_lidar'] = pred_boxes

                # 在dataset的class_names(str)之间进行转换， waymo->kitti.
                for k in range(pred_dict['name'].shape[0]):
                    pred_dict['name'][k] = map_name_to_kitti[pred_dict['name'][k]]

                return pred_dict

            # 下面开始转为KITTI格式
            for index, box_dict in enumerate(pred_dicts):
                frame_id = batch_dict['frame_id'][index]
                single_pred_dict = generate_single_sample_dict_for_kitti(index, box_dict)
                single_pred_dict['frame_id'] = frame_id
                # print('==> box_dict.keys()', box_dict.keys()) # ['pred_boxes', 'pred_scores', 'pred_labels', 'pred_pointseg']

                if output_path is not None:
                    # cur_det_file = output_path / ('%s.txt' % frame_id)
                    # print('==> output_path: ', output_path)
                    # print('==> frame_id: ', frame_id) # frame_id:  segment-10203656353524179475_7625_000_7645_000_with_camera_labels_100
                    # print('==> cur_det_file: ', cur_det_file)
                    frame_str = frame_id[:-4]
                    frame_num = frame_id[-3:]
                    frame_path = output_path / frame_str
                    frame_path.mkdir(parents=True, exist_ok=True)
                    cur_det_file = frame_path / ('preddet_%s.txt' % frame_num)

                    # 保存det信息
                    with open(cur_det_file, 'w') as f:
                        bbox = single_pred_dict['bbox']
                        loc = single_pred_dict['location']
                        dims = single_pred_dict['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                                % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                    bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                    dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                    loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                    single_pred_dict['score'][idx]), file=f)      

                    # 保存point seg的信息
                    pred_pointseg = box_dict['pred_pointseg'].cpu().numpy()
                    # print('==> box_dict[pred_pointseg].shape: ', box_dict['pred_pointseg'].shape) #[16384, 4]
                    cur_pointseg_file = frame_path / ('pointseg_%s.npy' % frame_num)
                    np.save(cur_pointseg_file, pred_pointseg)

                    # 转gt的信息也为kitti的格式
                    cur_gtdet_file = frame_path / ('gtdet_%s.txt' % frame_num)
                    cur_gt = batch_dict['gt_boxes'][index] # (batch_size, N, 7+C+1), 最后一个才是label
                    k = cur_gt.__len__() - 1
                    while k > 0 and cur_gt[k].sum() == 0:
                        k -= 1
                    cur_gt = cur_gt[:k + 1]
                    cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
                    # 准备gtbox_dict, 都是tensor结构的，在里面转换为np
                    gtbox_dict = {
                        'pred_scores': cur_gt.new_zeros( (cur_gt.shape[0]) ),
                        'pred_boxes': cur_gt[:, 0:7],
                        'pred_labels': cur_gt[:, -1].long(), 
                    }
                    single_gt_dict = generate_single_sample_dict_for_kitti(index, gtbox_dict)
                    # 保存gtdet信息
                    with open(cur_gtdet_file, 'w') as f:
                        bbox = single_gt_dict['bbox']
                        loc = single_gt_dict['location']
                        dims = single_gt_dict['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                                % (single_gt_dict['name'][idx], single_gt_dict['alpha'][idx],
                                    bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                    dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                    loc[idx][1], loc[idx][2], single_gt_dict['rotation_y'][idx],
                                    single_gt_dict['score'][idx]), file=f)  


        return annos


    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        database_save_path = save_path / ('pcdet_gt_database_%s_sampled_%d' % (split, sampled_interval))
        db_info_save_path = save_path / ('pcdet_waymo_dbinfos_%s_sampled_%d.pkl' % (split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count()):
    workers = 4
    
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=10,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist']
    )
    print('---------------Data preparation Done---------------')



def create_waymo_infos_modify(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count()):
    """
    只更新val set info
    """
    print('======================> cpus: ', multiprocessing.cpu_count())
    workers = 4
    print('======================> workers: ', workers)

    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    # train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos (updating)---------------')

    '''
    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)
    '''

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    '''
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=10,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist']
    )
    '''
    print('---------------Data preparation (updating) Done---------------')



if __name__ == '__main__':
    import argparse
    print('==============> code runs here 0..........')

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        print('==============> code runs here 1..........')
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        print('==============> code runs here 2..........')
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )
