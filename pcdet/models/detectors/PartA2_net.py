from .detector3d_template import Detector3DTemplate
import time
import torch

class PartA2Net(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if self.training:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time 
            recall_dicts.update( {'total_time': elapsed} )
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
