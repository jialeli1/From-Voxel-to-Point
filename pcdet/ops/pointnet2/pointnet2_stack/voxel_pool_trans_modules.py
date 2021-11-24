import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List


class NeighborVoxelSAModuleMSG(nn.Module):

    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], 
        nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)
        
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]

            cur_mlp_in = nn.Sequential(
                nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[1])
            )
            
            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[1])
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[2]),
                nn.ReLU()
            )

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                                        new_coords, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # change the order to [batch_idx, z, y, x]
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_features_list = []
        neighbor_feature_list = [] # for msg, but maybe only ssg used.
        neighbor_xyz_list = [] # for msg, but maybe only ssg used.
        for k in range(len(self.groupers)):
            # features_in: (1, C, M1+M2)
            features_in = features.permute(1, 0).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)
            # features_in: (1, M1+M2, C)
            features_in = features_in.permute(0, 2, 1).contiguous()
            # features_in: (M1+M2, C)
            features_in = features_in.view(-1, features_in.shape[-1])
            # grouped_features: (M1+M2, C, nsample)
            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
            )
            grouped_features[empty_ball_mask] = 0

            # grouped_features: (1, C, M1+M2, nsample)
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            
            # print('==> grouped_features.shape', grouped_features.shape)
            # print('==> grouped_features[0, :5, 100, :]', grouped_features[0, :5, 100, :])
            # print('==> grouped_features[0, :5, 400, :]', grouped_features[0, :5, 400, :])
            # print('==> grouped_features[0, :5, 800, :]', grouped_features[0, :5, 800, :])
            # torch.cuda.synchronize()
            # print('==> grouped_xyz.shape', grouped_xyz.shape)
            # print('==> new_xyz[100, ...]: ', new_xyz[100, ...])
            # print('==> grouped_xyz[100, ...]', grouped_xyz[100, ...])
            # print('==> new_xyz[10400, ...]: ', new_xyz[10400, ...])
            # print('==> grouped_xyz[10400, ...]', grouped_xyz[10400, ...])
            # print('==> new_xyz[10800, ...]: ', new_xyz[10800, ...])
            # print('==> grouped_xyz[10800, ...]', grouped_xyz[10800, ...])

            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample)
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            # grouped_xyz: (1, C, M1+M2, nsample)
            position_features = self.mlps_pos[k](grouped_xyz)
            grouped_new_features = grouped_features + position_features
            grouped_new_features = self.relu(grouped_new_features)
            
            # 把这个增加了pos的特征保存出来，作为邻域特征，不过是否需要对齐呢，去外面对齐？
            # 这里或许已经是对齐了的，通过合理配置mlp
            # (1, C, M1+M2, nsample)
            neighbor_feature_list.append(grouped_new_features)
            # (1, 3, M1+M2, nsample)
            neighbor_xyz_list.append(grouped_xyz) 

            # grouped_new_features -> new_features, 
            # 这里是先pooling了再变换维度，节省一点内存的。
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    grouped_new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    grouped_new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            
            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)
        
        # (M1 + M2 ..., C)
        new_features = torch.cat(new_features_list, dim=1)

        # (1, C, M1+M2, nsample_s1/nsample_s2...) -> (1, C, M1+M2, nsample_s1 + nsample_s2)
        neighbor_features = torch.cat(neighbor_feature_list, dim=-1) 
        # (1, C, M1+M2, nsample_s1 + nsample_s2) -> (M1+M2, nsample_s1 + nsample_s2, C)
        neighbor_features = neighbor_features.squeeze(dim=0).permute(1,2,0).contiguous()

        neighbor_xyz = torch.cat(neighbor_xyz_list, dim=-1) 
        neighbor_xyz = neighbor_xyz.squeeze(dim=0).permute(1,2,0).contiguous()

        return new_features, neighbor_features, neighbor_xyz


class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super().__init__()
        # donnt do self_attn
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=nc_mem, vdim=nc_mem)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)

        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: (B, L1, E1)
        mem: (B, L2, E2)

        """
        
        # mem是可以self-atten一下的
        # 但是通道太少了, 先不做
        """
        memory = self.norm1(memory)
        memory2 = self.self_attn(memory, memory, memory, attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        memory = memory + self.dropout1(memory2)
        """

        # tgt attend to mem.
        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        # 在multihead_attn里面会做qkv_proj, 将qkv都投影到d_model上
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class PointNeighborTransformer(nn.Module):
    def __init__(self, dim_in, dim_out, nhead=4, num_layers=1, drop=0.0, dim_feature=32, prenorm=True):

        super().__init__()    

        self.nc_in = dim_in
        self.nc_out = dim_out
        self.nhead = nhead

        self.pe = nn.Sequential(
            # conv-bn-relu-conv, no/bias when without/with norm
            nn.Conv2d(3, self.nc_in // 2, 1, bias=False),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1, bias=True)
            )

        self.chunk = nn.TransformerDecoder(
            TransformerDecoderLayerPreNorm(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead, nc_mem=dim_feature),
            num_layers=num_layers,
        )

        self.fc = nn.Linear(self.nc_in, self.nc_out, bias=True)

    def forward(self, xyz_tgt, xyz_mem, features_tgt, features_mem):
        """
        xyz_tgt: (M1+M2+..., 3) 
        xyz_mem: (M1+M2+..., N_mem, 3)
        features_tgt: (M1+M2+..., C_tgt(=d_model)), 这个应该是已经对齐到d_model上了的特征.
        features_mem: (M1+M2+..., N_mem, C_mem)
        """
        # (M1+M2+..., 3) -> (M1+M2+..., N_tgt=1, 3) -> (M1+M2+..., 3, N_tgt, 1) 
        xyz_tgt_flipped = xyz_tgt.unsqueeze(1).transpose(1,2).unsqueeze(-1)
        xyz_mem_flipped = xyz_mem.transpose(1,2).unsqueeze(-1)

        # (M1+M2+..., C_tgt, N_tgt, 1) 
        tgt = features_tgt.unsqueeze(1).transpose(1,2).unsqueeze(-1) + self.pe(xyz_tgt_flipped)
        mem = features_mem.transpose(1,2).unsqueeze(-1) + self.pe(xyz_mem_flipped)

        mem_mask = None

        # torch.nn.MultiheadAttention requires the (L, B, E) shape.
        # (N_tgt, M1+M2+..., C_tgt) 
        tgt = tgt.squeeze(-1).permute(2, 0, 1)
        mem = mem.squeeze(-1).permute(2, 0, 1)

        # back to (M1+M2+..., N_tgt, C_tgt) 
        transformed_feats = self.chunk(tgt, mem, memory_mask=mem_mask).permute(1,0,2)
        
        # one more fc or not?
        # (M1+M2+..., N_tgt, cout)
        output = self.fc(transformed_feats)


        return output




