from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_batch_cuda as pointnet2


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features

def top3_interpolate(xyz, new_xyz, feats, nsamples=None):
    """
    Args:
        xyz: tensor, (N, C),
        feats: tensor, (N, Cf), features at xyz location
        new_xyz: tensor, (M, C)
        nsamples: an int number, but not used here.
    """
    # 检查数据！
    # print('==> xyz.shape: ', xyz.shape)
    # print(xyz[:50, :])
    # print('==> new_xyz.shape: ', new_xyz.shape)
    # print(new_xyz[:50, :])
    
    # 1. 准备batch idx
    if len(xyz.shape) == len(new_xyz.shape) == len(feats.shape) == 2:
        xyz_batch = xyz.unsqueeze(dim=0) # (N, C) -> (1, N, C)
        new_xyz_batch = new_xyz.unsqueeze(dim=0)# (M, C) -> (1, M, C)
        feats_batch = feats.unsqueeze(dim=0).permute(0,2,1).contiguous() # (N, Cf) -> (1, N, Cf) -> (1,Cf,N)
    else:
        raise NotImplementedError
    
    # 1. 计算距离
    dist, idx = three_nn(new_xyz_batch, xyz_batch)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    # print('==> idx.shape: ', idx.shape) # (1, M, nsamples)
    interpolated_feats_batch = three_interpolate(feats_batch, idx, weight).contiguous()    

    # (1, Cf, M) -> (1,M,Cf) -> (M, Cf)
    interpolated_feats = interpolated_feats_batch.permute(0,2,1).squeeze(dim=0)
    # print('==> interpolated_feats.shape: ', interpolated_feats.shape)

    return interpolated_feats


def top3_interpolate_with_grad(xyz, new_xyz, feats, nsamples=None):
    """
    Args:
        xyz: tensor, (N, C),
        feats: tensor, (N, Cf), features at xyz location
        new_xyz: tensor, (M, C)
        nsamples: an int number, but not used here.

        interpolated_feats: tensor, (M, Cf)
    """
    # assert new_xyz.requires_grad == feats.requires_grad == True # just when training

    # 1. 准备batch idx
    if len(xyz.shape) == len(new_xyz.shape) == len(feats.shape) == 2:
        # (N, C) -> (1, N, C)
        xyz_batch = xyz.unsqueeze(dim=0) 
        # (M, C) -> (1, M, C), 这个阻隔了梯度，用于idx获取
        new_xyz_batch = new_xyz.detach().unsqueeze(dim=0)
        # (N, Cf) -> (1, N, Cf) -> (1, Cf, N)
        feats_batch = feats.unsqueeze(dim=0).permute(0,2,1).contiguous()
        # (N, C) -> (1, N, C) -> (1, C, N)
        trans_xyz_batch = xyz.unsqueeze(dim=0).permute(0,2,1).contiguous()
    else:
        raise NotImplementedError

    # 2. idx 收集，这里不需要梯度，但在idx获取完后计算权重的时候需要梯度
    # idx.shape: (1, M, nsamples)
    dist_wo_grad, idx = three_nn(new_xyz_batch, xyz_batch)

    # idx用于gather, gather的时候xyz/feat 都需要梯度的
    # (1, C, M, nsamples) -> (C, M, nsamples) -> (M, nsamples, C)
    threenn_xyz = grouping_operation(trans_xyz_batch, idx).squeeze(dim=0).permute(1,2,0).contiguous()
    threenn_feats = grouping_operation(feats_batch, idx).squeeze(dim=0).permute(1,2,0).contiguous()

    # 计算weight进行加权
    # (M, nsamples, C) - (M, 1, C) , -> (M, nsamples, C), -> (M, nsamples)
    dist = torch.norm( (threenn_xyz - new_xyz.unsqueeze(dim=1).expand(-1, idx.shape[-1], -1)), dim=-1)
    # print('==> dist: ', dist)


    # (M, nsamples)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    # print('==> weight: ', weight)

    # assert weight.requires_grad == True # just when training
    
    # (M, nsamples, C) * (M, nsamples, 1), 
    # (M, nsamples, C) -> (M, C)
    interpolated_feats = torch.sum(threenn_feats * weight.unsqueeze(-1), dim=1)

    return interpolated_feats