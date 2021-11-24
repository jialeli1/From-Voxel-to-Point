# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.autograd import Function

from . import ops as ops


class SparseConvFunction(Function):

    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_pair_num,
                num_activate_out):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors

        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False)

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):

    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_pair_num,
                num_activate_out):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, True, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            True, False)

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):

    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_pair_num,
                num_activate_out):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False, True)
        

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, indice_pairs, indice_pair_num,
                num_activate_out):
        out = ops.indice_maxpool(features, indice_pairs, indice_pair_num,
                                 num_activate_out)
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output,
                                               indice_pairs, indice_pair_num)
        return input_bp, None, None, None


indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply



class SparseGroupFunction(Function):

    @staticmethod
    def forward(ctx, features, indice_pairs, indice_pair_num,
                num_activate_out):
        """
        features: (N_b, N_ks, C)
        indice_pairs: (N_ks, 2, N_out)

        return: (N_ks, N_out, C), 因为不知道怎么索引数组，所以shape是这样.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features)
        return ops.indice_group(features, indice_pairs,
                               indice_pair_num, num_activate_out)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features = ctx.saved_tensors
        input_bp = ops.indice_group_backward(
            features, grad_output, indice_pairs, indice_pair_num)

        return input_bp, None, None, None



class SubMGroupFunction(Function):

    @staticmethod
    def forward(ctx, features, indice_pairs, indice_pair_num,
                num_activate_out):
        """
        features: (N_b, C)
        indice_pairs: (N_ks, 2, N_out)

        return: (N_ks, N_out, C), 因为不知道怎么索引数组，所以shape是这样.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features)
        return ops.indice_group(features, indice_pairs,
                               indice_pair_num, num_activate_out, 
                               False, True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (N_ks, N_out, C)

        features: (N_in, C)
        """
        indice_pairs, indice_pair_num, features = ctx.saved_tensors
        # print('==> sumGroup, grad_output.shape: ', grad_output.shape)
        # print('==> sumGroup, features.shape: ', features.shape)
        
        input_bp = ops.indice_group_backward(
            features, 
            grad_output, 
            indice_pairs, 
            indice_pair_num, 
            False, 
            True)
        # print('==> sumGroup, input_bp.shape: ', input_bp.shape)
        
        # print('==> sumGroup, grad_output[:, 10, :]: ', grad_output[:, 10, :])
        # print('==> sumGroup, grad_output[:, 100, :]: ', grad_output[:, 100, :])
        # print('==> sumGroup, input_bp[10, :]: ', input_bp[10, :])
        # print('==> sumGroup, input_bp[100, :]: ', input_bp[100, :])

        return input_bp, None, None, None




indice_group = SparseGroupFunction.apply
indice_subm_group = SubMGroupFunction.apply
