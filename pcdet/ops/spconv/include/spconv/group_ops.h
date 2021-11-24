// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPARSE_GROUP_OP_H_
#define SPARSE_GROUP_OP_H_

#include <cuda_runtime_api.h>
#include <spconv/indice.h>
#include <spconv/reordering.h>
// #include <ATen/ATen.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>

namespace spconv {
// torch.jit's doc says only support int64, so we need to convert to int32.

template <typename T>
torch::Tensor indiceGroup(torch::Tensor features,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM) {
  /*
  features: (N_b, C_in)
  output: (N_ks, N_b, C_in)
  */
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  // kernerlVolume is the N_ks
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  // keep the same as input.
  auto numOutPlanes = features.size(1); 
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;

  /*if (_subM){
    std::vector<int> indicePairNumVec(indicePairNumCpu.data_ptr<int>(),
  indicePairNumCpu.data_ptr<int>() + kernelVolume);
    indicePairNumVec.erase(indicePairNumVec.begin() + indicePairMaxOffset);

    auto indicePairVecMaxSizeIter = std::max_element(
        indicePairNumVec.begin(), indicePairNumVec.end());
    indicePairMaxSize = *indicePairVecMaxSizeIter;
  }*/
  
  auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // auto indicePairOptions =
  //     torch::TensorOptions().dtype(torch::kInt64).device(indicePairs.device());

  // (N_ks, N_b, C)
  torch::Tensor output = torch::zeros({kernelVolume, numActOut, numInPlanes}, options);
  // torch::Tensor output = torch::zeros({numActOut, numInPlanes}, options);

  torch::Tensor inputBuffer =
      torch::zeros({indicePairMaxSize, numInPlanes}, options);
  // 这个outputBuffer难道是共享的吗？
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // 对kernel的每个位置进行循环
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    // kernel的位置i处一共有nHot个相关的输出

    // std::cout << "kernelVolume: " << kernelVolume << std::endl;
    // std::cout << "nHot: " << nHot << std::endl;
    
    if (nHot <= 0) {
      // 对于non_act位置的点，直接跳过赋值
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    // torch::from_blob 应该是转换为tensor的意思
    // kernel的每一个位置i都重新创建outputBufferBlob
    // 每次循环都重新清空
    // auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);
    // torch::Tensor output_i = torch::zeros({numActOut, numInPlanes}, options);

    // 原本的整个过程是
    // features -> inputBuffer (-> outputBuffer) -> output
    // 现在是把中间到(-> outputBuffer)这一步省略，直接到scatter到output上

    // std::cout << "device: " << device << std::endl;

    if (device == torch::kCPU) {
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      // 把features -> inputBuffer中
      // 按照kernel的位置i处的index去取.
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse), //取位置i处的kernel的input
                 nHot);
      // std::cout << "after cpu gather. " << std::endl;
    } else {
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
      gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      // std::cout << "after gpu gather. " << std::endl;
      TV_CHECK_CUDA_ERR();
      // std::cout << "after TV_CHECK_CUDA_ERR(). " << std::endl;
      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data_ptr<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
    // totalGatherTime += timer.report() / 1000.0;
    // outputBufferBlob 和 inputBufferBlob都是torch::from_blob返回的一个指针
    // torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);

    // 这里不进行乘积 只进行赋值呢
    // 应该这样写还是用 memcpy ？
    // 不能使用memcppy, 因为这里的数据应该是在GPU上的.
    // 实际上在gather的时候进行了copy的，那么这里是否不再需要copy？
    // outputBufferBlob = torch::(inputBufferBlob)
    // outputBufferBlob = torch::clone(inputBufferBlob);
    // outputBufferBlob = inputBufferBlob;

    // std::cout << "inputBuffer.shape[0]: "<< inputBuffer.size(0) << std::endl;
    // std::cout << "inputBuffer.shape[1]: "<< inputBuffer.size(1) << std::endl;
    // std::cout << "inputBuffer[0]: "<< inputBuffer[0] << std::endl;

    // totalGEMMTime += timer.report() / 1000.0;

    // 把outputBuffer 放到output中的某个地方.
    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(output[i]),
                  tv::torch2tv<const T>(inputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
      // std::cout << "after cpu scatteradd. " << std::endl;
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      // 这里是不是可以选择直接把output的第i个位置送进去就好了？
      // add 也只是和0相加
      scatterFtor(tv::TorchGPU(), 
                  tv::torch2tv<T>(output[i]), 
                  tv::torch2tv<const T>(inputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);

      // std::cout << "after gpu scatteradd. " << std::endl;
      // std::cout << "output[i][0].data(): "<< output[i][0].data() << std::endl;
      // std::cout << "output[i].data(): "<< output[i].data() << std::endl;
      
      TV_CHECK_CUDA_ERR();
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // std::cout << "gather time " << totalGatherTime << std::endl;
  // std::cout << "gemm time " << totalGEMMTime << std::endl;
  // std::cout << "scatteradd time " << totalSAddTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceGroupBackward(torch::Tensor features,
                                  torch::Tensor outGrad,
                                  torch::Tensor indicePairs,
                                  torch::Tensor indiceNum,
                                  int64_t _inverse, 
                                  int64_t _subM) {
  /*
  features: (N_in, C)
  outGrad: (N_ks, N_out, C)
  
  inputGrad: (N_in, C)
  */
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  // std::cout << "features.shape: "<< features.sizes() << std::endl;
  // std::cout << "outGrad.shape: "<< outGrad.sizes() << std::endl;

  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto numOutPlanes = outGrad.size(2);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset =
      indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;
  auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);

  // std::cout << "inputGrad.shape: "<< inputGrad.sizes() << std::endl;


  // torch::Tensor inputBuffer = torch::zeros({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer = torch::zeros({indicePairMaxSize, numOutPlanes}, options);

  // std::cout << "outputBuffer.shape: "<< outputBuffer.sizes() << std::endl;


  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    
    // 这里缩减了条件，不区分subM
    // (nHot <= 0 || (subM && i == indicePairMaxOffset))
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      // 把out_grad 按kernel位置进行收集到所有和这个kernel位置有关的outputbuffer上
      /*
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      */
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtorOut;
      gatherFtorOut(tv::CPU(), tv::torch2tv<T>(outputBuffer),
                    tv::torch2tv<const T>(outGrad[i]),
                    tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                    nHot);
    } else {
      /*
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
      gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      TV_CHECK_CUDA_ERR();
      */
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtorOut;
      gatherFtorOut(tv::TorchGPU(), tv::torch2tv<T>(outputBuffer),
                    tv::torch2tv<const T>(outGrad[i]),
                    tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                    nHot);
      TV_CHECK_CUDA_ERR();
    }
    // std::cout << "outGrad[i][0].data(): "<< outGrad[i][0].data() << std::endl;

    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
    // auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);
    
    // std::cout << "outputBuffer.shape: "<< outputBuffer.sizes() << std::endl;
    // std::cout << "outputBuffer[0]: "<< outputBuffer[0] << std::endl;

    // 直接传递, feature_grad -> outputBufferBlob (-> inputBufferBlob) -> inputGrad
    // 同一个点收到多个group的梯度来源时候, 采用梯度累加的方式.
    
    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(inputGrad),
                  tv::torch2tv<const T>(outputBufferBlob),
                  tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                  nHot);
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      scatterFtor(tv::TorchGPU(), tv::torch2tv<T>(inputGrad),
                  tv::torch2tv<const T>(outputBufferBlob),
                  tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                  nHot);
      TV_CHECK_CUDA_ERR();
    }

    // std::cout << "inputGrad[0].data(): "<< inputGrad[0].data() << std::endl;

  }
  // std::cout << "final inputGrad[0].data(): "<< inputGrad[0].data() << std::endl;
  // std::cout << "final inputGrad[100].data(): "<< inputGrad[100].data() << std::endl;
  return inputGrad;
}

}  // namespace spconv

#endif
