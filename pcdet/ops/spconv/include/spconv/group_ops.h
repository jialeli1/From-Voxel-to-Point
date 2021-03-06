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
  // ??????outputBuffer????????????????????????
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // ???kernel???????????????????????????
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    // kernel?????????i????????????nHot??????????????????

    // std::cout << "kernelVolume: " << kernelVolume << std::endl;
    // std::cout << "nHot: " << nHot << std::endl;
    
    if (nHot <= 0) {
      // ??????non_act?????????????????????????????????
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    // torch::from_blob ??????????????????tensor?????????
    // kernel??????????????????i???????????????outputBufferBlob
    // ???????????????????????????
    // auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);
    // torch::Tensor output_i = torch::zeros({numActOut, numInPlanes}, options);

    // ????????????????????????
    // features -> inputBuffer (-> outputBuffer) -> output
    // ?????????????????????(-> outputBuffer)???????????????????????????scatter???output???

    // std::cout << "device: " << device << std::endl;

    if (device == torch::kCPU) {
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      // ???features -> inputBuffer???
      // ??????kernel?????????i??????index??????.
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse), //?????????i??????kernel???input
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
    // outputBufferBlob ??? inputBufferBlob??????torch::from_blob?????????????????????
    // torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);

    // ????????????????????? ??????????????????
    // ???????????????????????? memcpy ???
    // ????????????memcppy, ?????????????????????????????????GPU??????.
    // ????????????gather??????????????????copy????????????????????????????????????copy???
    // outputBufferBlob = torch::(inputBufferBlob)
    // outputBufferBlob = torch::clone(inputBufferBlob);
    // outputBufferBlob = inputBufferBlob;

    // std::cout << "inputBuffer.shape[0]: "<< inputBuffer.size(0) << std::endl;
    // std::cout << "inputBuffer.shape[1]: "<< inputBuffer.size(1) << std::endl;
    // std::cout << "inputBuffer[0]: "<< inputBuffer[0] << std::endl;

    // totalGEMMTime += timer.report() / 1000.0;

    // ???outputBuffer ??????output??????????????????.
    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(output[i]),
                  tv::torch2tv<const T>(inputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
      // std::cout << "after cpu scatteradd. " << std::endl;
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      // ????????????????????????????????????output??????i??????????????????????????????
      // add ????????????0??????
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
    
    // ?????????????????????????????????subM
    // (nHot <= 0 || (subM && i == indicePairMaxOffset))
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      // ???out_grad ???kernel????????????????????????????????????kernel???????????????outputbuffer???
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

    // ????????????, feature_grad -> outputBufferBlob (-> inputBufferBlob) -> inputGrad
    // ????????????????????????group?????????????????????, ???????????????????????????.
    
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
