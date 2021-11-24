# FromVoxelToPoint & MGAF-3DSSD

This is a reproduced repo of "From Voxel to Point: IoU-guided 3D Object Detection for Point Cloud with Voxel-to-Point Decoder" ([FromVoxelToPoint](https://arxiv.org/pdf/2108.03648.pdf)) and "Anchor-free 3D Single Stage Detector with Mask-Guided Attention for Point Cloud" ([MGAF-3DSSD](https://arxiv.org/pdf/2108.03634.pdf)) in ACM MM 2021. 

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
We provide codes and training configurations of FromVoxelToPoint & MGAF-3DSSD on the KITTI and Waymo datasets. Checkpoints will not be released.  

<!-- **Important Notes**: VoTr generally requires quite a long time (more than 60 epochs on Waymo) to converge, and a large GPU memory (32Gb) is needed for reproduction. -->
<!-- Please strictly follow the instructions and train with sufficient number of epochs. -->
<!-- If you don't have a 32G GPU, you can decrease the attention SIZE parameters in yaml files, but this may possibly harm the performance.  -->

## Requirements
The codes are tested in the following environment:
* Ubuntu 20.04.1 LTS 
* Python 3.6
* PyTorch 1.7.1+cu110
* CUDA 11.0
* OpenPCDet v0.3.0 (You can easily add the relevant codes to the latest OpenPCDet if you want.)
<!-- * spconv v1.2.1 -->
Note that we use a modified spconv to avoid sudo permission requirements during the installation process, which can be easily installed by setup.py.


## Installation
a. Clone this repository.
```shell
git clone https://github.com/jialeli1/From-Voxel-to-Point.git
```

b. Install the dependent python libraries as follows:
```shell
pip install -r requirements.txt 
```

c. Compile CUDA operators by running the following command:
* CUDA ops in OpenPCDet and the useful spconv.
```shell
python setup.py develop
```
* Deformable convolution that is modified from [Deformable-Convolution-V2-PyTorch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch.git).
```shell
cd pcdet/ops/DeformableConvolutionV2PyTorch
sh make.sh
```

## Dataset Preparation
We provide model configurations on KITTI and Waymo. Please follow OpenPCDet to prepare the datasets. You can also use "ln -s" to link an existing dataset here for a quick start.

## Training
Details are in paper. If you use different number of GPUs for training, it's necessary to change the respective training epochs to attain a decent performance. 

You can run training and evaluation commands following OpenPCDet. We also provide some examples on KITTI as follows.

### KITTI
* models
```shell
# MGAF-3DSSD: An RTX 3090 GPU (24G) can contrain 4 KITTI point clouds for training.
tools/cfgs/kitti_models/MGAF-3DSSD/mgaf-3dssd.yaml
tools/cfgs/kitti_models/MGAF-3DSSD/mgaf-3dssd_3classes.yaml

# FromVoxelToPoint: An RTX 3090 GPU (24G) can contrain 3 KITTI point clouds for training. It requires a large GPU memory for reproduction.
tools/cfgs/kitti_models/FV2P/fv2p.yaml
tools/cfgs/kitti_models/FV2P/fv2p_3classes.yaml
```

* training on KITTI
```shell script
cd tools

CUDA_VISIBLE_DEVICES=6,7 bash scripts/dist_train.sh 2 --cfg_file ./cfgs/kitti_models/MGAF-3DSSD/mgaf-3dssd.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/dist_train.sh 4 --cfg_file ./cfgs/kitti_models/FV2P/kitti_fv2p.yaml
```

* evaluation on KITTI
```shell
cd tools

CUDA_VISIBLE_DEVICES=7 python test.py --cfg_file ./cfgs/kitti_models/MGAF-3DSSD/mgaf-3dssd.yaml --eval_all
```

### Waymo
* models
```shell
# MGAF-3DSSD: 
tools/cfgs/waymo_models/MGAF-3DSSD/waymo_mgaf-3dssd_e36.yaml

# FromVoxelToPoint: 
tools/cfgs/waymo_models/FV2P/waymo_fv2p_e30.yaml
```


## Citation 
If you find this project useful in your research, please consider cite:

```
@inproceedings{fv2p_mm21,
  author    = {Jiale Li and
               Hang Dai and
               Ling Shao and
               Yong Ding},
  title     = {From Voxel to Point: IoU-guided 3D Object Detection for Point Cloud
               with Voxel-to-Point Decoder},
  booktitle = {{MM} '21: {ACM} Multimedia Conference},
  pages     = {4622--4631},
  year      = {2021},
}

@inproceedings{mgaf_mm21,
  author    = {Jiale Li and
               Hang Dai and
               Ling Shao and
               Yong Ding},
  title     = {Anchor-free 3D Single Stage Detector with Mask-Guided Attention for
               Point Cloud},
  booktitle = {{MM} '21: {ACM} Multimedia Conference},
  pages     = {553--562},
  year      = {2021},
}

```