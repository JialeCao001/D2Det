# D2Det

This code is a official implementation of "D2Det: Towards High Quality Object Detection and Instance Segmentation (CVPR2020)" based on the open source object detection toolbox [mmdetection](https://github.com/open-mmlab/mmdetection). 

## Introduction
We propose a novel two-stage detection method, D2Det,that collectively addresses both precise localization and ac-curate classification. For precise localization, we introducea dense local regression that predicts multiple dense boxoffsets for an object proposal. Different from traditional re-gression and keypoint-based localization employed in two-stage detectors, our dense local regression is not limited toa quantized set of keypoints within a fixed region and hasthe ability to regress position-sensitive real number denseoffsets, leading to more precise localization. The dense lo-cal regression is further improved by a binary overlap pre-diction strategy that reduces the influence of background re-gion on the final box regression. For accurate classification,we introduce a discriminative RoI pooling scheme that sam-ples from various sub-regions of a proposal and performsadaptive weighting to obtain discriminative features.

## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) of mmdetection.

## Train and Inference
Please use the following commands for training and testing by single GPU or multiple GPUs.

######  Train with a single GPU
```shell
python tools/train.py ${CONFIG_FILE}
```

######  Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
######  Test with a single GPU

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

######  Test with multiple GPUs

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```


-- CONFIG_FILE about D2Det is in [configs/D2Det](configs/D2Det).

-- Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for more details.


## Results
|    name     |  type  | validation | test-dev | download|
| :-------------: | :-------------------: | :-----: | :------: | :-----------------: |
|     D2Det-res50     |  object detection  |   xx.x    |    33.9     |          [model](-)         |
|      D2Det-res101      |  object detection  |  44.9    |    45.4      |       [model](https://drive.google.com/open?id=14Cw9Y3vSdirkR3xLcb6F6H1hHr3qzLNj)         |
|     D2Det-res101-dcn    |  object detection  |  46.9   |    47.5    |        [model](https://drive.google.com/open?id=1jDeAj_rMKLMf64BGwqiysis9IyZzTQ6w)         |
|     D2Det-res101     |  instance segmentation  |  xx.x    |    40.2    |          [model](https://drive.google.com/open?id=1rsYWWJ7zJ7-sSWz5q6aiuGFJS5bduSDo)         |

## Ciatation
If the project helps your research, please cite this paper.

```
@misc{Cao_D2Det_ICCV_2020,
  author =       {Jiale Cao and Hisham Cholakkal and Rao Muhammad Anwer and Fahad Shahbaz Khan and Yanwei Pang and Ling Shao},
  title =        {D2Det: Towards High Quality Object Detection and Instance Segmentation},
  journal =      {Proc. IEEE Conference on Computer Vision and Pattern Recognition},
  year =         {2020}
}
```