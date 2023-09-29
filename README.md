# Matrix-SSL

Official implementation of paper 'Matrix Information Theory for Self-supervised Learning' (https://arxiv.org/abs/2305.17326).



## Introduction

PyTorch Implementation for Matrix-SSL, Modified from MEC Codes.



## Installation

Requirement:
- Conda

Once installed conda, you can create the `ssl` environment using 
`conda env create -f environment_ssl.yaml`.



## Pretraining
In order to get the best checkpoint over 100 epochs, run 
```bash
$ python main_pretrain.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10501' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mce_mu 1.0 --mce_lambd 1.0 --HSIC --gamma 1.0 \ 
  /path/to/imagenet/dataset/
```



## Linear Probing

To evaluate the pre-trained models by linear probing, simply run:
```bash
$ python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10051' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained /path/to/pretrained/checkpoint \
  --lars \
  /path/to/imagenet/dataset
```
## Transfer learning on COCO

We use exactly the same code as [MoCo](https://github.com/facebookresearch/moco/tree/main/detection). One can just follow its instructions, firstly convert the pre-trained weights to detectron2's format, then do transfer learning. 

## Acknowledgement

This repo is mainly based on [MEC](https://github.com/xinliu20/MEC) and [MoCo](https://github.com/facebookresearch/moco). Many thanks to their wonderful work!



