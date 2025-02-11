# DRFNet-Inpainting

code for paper 'Degression receptive field network for image inpainting'.

## Prerequisites

- Python 3.7
- NVIDIA GPU + CUDA cuDNN 10.1
- PyTorch 1.8.1

## TODO

- [x] Releasing evaluation code.
- [x] Releasing inference codes.
- [ ] Releasing pre-trained weights.
- [x] Releasing training codes.


## Download Datasets

We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the testing mask.

## Run
1. train the model
```
train.py --dataroot no_use --name Psv_DRFNet --model pix2pixglg --netG1 unet_256 --netD snpatch --gan_modes lsgan --input_nc 4 --no_dropout --direction AtoB --display_id 0
```
2. test the model 
```
test_and_save_epoch.py --dataroot no_use --name Psv_DRFNet --model pix2pixglg --netG1 unet_256 --gan_mode nogan --input_nc 4 --no_dropout --direction AtoB --gpu_ids 0
```


## Citation

@article{MENG2024109397,
title = {Degression receptive field network for image inpainting},
journal = {Engineering Applications of Artificial Intelligence},
volume = {138},
pages = {109397},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.109397},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624015550},
author = {Jiahao Meng and Weirong Liu and Changhong Shi and Zhijun Li and Chaorong Liu},
keywords = {Image inpainting, Generative adversarial networks, Degression receptive field, Coarse to fine inpainting network, Object removal and image editing, Deep learning},
abstract = {—Multi-stage image inpainting methods from coarse-to-fine have achieved satisfactory inpainting results in recent years. However, an in-depth analysis of multi-stage inpainting networks reveals that simply increasing complexity of refined network may lead to degradation problems. The paper proposes a degression receptive field network (DRFNet) via multi-head attention mechanism and U-shaped network with different receptive fields to address above phenomenon that existing image inpainting methods have detail blur and artifacts due to insufficient constraints. Initially, DRFNet innovatively takes receptive field as a perspective and consists of five sub-networks with decreasing receptive fields. Secondly, an easy-to-use TransConv module is designed to overcome the problem of local-pixel influence in convolution. Experiments show that comprehensive optimal rate of DRFNet on L1 error, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Fréchet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS) is more than 82.86% on all three benchmark datasets, which achieves state-of-the-art results. Moreover, real-world experiments demonstrate the potential of DRFNet for object removal and image editing. The code is available at: https://github.com/IPCSRG/DRFNet-Inpainting.git.}
}

## Acknowledgments

This code based on [LGNet](https://github.com/weizequan/LGNet).
The evaluation code is borrowed from [TFill](https://github.com/lyndonzheng/TFill).
Please consider to cite their papers.
```
@ARTICLE{9730792,
  author={Quan, Weize and Zhang, Ruisong and Zhang, Yong and Li, Zhifeng and Wang, Jue and Yan, Dong-Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Inpainting With Local and Global Refinement}, 
  year={2022},
  volume={31},
  pages={2405-2420}
}
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei and Phung, Dinh},
    title     = {Bridging Global Context Interactions for High-Fidelity Image Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {11512-11522}
}
```
