# DRFNet-Inpainting

code for paper "Degression receptive field network for image inpainting".

## Prerequisites

- Python 3.7
- NVIDIA GPU + CUDA cuDNN 10.1
- PyTorch 1.8.1

## TODO

- [x] Releasing evaluation code.
- [x] Releasing inference codes.
- [ ] Releasing pre-trained weights. (before 2025.2.30)
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
```
@article{MENG2024109397,
  author = {Meng, Jiahao and Liu, Weirong and Shi, Changhong and Li, Zhijun and Liu, Chaorong},
  journal = {Engineering Applications of Artificial Intelligence},
  title = {Degression receptive field network for image inpainting},
  year = {2024},
  volume = {138},
  pages = {109397},
}
```
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
