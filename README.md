# Vision Transformer with Deformable Attention

This repository contains the code of semantic segmentation for the paper Vision Transformer with Deformable Attention \[[arXiv](https://arxiv.org/abs/2201.00520)\], and DAT++: Spatially Dynamic Vision Transformerwith Deformable Attention (extended version)\[[arXiv](https://arxiv.org/abs/2309.01430)]. 

This code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [Swin Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation). To get started, you can follow the instructions in [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/README.md).

Other links:

- [Classification](https://github.com/LeapLabTHU/DAT)
- [Detection](https://github.com/LeapLabTHU/DAT-Detection)

## Dependencies

In addition to the dependencies of the [classification](https://github.com/LeapLabTHU/DAT) codebase, the following packages are required:

- mmcv-full == 1.4.0
- mmsegmentation == 0.29.0

## Evaluating Pretrained Models

### SemanticFPN

| Backbone | Schedule  | mIoU | mIoU+MS | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 80K | 48.4 | 48.8 | [config](configs/dat/fpn_tiny_80k_dp04_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroQipc3erCDV2MGng?e=5CiCVE) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/9a9df06dec1440a39bcb/) |
| DAT-S++ | 80K | 49.9 | 50.7 | [config](configs/dat/fpn_small_80k_dp04_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroTm5q7mwZFRvqKnw?e=RzLZgu) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f4333e084ef74ca3b8f6/) |
| DAT-B++ | 80K | 50.4 | 51.1 | [config](configs/dat/fpn_base_80k_dp07_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroRSpGKqGHBCD77MA?e=NK3Jsw) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/9df9eb212f154168aab7/) |

### UperNet

| Backbone | Schedule  | mIoU | mIoU+MS | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 160K | 49.4 | 50.3 | [config](configs/dat/upn_tiny_160k_dp03_lr6.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroSjoSXmkw32WulGQ?e=xtELZ9) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/de75c7e1622847268f4f/) |
| DAT-S++ | 160K | 50.5 | 51.2 | [config](configs/dat/upn_small_160k_dp05_lr6.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroU0mRYDsSLan5LPA?e=MvQ9rr) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/2f4a2be90a08477fa4b3/) |
| DAT-B++ | 160K | 51.0 | 51.5 | [config](configs/dat/upn_base_160k_dp07_lr6.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroVYhIIowSbZOFXFw?e=uNShfl) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/7c442d4371c4433aa428/) |

To evaluate a pretrained checkpoint, please download the pretrain weights to your local machine and run the mmsegmentation test scripts as follows:

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, MS testing
bash tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

**Please notice: Before training or evaluation, please set the `data_root` variable in `configs/_base_/datasets/ade20k.py` to the path where ADE20K data stores.**

Since evaluating models needs no pretrain weights, you can set the `pretrained = None` in `<CONFIG_FILE>`.

## Training

To train a segmentor with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> 
```

**Please notice: Make sure the `pretrained` variable in `<CONFIG_FILE>` is correctly set to the path of pretrained DAT model.**

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer), we thank to their efficient and neat codebase. The computational resources supporting this work are provided by [Hangzhou
High-Flyer AI Fundamental Research Co.,Ltd](https://www.high-flyer.cn/).


## Citation

If you find our work is useful in your research, please consider citing:

```
@article{xia2023dat,
    title={DAT++: Spatially Dynamic Vision Transformer with Deformable Attention}, 
    author={Zhuofan Xia and Xuran Pan and Shiji Song and Li Erran Li and Gao Huang},
    year={2023},
    journal={arXiv preprint arXiv:2309.01430},
}

@InProceedings{Xia_2022_CVPR,
    author    = {Xia, Zhuofan and Pan, Xuran and Song, Shiji and Li, Li Erran and Huang, Gao},
    title     = {Vision Transformer With Deformable Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4794-4803}
}
```

## Contact

If you have any questions or concerns, please send email to [xzf23@mails.tsinghua.edu.cn](mailto:xzf23@mails.tsinghua.edu.cn).