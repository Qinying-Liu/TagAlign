# TagAlign - Official Pytorch Implementation

> **TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification** <br>
> Qinying Liu, Kecheng Zheng, Wei Wu, Zhan Tong, Yu Liu, Wei Chen, Zilei Wang, Yujun Shen<br>

[![Arxiv](https://img.shields.io/badge/Arxiv-TagAlign.pdf-red)](https://arxiv.org/pdf/2312.14149.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-4)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-4?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-3)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-3?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-10)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-10?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-9)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-9?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-8)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-8?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/unsupervised-semantic-segmentation-with-7)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-7?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/open-vocabulary-semantic-segmentation-on-5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-5?p=tagalign-improving-vision-language-alignment)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tagalign-improving-vision-language-alignment/open-vocabulary-semantic-segmentation-on-1)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-1?p=tagalign-improving-vision-language-alignment)

<div align="center">
<img src="figs/pipeline.png" width="100%">
</div>

## Requirements
* apex==0.1
* clip==1.0
* mmcv-full==1.4.7
* mmsegmentation==0.21.1
* torch==1.11.0

## Data Preparation
For the training phase, we utilize the CC12M dataset. Researchers can procure the CC12M dataset either directly from its [source](https://github.com/google-research-datasets/conceptual-12m) or by employing the [img2dataset](https://github.com/rom1504/img2dataset) tool. The dataset should adhere to the following file structure:

```shell
CC12M
├── 000002a0c848e78c7b9d53584e2d36ab0ac14785.jpg
├── 000002ca5e5eab763d95fa8ac0df7a11f24519e5.jpg
├── 00000440ca9fe337152041e26c37f619ec4c55b2.jpg
...
```
In addition, we provide the captions of the images in [meta_file(todo)]().

For evaluation, refer to the [GroupVit](https://github.com/NVlabs/GroupViT) to properly prepare the datasets. Make sure to update the image directories in 'segmentation/configs/_base_/datasets/*.py' as necessary.

## Train and Evaluate
1. Modify the 'tagalign.yml'. We provide the processed [tag_file(todo)]() and [label_file(todo)](). 

2. Train the TagAlign model by run 
   ```
   torchrun --rdzv_endpoint=localhost:6000 --nproc_per_node=auto main.py --cfg configs/tagalign.yml
   ```
3. You can evaluate the TagAlign model by running the command below.
   ```
   torchrun --rdzv_endpoint=localhost:6000 --nproc_per_node=auto main.py --cfg configs/eval.yml --eval --resume $WEIGHT
   ```
   $WEIGHT is the path of the pre-trained checkpoints. We provide our pre-trained weights in [weights(todo)]().

 ## Citation

 If you find our work to be useful for your research, please consider citing.

```
@article{liu2023tagalign,
  title={TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification},
  author={},
  journal={},
  year={2023}
}
```

 ## References

* [GroupViT](https://github.com/NVlabs/GroupViT)
* [TCL](https://github.com/kakaobrain/tcl)
* [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery)
* [OVSegmentor/](https://github.com/Jazzcharles/OVSegmentor)


