# TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification

This repository is the official implementation of [TagAlign](https://arxiv.org/abs/2301.09121).

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


