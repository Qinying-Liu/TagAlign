# TagAlign - Official Pytorch Implementation

> **TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification** <br>
> Qinying Liu, Kecheng Zheng, Wei Wu, Zhan Tong, Yu Liu, Wei Chen, Zilei Wang, Yujun Shen<br>

[![Arxiv](https://img.shields.io/badge/Arxiv-TagAlign.pdf-red)](https://arxiv.org/pdf/2312.14149.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-TagAlign-yellow)](https://drive.google.com/drive/folders/1jhc3Eef6mUpsIYuMY5XWzeDqrIhJSDDF?usp=sharing)

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

## 📜 News
[2024/12/11] Release [tag](https://drive.google.com/drive/folders/1jhc3Eef6mUpsIYuMY5XWzeDqrIhJSDDF?usp=sharing).

[2023/12/25] The [paper](https://arxiv.org/abs/2312.14149) and [project page](https://qinying-liu.github.io/Tag-Align/) are released!

## 💡 Highlights
- 🔥 **3.65%** mIOU improvement on a broad suite of semantic segmentation datasets (VOC: PASCAL VOC, Context: PASCAL Context, Object: COCO-Object, IN: ImageNet-S, Stuff: COCO-Stuff, City: Cityscapes, ADE: ADE20K).
- 🔥 **A strong CLIP encoder** with the help of designed parsing pipeline that is fully automatic and thus enjoys good scalability.
  
## 👨‍💻 Todo
- [ ] Checkpoints of TagAlign
- [ ] Web demo and local demo of TagAlign
- [x] Meta-files of TagAlign 
- [x] Training and evaluation code for TagAlign

## 🛠️ Usage

### Installation
* apex==0.1
* clip==1.0
* mmcv-full==1.4.7
* mmsegmentation==0.21.1
* torch==1.11.0

### Data Preparation
For the training phase, we utilize the CC12M dataset. Researchers can procure the CC12M dataset either directly from its [source](https://github.com/google-research-datasets/conceptual-12m) or by employing the [img2dataset](https://github.com/rom1504/img2dataset) tool. 

For evaluation, refer to the [GroupVit](https://github.com/NVlabs/GroupViT) to properly prepare the datasets. Make sure to update the image directories in 'segmentation/configs/_base_/datasets/*.py' as necessary.

### Train and Evaluate
1. Modify the 'tagalign.yml'. We provide the processed tag_file (object_list.csv) and label_file (CC12M_url_object_index.json). We provide the URLs of the images; please change them to your local paths.

2. Train the TagAlign model by run 
   ```
   torchrun --rdzv_endpoint=localhost:6000 --nproc_per_node=auto main.py --cfg configs/tagalign.yml
   ```
3. You can evaluate the TagAlign model by running the command below.
   ```
   torchrun --rdzv_endpoint=localhost:6000 --nproc_per_node=auto main.py --cfg configs/eval.yml --eval --resume $WEIGHT
   ```
   $WEIGHT is the path of the pre-trained checkpoints. We provide our pre-trained weights in [weights(TODO)]().

 ## ✒️ Citation

 If you find our work to be useful for your research, please consider citing.

```
@article{liu2023tagalign,
  title={TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification},
  author={Liu, Qinying and Zheng, Kecheng and Wei, Wu and Tong, Zhan and Liu, Yu and Chen, Wei and Wang, Zilei and Shen, Yujun},
  journal={arXiv preprint arXiv:2312.14149},
  year={2023}
}
```

 ## ❤️ Acknowledgements

* [TCL](https://github.com/kakaobrain/tcl): The codebase we built upon. Thanks for their wonderful work.
* [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery): An effective training-free strategy for enhancing the fine-grained localization capabilities of CLIP.


