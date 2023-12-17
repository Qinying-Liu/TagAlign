# 训练：
1. 使用Gdecoder作为projector: torchrun --rdzv_endpoint=localhost:5000 --nproc_per_node=auto main.py --cfg configs/tagalign.yml
2. 使用transformer作为projector: torchrun --rdzv_endpoint=localhost:5000 --nproc_per_node=auto main.py --cfg configs/tagalign_transformer.yml


# 测试过程：
1. 使用Gdecoder作为projector: torchrun --rdzv_endpoint=localhost:5000 --nproc_per_node=auto main.py --cfg configs/eval.yml --eval --resume /nas2/lqy/taglign/tagalign/checkpoint.pth 
2. 使用transformer作为projector: torchrun --rdzv_endpoint=localhost:5000 --nproc_per_node=auto main.py --cfg configs/eval_transformer.yml --eval --resume /nas2/lqy/taglign/tagalign_transformer/checkpoint.pth 

主要是需要修改配置文件: ‘eval.yml’, 包含的超参数有: 
bg_thresh: 背景阈值， 注意[coco_stuff， context59， voc20， cityscapes， ade20k]没有背景类
clip_w: 和CLIP得到的结果进行结合， 即clip_w * mask + (1 - clip_w) * clip_mask
task: 要测试的数据集，每个数据集的路径需要在TagAlign/segmentation/configs/_base_/datasets里面各个文件下修改