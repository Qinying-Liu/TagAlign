# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------


import os.path as osp
import random
import warnings
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from mmcv.parallel import collate
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
if timm.__version__ == '0.6.12':
    from timm.data.transforms import str_to_pil_interp as _pil_interp
else:
    from timm.data.transforms import _pil_interp
# this works for timm==0.3.2
# from timm.data.transforms import _pil_interp 
from torchvision import transforms
import torch.nn as nn
from PIL import ImageFilter,Image
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
import math
import torchvision.transforms.functional as F
import shutil

from .formatting import ToDataContainer
from .tokenizer import SimpleTokenizer
from .clip_dataset import ClipDataset
from ipdb import set_trace

from sclip.clip import tokenize


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(batch):  
    img = torch.stack([b['image'] for b in batch])
    caption = torch.stack([b['caption'] for b in batch])
    tag_label = torch.stack([b['tag_label'] for b in batch])
        
    return {    
        'image':img,
        'caption':caption,
        'tag_label':tag_label,
    }

def build_loader(config):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

    dataset_train = build_dataset(config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build train dataset')

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)        
    print('train batch size: ', config.batch_size)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn, ### NOTEL THIS ###
        #shuffle=False,
    )
    return dataset_train, data_loader_train


def build_dataset(config):
    img_transform = build_img_transform(True, config.img_aug)
    text_transform = build_text_transform()
    split = 'train'

    image_reader = config[split].get('image_reader', {})
    dataset = ClipDataset(
        root_dir=config[split]['root_dir'],
        meta_file=config[split]['meta_file'],
        tag_file=config[split]['tag_file'],
        num_tags=config[split]['num_tags'],
        img_transform=img_transform,
        text_transform=text_transform,
        read_from=config[split]['read_from'],
        evaluator=None, # no evaluator for now
        image_reader_type=image_reader.get('type', 'pil'),
        fseek=config[split].get('fseek',False),
        split=split,
    )
    print('dataset len: ', len(dataset))
    return dataset


def build_img_transform(is_train, config):
    if not config.deit_aug:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.img_size, scale=config.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.img_size + 32),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        return transform

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != 'none' else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
            interpolation=config.interpolation,
        )
    else:
        size = int((256 / 224) * config.img_size)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=_pil_interp(config.interpolation)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    # if with_dc:
    #     transform = transforms.Compose([*transform.transforms, ToDataContainer()])

    return transform


def build_text_transform():
    transform = Tokenize()

    return transform


class Tokenize:
    """Wrapper class for CLIP tokenize function."""

    def __init__(self, max_seq_len=77, truncate=True):
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        result = tokenize(texts, self.max_seq_len, self.truncate)

        if expanded_dim:
            return result[0]

        return result
