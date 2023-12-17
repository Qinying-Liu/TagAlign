from re import L
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
from typing import List
from .base_dataset import BaseDataset
from .image_reader import build_image_reader
import os
import omegaconf
import clip
from ipdb import set_trace
from .tokenizer import SimpleTokenizer
from .templates import full_imagenet_templates
from PIL import Image
from io import BytesIO
from .zip_reader import ZipReader
from zipfile import ZipFile, BadZipFile
import multiprocessing
import csv
# from pcache_fileio import fileio

class ClipDataset(BaseDataset):
    """
    Clip Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'
        - topnoun: 'none' / 'coco_top50' / 'cc3m_top50' / ...
    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG", "label": 0, "label_name": "dog"}\n"
    """

    def __init__(self, root_dir, meta_file, tag_file, num_tags=10000, img_transform=None, text_transform=None,
                 read_from='dir', evaluator=None, image_reader_type='pil',fseek=False, split='train'):
        if not isinstance(meta_file, List) and not isinstance(meta_file, omegaconf.listconfig.ListConfig):
            meta_file = [meta_file]
        if not isinstance(root_dir, List) and not isinstance(meta_file, omegaconf.listconfig.ListConfig):
            root_dir = [root_dir]

        self.meta_file = meta_file 
        self.root_dir = root_dir 
        self.read_from = read_from 
        
        if self.read_from =='zip':
            self.zip_dict = {}
            self._path = self.root_dir[0]
            self._zip_file = ZipFile(self._path)
  
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)

        self.fseek = fseek
        self.initialized = False
        self.num = 0
        self.split=split

        self.tokenizer = SimpleTokenizer()

        meta_tag = {}
        for rd, each_tag_file in zip(root_dir, tag_file):
            with open(each_tag_file, 'r') as f:
                each_meta_tag = json.load(f) 
                for filename, tag in each_meta_tag.items():
                    filepath = osp.join(rd, filename)
                    meta_tag[filepath] = tag
        self.meta_tag = meta_tag

        self.num_tags = num_tags
        self.class_samples = [[] for _ in range(num_tags)] 
        self.metas = []
        ### fseek uses file seek to load each line with pointer online ###
        ### this saves the memory while adding the loading time ###
        if self.fseek:
            self.line_offsets = []
            for each_meta_file in meta_file:
                line_offset = []
                offset = 0
                with open(each_meta_file) as f:
                    for line in f:
                        line_offset.append(offset)
                        offset += len(line.encode('UTF-8'))
                    f.close()
                self.num += len(line_offset)
                self.line_offsets.append(line_offset)
        else:
            ### read from local file and load all metafile info ###
            for rd, each_meta_file in zip(root_dir, meta_file):
                with open(each_meta_file) as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)
                    for line in csv_reader:
                        filename = osp.join(rd, line[0])
                        if filename in self.meta_tag:
                            for tag in self.meta_tag[filename]:
                                self.class_samples[tag].append(self.num)
                            info = {'filename':filename, 'caption':line[1]}
                            self.metas.append(info)
                            self.num += 1

        class_num = [len(per_class_samples) for per_class_samples in self.class_samples]
        class_freq = [per_class_num / self.num for per_class_num in class_num]
        self.class_freq = class_freq

        super(ClipDataset, self).__init__(root_dir=root_dir,
                                          meta_file=meta_file,
                                          read_from=read_from,
                                          transform=img_transform,
                                          evaluator=evaluator)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def fetch_file(self, filename):
        """Shortcut to reader's `fetch_file()`."""

        proc = multiprocessing.current_process()
        pid = proc.pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipReader()
            for rd in self.root_dir:
                self.zip_dict[pid].open(rd)
        zip_file = self.zip_dict[pid]

        return zip_file.fetch_file(os.path.dirname(filename), os.path.basename(filename))

    def __len__(self):        
        return self.num

    def _str2list(self, x):
        if type(x) is list:
            return x
        elif type(x) is str:
            return [x]
        else:
            raise RuntimeError(
                "unknown value for _str2list: {}".format(type(x)))

    def _load_meta(self, idx):
        if self.fseek:
            source_id = 0
            while idx >= len(self.line_offsets[source_id]):
                idx -= len(self.line_offsets[source_id])
                source_id += 1 #fixed
            with open(self.meta_file[source_id]) as f:
                f.seek(self.line_offsets[source_id][idx])
                line = f.readline()
                meta = json.loads(line)
                filename = osp.join(self.root_dir[source_id], meta['filename'])
                meta['filename'] = filename
                f.close()
            return meta
        else:
            return self.metas[idx]


    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = curr_meta['filename']

        tag_label = self.meta_tag[filename]
        tag_binarized_label = torch.zeros(self.num_tags, dtype=torch.float)
        if len(tag_label) > 0:
            tag_binarized_label[tag_label] = 1 

        caption = curr_meta['caption'] if 'caption' in curr_meta else ''
        ret_info = {}

        ############# ############# #############

        try:
            # assert self.is_contains_chinese(caption) == False
            while self.is_contains_chinese(caption):
                curr_meta = self._load_meta((idx + 1) % self.num)
                filename = curr_meta['filename']
                caption = curr_meta['caption'] if 'caption' in curr_meta else ''
                
            if self.read_from == 'dir':
                ### load from dir ###
                # if not osp.isfile(filename):
                #     filename = '/home/lqy/00000/000000000.jpg'
                img = Image.open(filename).convert('RGB')
            elif self.read_from == 'zip':
                proc = multiprocessing.current_process()
                pid = proc.pid # get pid of this process.
                if pid not in self.zip_dict:
                    self.zip_dict[pid] = ZipFile(self._path)
                zip_file = self.zip_dict[pid]
                img = Image.open(BytesIO(zip_file.read(os.path.basename(filename)))).convert('RGB')
                # img = Image.open(BytesIO(np.frombuffer(self.fetch_file(filename), dtype=np.uint8))).convert('RGB')
            else:
                ### load from bytes ###
                img_bytes = self.read_file(curr_meta)
                img = self.image_reader(img_bytes, filename)

            if self.img_transform is not None:
                image = self.img_transform(img)

            if self.text_transform is not None:
                caption = self.text_transform(caption)

            ret_info['image'] = image
            ret_info['caption'] = caption
            ret_info['tag_label'] = tag_binarized_label
            return ret_info    
                        
        except Exception as e:        
            print(e)
            # return self.__getitem__(0)
    
    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False