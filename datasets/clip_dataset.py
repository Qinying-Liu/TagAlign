# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

from re import L
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
from typing import List
from .base_dataset import BaseDataset
# from prototype.data.image_reader import build_image_reader
from .image_reader import build_image_reader
# import linklink as link
import random
import os
import omegaconf
import clip
from ipdb import set_trace
from .tokenizer import SimpleTokenizer
from .templates import full_imagenet_templates
from nltk.stem import WordNetLemmatizer
from PIL import Image
from io import BytesIO
from .zip_reader import ZipReader
from zipfile import ZipFile, BadZipFile
import multiprocessing
import csv
from pcache_fileio import fileio

lemmatizer = WordNetLemmatizer()

### frequently appeared 100 entities ###
TOP_CLASSES_1=[
    'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'kid', 'child', 'children', 'baby', 'student', 'bride', 'groom', 'couple', 'prince', 'princess', \
    'car', 'bus', 'truck', 'motorcycle', 'train', 'bicycle', 'boat', 'aeroplane', 'airplane', 'motorbike', 'bike',\
    'cup', 'bottle', 'bowl', 'knife', 'spoon',  'glass', 'fork',\
    'chair', 'table', 'bench', 'clock', 'laptop', 'light', 'vase', 'plant', 'remote', 'microwave', 'toaster', 'oven','mouse', 'keyboard','sofa', 'monitor','desk', 'tv','TV', 'couch', 'flower','refrigerator', \
    'house', 'building', 'hotel',\
    'handbag', 'umbrella','book', 'backpack', 'phone', 'shirt', 'tie', 'suitcase','T-shirt', 'bag',  'box', \
    'sink','bed','toilet',\
    'cat','dog',  'horse', 'bird','cow', 'sheep' ,'elephant', 'bear', 'zebra', 'giraffe', \
    'ball', 'racket', 'skateboard', 'skis', 'snowboard', 'surfboard', 'kite', \
    'pizza', 'cake', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'donut' ,\
]

### some of the entities are similar, map them to a single one ###
syn_dict = {
    'people':'people', 'man':'people', 'men':'people', 'woman':'people', 'women':'people', 'girl':'people', 'boy':'people', 'lady':'people', 'kid':'people', 'child':'people', 'children':'people', 'baby':'people', 'student':'people', 'bride':'people', 'groom':'people', 'couple':'people', 'prince':'people', 'princess':'people',\
    'airplane': 'aeroplane','motorbike': 'motorcycle','bike': 'bicycle',\
    'TV':'tv', 'desk': 'table', 'couch':'sofa',\
    'building': 'house', 'hotel': 'house', \
    'T-shirt': 'shirt','T-Shirt': 'shirt', 'handbag': 'bag', \
}

### unique entities ###
TOP_UNIQUE_CLASSES = [
    'people', 'car', 'bus', 'truck', 'motorcycle', \
    'train', 'bicycle', 'boat', 'aeroplane', 'cup', \
    'bottle', 'bowl', 'knife', 'spoon',  'glass', \
    'fork', 'chair', 'table', 'bench', 'clock', \
    'laptop', 'light', 'vase', 'plant', 'remote',\
    'microwave', 'toaster', 'oven','mouse', 'keyboard',\
    'sofa', 'monitor', 'tv', 'flower','refrigerator', \
    'house', 'bag', 'umbrella','book', 'backpack', \
    'phone', 'shirt', 'tie', 'suitcase', 'box',\
    'sink','bed','toilet', 'cat','dog', \
    'horse', 'bird','cow', 'sheep' ,'elephant', \
    'bear', 'zebra', 'giraffe',  'ball', 'racket', \
    'skateboard', 'skis', 'snowboard', 'surfboard', 'kite',\
    'pizza', 'cake', 'apple', 'banana', 'sandwich',\
    'orange', 'carrot', 'donut' ,\
]

TOP_UNIQUE_CLASSES_IDX = {}
for i, x in enumerate(TOP_UNIQUE_CLASSES):
    TOP_UNIQUE_CLASSES_IDX[x] = i

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

    def __init__(self, root_dir, meta_file, tag_file, num_tags=4585, img_transform=None, text_transform=None,
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

        with open(tag_file, 'r') as f:
            meta_tag = json.load(f) 
            data = {}
            for k, v in meta_tag.items():
                data[osp.basename(k)] = v
            self.meta_tag = data

        self.num_tags = num_tags

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
                print(each_meta_file)
                with open(each_meta_file) as f:
                    # lines = f.readlines()
                    csv_reader = csv.reader(f)
                    next(csv_reader)
                    for line in csv_reader:
                        filename = osp.join(rd, osp.basename(line[0]))
                        info = {'filename':filename, 'caption':line[1]}
                        self.metas.append(info)
                        self.num += 1

        #   ### read from local file and load all metafile info ###
        #    for rd, each_meta_file in zip(root_dir, meta_file):
        #         with open(each_meta_file) as f:
        #             lines = f.readlines()
        #         self.num += len(lines)

        #         for line in lines:
        #             info = json.loads(line)
        #             filename = osp.join(rd, info['filename'])
        #             info['filename'] = filename
        #             self.metas.append(info)

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

        # return self.reader.fetch_file(self.root_dir, filename)
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
        
    def _load_meta_class_dict(self, class_label_dir, sample_list_dir):
        # load class dict which is used to sample cross_image
        with open(sample_list_dir) as f:
            lines = f.readline()
            self.class_dict = json.loads(lines)

        # load class label for each sample    
        with open(class_label_dir) as f:
            lines = f.readline()
            self.class_label = json.loads(lines)


    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = curr_meta['filename']

        tag_label = self.meta_tag[os.path.basename(filename)]
        # tag_label = self.meta_tag[filename]
        tag_label_embedding = torch.zeros(self.num_tags, dtype=torch.float)
        if len(tag_label) > 0:
            tag_label_embedding[tag_label] = 1 

        caption = curr_meta['caption'] if 'caption' in curr_meta else ''
        ret_info = {}

        #############

        try:
            # assert self.is_contains_chinese(caption) == False
            while self.is_contains_chinese(caption):
                curr_meta = self._load_meta((idx + 1) % self.num)
                filename = curr_meta['filename']
                caption = curr_meta['caption'] if 'caption' in curr_meta else ''
                
            if self.read_from == 'dir':
                ### load from dir ###
                # if not osp.isfile(filename):
                #     filename = '/home/lqy/0000000.jpg'
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

            ret_info['tag_label'] = tag_label_embedding
            return ret_info    
                        
        except Exception as e:        
            print(e)
            # return self.__getitem__(0)
    
    # def judge_noun(self, n):
    #     n = n.replace('.', '')
    #     ans = n.split("'s")[0].split(',')[0]
    #     ### conduct Lemmatization ###
    #     # ans = nlp(ans)[0].lemma_
        
    #     if ans in syn_dict:
    #         ans = syn_dict[ans]
    #     elif len(ans) >= 2 and ans[-2:] == 'es' and ans[:-2] in syn_dict:
    #         ans = syn_dict[ans[:-2]]    
    #     elif len(ans) >= 1 and ans[-1] == 's' and ans[:-1] in syn_dict:
    #         ans = syn_dict[ans[:-1]]
    #     elif ans.lower() in syn_dict:
    #         ans = syn_dict[ans.lower()]
    #     elif len(ans) >= 2 and ans[-2:] == 'es' and ans.lower()[:-2] in syn_dict:
    #         ans = syn_dict[ans.lower()[:-2]]
    #     elif len(ans) >= 1 and ans[-1] == 's' and ans.lower()[:-1] in syn_dict:
    #         ans = syn_dict[ans.lower()[:-1]]

    #     if ans in TOP_UNIQUE_CLASSES:
    #         return 1, ans
    #     elif len(ans) >= 2 and ans[-2:] == 'es' and ans[:-2] in TOP_UNIQUE_CLASSES:
    #         return 1, ans[:-2]
    #     elif len(ans) >= 1 and ans[-1] == 's' and ans[:-1] in TOP_UNIQUE_CLASSES:
    #         return 1, ans[:-1]
    #     elif ans.lower() in TOP_UNIQUE_CLASSES:
    #         return 1, ans.lower()
    #     elif len(ans) >= 2 and ans.lower()[-2:] == 'es' and ans.lower()[:-2] in TOP_UNIQUE_CLASSES:
    #         return 1, ans.lower()[:-2]
    #     elif len(ans) >= 1 and ans.lower()[-1] == 's' and ans.lower()[:-1] in TOP_UNIQUE_CLASSES:
    #         return 1, ans.lower()[:-1]
    #     return 0, n
    
    def judge_noun(self, n):
        n = n.replace('.', '')
        # ans = n.split("'s")[0].split(',')[0]
        # ans = n.strip("'s").strip(",")
        ans = n
        ### conduct Lemmatization ###
        # ans = nlp(ans.lower())[0].lemma_
        ans = lemmatizer.lemmatize(ans.lower())
        
        if ans in syn_dict:
            ans = syn_dict[ans]
        
        if ans in TOP_UNIQUE_CLASSES:
            return 1, ans
        return 0, n       
    
    def build_question_and_answer(self, caption, nouns):
        words = caption.split(' ')
        question = ''
        ans_list = []

        token_mapper = {}
        word_mapper = {}
        assert self.mask_type == 'class'
        for word in words:
            word_after = word
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + newword + ' '
                ans_list.append(newword)
                token_id = self.tokenizer.encode(newword)[0]
                token_mapper[token_id] = TOP_UNIQUE_CLASSES_IDX[newword]
                word_mapper[token_id] = 332   ### this is 'M'
            else:
                question = question + word + ' '
                    
        question = question.replace("'", '').strip()
        raw_question = question
        
        question, _, _, _ = self.text_transform(raw_question)
        question = torch.tensor([word_mapper[int(word)] if int(word) in word_mapper else word for word in question])
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(ans_list)))
        answer, _, _, _ = self.text_transform(raw_answer)
        
        return raw_question, question, raw_answer, answer


    def build_question_and_answer_for_distilbert(self, caption, nouns):
        words = caption.split(' ')
        question = ''
        entity_list = []

        ### default, mask all entites ###
        assert self.mask_type == 'class'
        for word in words:
            word_after = word
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + '[MASK]' + ' '
                entity_list.append(newword)
            else:
                question = question + word + ' '
    
        question = question.replace("'", '').strip()
        raw_question = question
        #### build and transform answers ###
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(entity_list)))    
        return raw_question, None, raw_answer, None

    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def _get_label_text(self, text):
        # label_text = ['a photo of ' + text + '.']
        if self.label_texts_ensemble == 'prompt6':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt6'
        elif self.label_texts_ensemble == 'prompt8':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt8'
        elif self.label_texts_ensemble == 'prompt80':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt80'
        elif self.label_texts_ensemble == 'cc':
            return [text]
        else:
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt1'
        label_text = []
        with open(f) as fin:
            for line in fin.readlines():
                label_text.append(line.replace('{0}', text))
        return label_text

    def get_label_texts(self,):
        label_to_name = {}
        for curr_meta in self.metas:
            label = int(curr_meta['label']) if 'label' in curr_meta else None
            label_name = curr_meta['label_name'] if 'label_name' in curr_meta else None
            if label is not None and label_name is not None:
                label_to_name[label] = label_name
        labels = list(label_to_name.keys())
        labels.sort()

        label_texts = []
        label_text_len = []
        for label in labels:
            label_name = label_to_name[label]
            label_text = self._get_label_text(label_name)
            label_texts.extend(label_text)
            label_text_len.append(len(label_text))

        all_len = sum(label_text_len)
        offset = 0
        label_num = len(labels)
        label_texts_ensemble_matrix = torch.zeros(all_len, label_num)
        for lbl, ltl in enumerate(label_text_len):
            label_texts_ensemble_matrix[offset: offset + ltl, lbl] = 1
            offset += ltl

        return label_texts, label_texts_ensemble_matrix

    def dump(self, writer, output):
        filenames = output['filenames']
        image_ids = output['image_ids']
        label_names = output['label_names']
        captions = output['captions']
        tags = output['tags']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        labels = self.tensor2numpy(output['labels'])
        for _idx in range(len(filenames)):
            res = {
                'image_id': int(image_ids[_idx]),
                'filename': filenames[_idx],
                'label': int(labels[_idx]),
                'label_name': label_names[_idx],
                'caption': captions[_idx],
                'tag': tags[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]]
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()