# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import os
import csv
import time
import json
import shutil
import logging
import hashlib
import functools

import numpy as np
from numpy.core.numeric import indices
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import optim

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, ViTFeatureExtractor
from torch.utils.tensorboard import SummaryWriter

from index_scorer import FaissScorerExhaustiveGPU, FaissScorerExhaustiveMultiGPU
from data_utils import DatasetForImages
from modeling_encoder import (
    VisionT5SimpleBiEncoder,
    VisionT5MeanBiEncoder,
)

logger = logging.getLogger(__name__)


def create_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_directory_info(args, create_dir=True):

    model_dir = os.path.join(args.output_dir, "{}-{}-{}".format(
        args.model_cls.replace('/', '_'), 
        args.vision_model.replace('/', '_'), 
        args.language_model.replace('/', '_')))
    if args.dir_suffix is not None:
        model_dir = '_'.join([model_dir, args.dir_suffix])
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'model_dir': model_dir,
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            create_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info

def get_env_var(env_var, type_cls, default_val):
    if env_var in os.environ:
        return type_cls(os.environ[env_var])
    return default_val


def isin(tgt, pred_topk):
    for t in tgt:
        if t in pred_topk:
            return 1
    return 0

def topk_acc(targets, predictions, metric_meter):
    for topk in metric_meter.keys():
        for tgt, pred in zip(targets, predictions):
            pred_topk = pred[:topk]
            val = isin(tgt, pred_topk)
            metric_meter[topk].update(val)
    return metric_meter
            



MODEL_CLS = {
    "VisionT5SimpleBiEncoder": {
        "model_cls": VisionT5SimpleBiEncoder,
    },
    "VisionT5MeanBiEncoder": {
        "model_cls": VisionT5MeanBiEncoder,
    },
}



def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)

    parser.add_argument("--model_cls", default="VisionT5MeanBiEncoder", 
                        choices=["VisionT5SimpleBiEncoder", 
                                "VisionT5MeanBiEncoder"],
                        type=str, help="model class")
    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)
    parser.add_argument("--markdown_out",
                        default="md", type=str)

    # resume
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
    
    parser.add_argument("--topk", default=10,
                        type=int, help="top k")
    parser.add_argument("--image_size", default=180,
                        type=int, help="image size for html formatting")

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=16,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")

    # distributed setting
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size. (num_nodes*num_dev_per_node)")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")
    
    parser.add_argument('--model_gpu',
                        default=0, type=int)
    parser.add_argument('--scorer_gpus', nargs="+",
                        default=[1,2,3,4], type=int)

    args = parser.parse_args()

    
    path_info = create_directory_info(args, create_dir=False)

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")

    model_device = torch.device('cuda:{}'.format(args.model_gpu))

    faiss_scorer = FaissScorerExhaustiveMultiGPU(
            fvec_root=args.fvecs_dir,
            gpu_list=args.scorer_gpus
        )
    
    ref_data = [
            item for item in tqdm.tqdm(csv.DictReader(
                open(args.data_path, "r"), 
                delimiter="\t", 
                quoting=csv.QUOTE_MINIMAL, 
                fieldnames=['path', 'image_url']
            ), desc="loading item...")
        ]

    # get model class
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]

    # load model
    model = model_cls.from_pretrained(args.hf_path)
    model = model.to(model_device)

    # get tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    
    model.eval()

    markdown_out_dir = args.markdown_out
    if not os.path.isdir(markdown_out_dir):
        os.makedirs(markdown_out_dir, exist_ok=True)


    with torch.no_grad():
        text_query = [
            "심플한 원피스를 입은 젊은 여성",
            "원버튼 카멜브라운 캐시미어 롱코트",
        ]
        #text_feature = text_tokenizer(text_query, return_tensors="pt", truncation=True)
        text_feature = text_tokenizer(text_query, return_tensors="pt", truncation='longest_first', padding=True)
        
        q_vecs = model.encode_text({
            "input_ids":text_feature["input_ids"].to(model_device),
            "attention_mask":text_feature["attention_mask"].to(model_device),})
        q_vecs = q_vecs.cpu().numpy()

        scores, indice = faiss_scorer.get_topk(q_vecs, args.topk)

        result_list = []

        for t, score, index in zip(range(len(text_query)), scores, indice):
            result = [ {
                    "k": k+1,
                    "score": s,
                    "image_url": ref_data[i]["image_url"]
                } for k, s, i in zip(range(args.topk), score, index)]
            result_list.append(result)
        
        # results = [ {
        #     "k": t+1,
        #     "score": score,
        #     "image_url": ref_data[index]["image_url"]
        # } for t, score, index in zip(range(args.topk), scores[0], indice[0])]

        img_size=args.image_size

        for query, result in zip(text_query, result_list):
            print(f"query: {query}\nresults\n"+'-'*40)
            print(result)
            print('-'*40+'\n\n')

            md5_hash = hashlib.md5(query.encode("utf-8"))
            hash_str = md5_hash.hexdigest()
            markdown_path = os.path.join(markdown_out_dir, hash_str+".md")


            HTML_STR = f"""
<table>
    <tr>
    <td>Query</td>
    <td colspan="3">{query}</td>
</tr>
<tr>
    <td>Top 1</td>
    <td>Top 2</td>
    <td>Top 3</td>
    <td>Top 4</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[0]['image_url']}"
            alt="score: {result[0]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[1]['image_url']}"
            alt="score: {result[1]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[2]['image_url']}"
            alt="score: {result[2]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[3]['image_url']}"
            alt="score: {result[3]['score']:.2f}">
    </td>
</tr>
<tr>
    <td>Top 5</td>
    <td>Top 6</td>
    <td>Top 7</td>
    <td>Top 8</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[4]['image_url']}"
            alt="score: {result[4]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[5]['image_url']}"
            alt="score: {result[5]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[6]['image_url']}"
            alt="score: {result[6]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[7]['image_url']}"
            alt="score: {result[7]['score']:.2f}">
    </td>
</tr>
</table>
            """
            with open(markdown_path, "w") as f:
                f.write(HTML_STR)



if __name__ == "__main__":

    main()

