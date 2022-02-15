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
import copy
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

from index_scorer import FaissScorerExhaustiveGPU, FaissScorerExhaustiveMultiGPU, FaissScorer
from data_utils import DatasetForVLAlign
from modeling_encoder import (
    VisionT5SimpleBiEncoder,
    VisionT5MeanBiEncoder,
)

from training_retriever import (
    create_directory_info, 
    MODEL_CLS)

logger = logging.getLogger(__name__)

def batchify_no_collate(a, bsz):
    k = len(a)//bsz
    return [a[i*bsz:(i+1)*bsz] if i<k else a[i*bsz:] for i in range(k+1)]


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="../downloaded_data/train-filtered_wo_cc12m.json", type=str)
    parser.add_argument("--out_path",
                        default="../downloaded_data/train-filtered_wo_cc12m-hn.json", type=str)

    parser.add_argument("--tsv_path",
                        default="../downloaded_data/whole-filtered_wo_cc12m.tsv", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)
    parser.add_argument("--index_path",
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

    # resume
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
    
    parser.add_argument("--topk", default=20,
                        type=int, help="top k")

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=64,
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
                        default=[1,2,3], type=int)

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
    # faiss_scorer = FaissScorer(
    #     index_path=args.index_path,
    #     fvec_root=args.fvecs_dir,
    # )
    
    ref_data = [
            item for item in tqdm.tqdm(csv.DictReader(
                open(args.tsv_path, "r"), 
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

    
    # dataset
    dataset = json.load(open(args.data_path, "r"))
    dataset_batched = batchify_no_collate(dataset, args.batch_size)

    data_w_hard_negative = []

    with torch.no_grad():

        for batch in tqdm.tqdm(dataset_batched, desc="hardnegative..."):
            text_query = [ item["description"] for item in batch]
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
                        "score": float(s),
                        "path": ref_data[i]["path"],
                        "image_url": ref_data[i]["image_url"],
                    } for k, s, i in zip(range(args.topk), score, index)]
                result_list.append(result)

            for item, result in zip(batch, result_list):
                new_item = copy.deepcopy(item)
                new_item["hard_negative_img"] = result
                data_w_hard_negative.append(new_item)

    json.dump(data_w_hard_negative, open(args.out_path, "w"), indent=4)



if __name__ == "__main__":

    main()


# CUDA_VISIBLE_DEVICES="0,1" python create_hard_negative.py \
# --data_path ../downloaded_data/train-filtered_wo_cc12m.json \
# --out_path ../downloaded_data/train-filtered_wo_cc12m-hn.json \
# --tsv_path ../downloaded_data/whole-filtered_wo_cc12m.tsv \
# --fvecs_dir fvecs_whole-filtered_wo_cc12m \
# --hf_path ../hf_model \
# --model_gpu 0 \
# --scorer_gpus 1 \
# --batch_size 128

# CUDA_VISIBLE_DEVICES="2,3" python create_hard_negative.py \
# --data_path ../downloaded_data/validation-filtered_wo_cc12m.json \
# --out_path ../downloaded_data/validation-filtered_wo_cc12m-hn.json \
# --tsv_path ../downloaded_data/whole-filtered_wo_cc12m.tsv \
# --fvecs_dir fvecs_whole-filtered_wo_cc12m \
# --hf_path ../hf_model \
# --model_gpu 0 \
# --scorer_gpus 1 \
# --batch_size 128


# CUDA_VISIBLE_DEVICES="4,5,6,7" python create_hard_negative.py \
# --data_path ../downloaded_data/train-filtered.json \
# --out_path ../downloaded_data/train-filtered-hn.json \
# --tsv_path ../downloaded_data/whole-filtered.tsv \
# --fvecs_dir fvecs_whole-filtered \
# --hf_path ../hf_model \
# --model_gpu 0 \
# --scorer_gpus 1 2 3 \
# --batch_size 128
