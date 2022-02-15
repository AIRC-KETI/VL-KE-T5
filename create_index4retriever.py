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
import time
import json
import shutil
import logging
import functools

import numpy as np
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


from data_utils import DatasetForImages
from modeling_encoder import (
    VisionT5SimpleBiEncoder,
    VisionT5MeanBiEncoder,
)

from training_retriever import (
    create_directory_info,
    get_env_var,
    MODEL_CLS,
)

from index_scorer import (
    write_fvecs,
    write_fvecs_append
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)
    parser.add_argument("--image_root_dir",
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
    
    parser.add_argument("--batch_write", action='store_true',
                        help="write fvecs for each iteration")

    args = parser.parse_args()
    

    args.local_rank = get_env_var('LOCAL_RANK', int, args.local_rank)
    args.local_world_size = get_env_var('LOCAL_WORLD_SIZE', int, args.local_world_size)
    args.rank = get_env_var('RANK', int, args.rank)
    args.world_size = get_env_var('WORLD_SIZE', int, args.world_size)

    # check world size
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    path_info = create_directory_info(args, create_dir=False)

    # if the world size is bigger than 1, init process group(sync)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")
    if args.distributed or args.local_rank == 0:
        if not os.path.isdir(args.fvecs_dir):
            os.makedirs(args.fvecs_dir, exist_ok=True)

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")
    
    # device
    device = torch.device('cuda')

    # get model class
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]

    # create model
    model = model_cls.from_pretrained(args.hf_path)
    logger.info("weights were loaded from pre_trained: {}".format(args.hf_path))
    model = model.cuda()

    # get tokenizer
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)


    logger.info("loading dataset: {}".format(args.data_path))
    # create dataset
    dataset = DatasetForImages(
        args.data_path,
        image_tokenizer=image_tokenizer,
        shard_idx=args.rank,
        num_shards=args.world_size,
        image_root_dir=args.image_root_dir)

    collate_fn = dataset.get_collate_fn()
    
    # create data loader
    data_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            sampler=None,
                            collate_fn=collate_fn)

    # wrap model using DDP
    model_ptr = model
    if args.distributed:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank)
        model_ptr = model.module


    # run

    target_path = os.path.join(args.fvecs_dir, "{:03d}.fvecs".format(args.rank))
    if os.path.isfile(target_path):
        os.remove(target_path)
    
    np_container = []
    with torch.no_grad():
        for example in tqdm.tqdm(data_loader, desc="conv..."):
            # fvecs: [bsz, model_dim]
            fvecs = model_ptr.encode_image({
                "pixel_values":example["pixel_values"].to(device)})
            fvecs_np = fvecs.cpu().numpy()    
    
            if args.batch_write:
                write_fvecs_append(target_path, fvecs_np)
            else:
                np_container.append(fvecs_np)
        
        if not args.batch_write:
            np_arr = np.concatenate(tuple(np_container), axis=0)
            print("np array({}) to file {}".format(np_arr.shape, target_path))
            write_fvecs(target_path, np_arr)


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    main()



# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 create_index4retriever.py \
# --hf_path default \
# --dir_suffix od1

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 create_index4retriever.py \
# --data_path ../downloaded_data/cc12m/cc12m_filtered_new.tsv \
# --image_root_dir ../downloaded_data/cc12m/images_384 \
# --fvecs_dir fvecs_cc12m_freeze_lm \
# --hf_path default \
# --dir_suffix freeze_lm \
# --batch_size 64 
