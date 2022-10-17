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
import gc
import time
import json
import shutil
import logging
import functools

from tqdm import tqdm
import numpy as np

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


from data_utils import DatasetForVLAlign
from modeling_encoder import (
    VisionT5SimpleBiEncoder,
    VisionT5MeanBiEncoder,
    VisionT5SimpleBiEncoderHN,
    VisionT5MeanBiEncoderHN,
)


logger = logging.getLogger(__name__)


def broadcast(tensors, rank=0):
    rt = tensors.clone().detach()
    torch.distributed.broadcast(rt, rank)
    return rt

def reduce_tensor(tensor, args):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def reduce_sum_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def all_gather(tensors, args, **kwargs):
    rt = tensors.clone().detach()
    tensor_list = [torch.zeros_like(rt) for _ in range(args.world_size)]
    torch.distributed.all_gather(tensor_list, rt)
    return tensor_list


def compute_loss(model, batch, loss_fn, args):
    outputs = model(batch)
    # outputs: language_repr, vision_repr- [batch_size, model_dim]

    batch_size = outputs["language_repr"].size(0)
    scores = torch.mm(outputs["language_repr"], outputs["vision_repr"].t())
    # scores(diagonal): [batch_size, batch_size]

    target = torch.arange(batch_size).to(outputs["language_repr"].device)

    retrieve_loss = loss_fn(scores/args.logit_temperature, target) + loss_fn(scores.t()/args.logit_temperature, target)

    return retrieve_loss


def compute_loss_over_device(model, batch, loss_fn, args):
    outputs = model(batch)
    # outputs: language_repr, vision_repr- [batch_size, model_dim]

    language_repr = outputs["language_repr"]
    vision_repr = outputs["vision_repr"]

    batch_size = language_repr.size(0)

    # blocking call (all_gather)
    with torch.no_grad():
        language_repr_gathered = all_gather(language_repr, args)
        vision_repr_gathered = all_gather(vision_repr, args)
        # language_repr_gathered, vision_repr_gathered - [world_size, batch_size, model_dim]

    language_repr_gathered[args.rank] = language_repr
    vision_repr_gathered[args.rank] = vision_repr

    language_repr_cat = torch.cat(language_repr_gathered, dim=0)
    vision_repr_cat = torch.cat(vision_repr_gathered, dim=0)
    # language_repr_cat, vision_repr_cat - [batch_size*world_size, model_dim]

    scores = torch.mm(language_repr_cat, vision_repr_cat.t())
    target = torch.arange(batch_size * args.world_size).to(language_repr.device)

    retrieve_loss = loss_fn(scores, target)

    return retrieve_loss



def retrieval_eval(model, batch):
    outputs = model(batch)
    # outputs: language_repr, vision_repr- [batch_size, model_dim]

    batch_size = outputs["language_repr"].size(0)
    scores = torch.mm(outputs["language_repr"], outputs["vision_repr"].t())

    target = torch.arange(batch_size).to(outputs["language_repr"].device)

    # scores: [batch_size, batch_size]
    ranked = scores.argsort(dim=1, descending=True)
    # [[0.1, 0.3, -0.2, 0.14 ]] -> [[1, 3, 0, 2]] (index of score - descending order)
    idx2ranked_t = ranked.argsort(dim=1)

    # [[1, 3, 0, 2]] -> [[2, 0, 3, 1]] (index to rank)
    rrs = []
    for t, idx2ranked in zip(target, idx2ranked_t):
        rrs.append(1 / (idx2ranked[t].item() + 1))
    
    # reciprocal rank for 1st, 2nd hop
    return {
        "mrr": torch.tensor(np.mean(rrs)).to(outputs["language_repr"].device)
        }


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def get_env_var(env_var, type_cls, default_val):
    if env_var in os.environ:
        return type_cls(os.environ[env_var])
    return default_val


MODEL_CLS = {
    "VisionT5SimpleBiEncoder": {
        "model_cls": VisionT5SimpleBiEncoder,
    },
    "VisionT5MeanBiEncoder": {
        "model_cls": VisionT5MeanBiEncoder,
    },
    "VisionT5SimpleBiEncoderHN": {
        "model_cls": VisionT5SimpleBiEncoderHN,
    },
    "VisionT5MeanBiEncoderHN": {
        "model_cls": VisionT5MeanBiEncoderHN,
    },
}


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path",
                        default="data/vl_parallel/train_384_filtered.json", type=str)
    parser.add_argument("--validation_path",
                        default="data/vl_parallel/validation_384_filtered.json", type=str)
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
    parser.add_argument("--resume", default=None, type=str,
                        help="path to checkpoint.")
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
                    

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=16,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")
    parser.add_argument("--print_freq", default=50,
                        type=int, help="print frequency")
    parser.add_argument("--global_steps", default=0,
                        type=int, help="variable for global steps")

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

    # default settings for training
    parser.add_argument("--epochs", default=3, type=int,
                        help="number of epochs for training")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--save_freq", default=1000,
                        type=int, help="steps to save checkpoint")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-4,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup", default=0.1,
                        type=float, help="warm-up proportion for linear scheduling")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.1,
                        type=float, help="label smoothing for cross entropy")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--off_scheduling", action='store_false',
                        help="off_scheduling")
    parser.add_argument("--max_validation_steps", default=1000, type=int,
                        help="max steps for validation")
    
    # ddp settings for sync
    parser.add_argument("--seed", default=0,
                        type=int, help="seed for torch manual seed")
    parser.add_argument("--deterministic", action='store_true',
                        help="deterministic")
    parser.add_argument("--save_every_epoch", action='store_true',
                        help="save check points on every epochs")
    parser.add_argument("--freeze_lm", action='store_true',
                        help="freeze language model")

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
    
    # create directory and summary logger
    best_score = 0
    summary_logger = None
    if args.local_rank == 0 or not args.distributed:
        path_info = create_directory_info(args)
        summary_logger = SummaryWriter(path_info["logs_dir"])
    path_info = create_directory_info(args, create_dir=False)

    # if the world size is bigger than 1, init process group(sync)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    
    # device
    device = torch.device('cuda')

    # deterministic seed
    if args.deterministic:
        torch.manual_seed(args.seed)
        data_seed = args.seed
    else:
        data_seed = torch.randint(9999, (1,), device=device, requires_grad=False)
        if args.distributed:
            data_seed = broadcast(data_seed)
        data_seed = data_seed.cpu().item()
        logger.info("[rank {}]seed for data: {}".format(args.rank if args.distributed else 0, data_seed))

    # update batch_size per a device
    args.batch_size = int(
        args.batch_size / args.gradient_accumulation_steps)

    # get model class
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]

    # create model
    model = model_cls(args)
    if args.freeze_lm:
        model.language_encoder.freeze_encoder()
    model = model.cuda()

    # get optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    # get tokenizer
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # create dataset
    train_dataset = DatasetForVLAlign(
            file_path=args.train_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

        
    validation_dataset = DatasetForVLAlign(
            file_path=args.validation_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

    collate_fn = validation_dataset.get_collate_fn()
    
    
    # create sampler for distributed data loading without redundant
    train_sampler = None
    validation_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            seed=data_seed)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset,
            seed=data_seed)

    # create data loader
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=args.num_workers,
                            sampler=train_sampler,
                            collate_fn=collate_fn)

    validation_loader = DataLoader(validation_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             sampler=validation_sampler,
                             collate_fn=collate_fn)
    
    
    # learning rate scheduler
    scheduler = None
    if args.off_scheduling:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            last_epoch=-1,
            steps_per_epoch=int(len(train_loader)/args.gradient_accumulation_steps),
            pct_start=args.warmup,
            anneal_strategy="linear"
        )


    # wrap model using DDP
    if args.distributed:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank)
    

    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and scheduler is not None:
                    if checkpoint['scheduler'] is not None:
                        scheduler.load_state_dict(checkpoint['scheduler'])

                args.start_epoch = checkpoint['epoch']
                if args.resume.endswith('-train'):
                    args.global_steps = checkpoint['global_step']
                    logger.info("=> global_steps '{}'".format(args.global_steps))
                    args.start_epoch-=1

                if args.local_rank == 0 or not args.distributed:
                    best_score = checkpoint['best_score'] if 'best_score' in checkpoint else 0
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.resume.lower()=='true':
                args.resume = path_info['ckpt_path']
                resume()
            elif args.resume.lower()=='best':
                args.resume = path_info['best_model_path']
                resume()
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    
    optimizer.param_groups[0]['capturable'] = True

    # save model as huggingface model
    if args.hf_path:
        if args.hf_path.lower()=='default':
            args.hf_path = os.path.join(path_info["model_dir"], "hf")

        if args.local_rank == 0 and args.distributed:
            model.module.save_pretrained(args.hf_path)
            logger.info('hf model is saved in {}'.format(args.hf_path))
        elif not args.distributed:
            model.save_pretrained(args.hf_path)
            logger.info('hf model is saved in {}'.format(args.hf_path))
        exit()

    
    # run training
    for epoch in range(args.start_epoch, args.epochs):
        # set epoch to train sampler 
        # for different order of example between epochs
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # training
        train(train_loader, model, optimizer, scheduler, epoch, args, path_info, summary_logger=summary_logger)
        args.global_steps = 0

        scores = validate(validation_loader, model, epoch, args)

        if args.local_rank == 0 or not args.distributed:
            curr_best = max(scores['mrr'], best_score)
            is_best = curr_best > best_score
            if is_best:
                best_score = curr_best
                best_result = {k: v for k, v in scores.items()}
                best_result["epoch"] = epoch
                with open(os.path.join(path_info["model_dir"], "best_score.json"), "w") as f:
                    json.dump(best_result, f, indent=4)

            ckpt_path = os.path.join(path_info["weights_dir"], "ckpt-{}.pth".format(epoch))  if args.save_every_epoch else path_info["ckpt_path"]

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
            }, is_best,
                ckpt_path,
                path_info["best_model_path"])

            summary_logger.add_scalar('eval/loss',
                                    scores['loss'], epoch)
            summary_logger.add_scalar('eval/mrr',
                                scores['mrr'], epoch)


def train(train_loader, model, optimizer, scheduler, epoch, args, path_info, summary_logger=None):
    loss_fn = CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

    # calc batch time
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    steps_per_epoch = len(train_loader)
    pass_n_steps = args.global_steps % steps_per_epoch
    logger.info("pass_n_steps: {}".format(pass_n_steps))

    # switch to train mode (for drop out)
    model.train()
    end = time.time()

    # zero grad
    optimizer.zero_grad()

    progress_bar = tqdm(range(steps_per_epoch), desc="[{}]".format(epoch), disable=args.local_rank != 0)

    for step_inbatch, batch in enumerate(train_loader):
        if step_inbatch < pass_n_steps:
            progress_bar.update(1)
            continue
        # compute loss
        loss = compute_loss(model, batch, loss_fn, args)
        # loss = compute_loss_over_device(model, batch, loss_fn, args)

        # backward pass            
        loss.backward()
        if (step_inbatch + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # schedule learning rate
            if scheduler is not None:
                scheduler.step()

        losses.update(loss.clone().detach().item())

        global_step = epoch*steps_per_epoch + step_inbatch
        if (global_step + 1) % args.print_freq == 0:
            
            with torch.no_grad():
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                mrr = retrieval_eval(model, batch)
                avg_mrr = reduce_tensor(mrr["mrr"], args)

                if args.local_rank == 0 or not args.distributed:

                    summary_logger.add_scalar('train/loss',
                                      losses.avg, global_step)
                    summary_logger.add_scalar('train/mrr',
                                      avg_mrr.item(), global_step)

                    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        losses.avg, avg_mrr.item()
                    )

                    logger.info('-----Training----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              args.batch_size/batch_time.val,
                              args.batch_size/batch_time.avg,
                              batch_time=batch_time) + score_log)

        
        if (global_step + 1) % args.save_freq == 0:
            if args.local_rank == 0 or not args.distributed:
                save_checkpoint({
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
                    }, False,
                        path_info["ckpt_path"]+"-train",
                        path_info["best_model_path"])


def validate(eval_loader, model, epoch, args):
    # loss function
    loss_fn = CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

    steps_per_epoch = len(eval_loader)

    max_validation_steps = steps_per_epoch
    if args.max_validation_steps > -1:
        max_validation_steps = min(steps_per_epoch, args.max_validation_steps)

    # score meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    mrr_meter = AverageMeter()    

    # switch to evaluate mode (for drop out)
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in zip(range(max_validation_steps), eval_loader):
            loss = compute_loss(model, batch, loss_fn, args)
            mrr = retrieval_eval(model, batch)

            avg_loss = reduce_tensor(loss, args)
            avg_mrr = reduce_tensor(mrr["mrr"], args)

            losses.update(avg_loss.item())
            mrr_meter.update(avg_mrr.item())

            if step_inbatch % args.print_freq == (args.print_freq - 1):
                gc.collect()

                batch_time.update((time.time() - end)/min(args.print_freq, step_inbatch + 1))
                end = time.time()

                if args.local_rank == 0 or not args.distributed:

                    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        losses.avg, mrr_meter.avg
                    )

                    logger.info('-----Evaluation----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              min(args.print_freq, step_inbatch + 1)/batch_time.val,
                              min(args.print_freq, step_inbatch + 1)/batch_time.avg,
                              batch_time=batch_time)+score_log)
    
    scores = {
        "loss": losses.avg,
        "mrr": mrr_meter.avg
    }
    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        scores["loss"], scores["mrr"]
                    )
    
    if args.local_rank == 0 or not args.distributed:
        logger.info('-----Evaluation----- \nEpoch: [{0}]\t'.format(
                                epoch)+score_log)

    return scores



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    main()



# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --train_path ../downloaded_data/whole-filtered_wo_cc12m.json \
# --validation_path ../downloaded_data/validation-filtered_wo_cc12m.json \
# --image_root_dir ../downloaded_data \
# --dir_suffix freeze_lm \
# --epochs 3 \
# --batch_size 32 \
# --freeze_lm


# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --train_path ../downloaded_data/train-filtered.json \
# --validation_path ../downloaded_data/validation-filtered.json \
# --image_root_dir ../downloaded_data \
# --dir_suffix cc12m_e16_lr1e3 \
# --epochs 16 \
# --batch_size 512 \
# --warmup 0.01 \
# --learning_rate 0.001 \
# --weight_decay 0.00001 \
# --save_every_epoch \
# --gradient_accumulation_steps 32


# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --train_path ../downloaded_data/train-filtered.json \
# --validation_path ../downloaded_data/validation-filtered.json \
# --image_root_dir ../downloaded_data \
# --dir_suffix cc12m_e16_lr1e4 \
# --epochs 16 \
# --batch_size 512 \
# --warmup 0.01 \
# --learning_rate 0.0001 \
# --weight_decay 0.00001 \
# --save_every_epoch \
# --gradient_accumulation_steps 32


# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --train_path ../downloaded_data/train-filtered.json \
# --validation_path ../downloaded_data/validation-filtered.json \
# --image_root_dir ../downloaded_data \
# --dir_suffix cc12m_e3_lr1e3 \
# --epochs 3 \
# --batch_size 512 \
# --warmup 0.01 \
# --learning_rate 0.001 \
# --weight_decay 0.00001 \
# --save_every_epoch \
# --gradient_accumulation_steps 32



# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --train_path ../downloaded_data/train-filtered.json \
# --validation_path ../downloaded_data/validation-filtered.json \
# --image_root_dir ../downloaded_data \
# --dir_suffix cc12m_e3_lr1e3 \
# --epochs 3 \
# --batch_size 512 \
# --warmup 0.01 \
# --learning_rate 0.001 \
# --weight_decay 0.00001 \
# --save_every_epoch \
# --gradient_accumulation_steps 32 \
# --resume output/VisionT5MeanBiEncoder-google_vit-base-patch16-384-KETI-AIR_ke-t5-base_cc12m_e3_lr1e3/weights/checkpoint.pth-train

