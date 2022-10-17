#!/usr/bin/env python
# coding=utf-8

# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
from curses import raw
import json
import logging
import math
import os
import random
from datetime import timedelta
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedDataParallelKwargs
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    ViTFeatureExtractor,
    SchedulerType,
    get_scheduler,
    default_data_collator,
)

import datasets
from datasets import load_dataset

from data_utils import DatasetForVLAlign
from modeling_veldt5 import VELDT5Model


logger = get_logger(__name__)

# epochs=1
# learning_rate=0.001
# scheduler_type=linear
# accelerate launch training_veldt5_accelerate.py \
# --vision_model 'google/vit-base-patch16-384' \
# --language_model 'KETI-AIR/ke-t5-base' \
# --gradient_accumulation_steps 32 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --warmup_portion 0.02 \
# --logging_steps 20 \
# --checkpointing_steps 10000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir veld_e${epochs}_${scheduler_type}


# accelerate launch training_veldt5_accelerate.py \
#     --max_train_steps_per_epoch 100 \
#     --max_validation_steps 20 \
#     --logging_steps 5 \
#     --with_tracking \
#     --output_dir test


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    # data
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name_lm",
        type=str,
        default="sent_dataset.py",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name_lm",
        type=str,
        default="base",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="../huggingface_datasets",
        help="The path to cache directory for huggingface datasets.",
    )
    parser.add_argument(
        "--hf_data_dir_lm",
        type=str,
        default="../sent_eq_4k_25/*/",
        help="The path to data directory for huggingface datasets.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=1,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=256,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    
    parser.add_argument("--train_path",
                        default="../../downloaded_data/train-filtered.json", type=str)
    parser.add_argument("--validation_path",
                        default="../../downloaded_data/validation-filtered.json", type=str)
    parser.add_argument("--image_root_dir",
                        default="../../downloaded_data", type=str)

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="image_text_pair_datasets.py",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="base",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_data_dir",
        type=str,
        default="../../downloaded_data",
        help="The path to data directory for huggingface datasets.",
    )



    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    # training
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--contrastive_weight", default=1.0,
                        type=float, help="The weighting value for contrastive loss")
    parser.add_argument("--captioning_weight", default=2.0,
                        type=float, help="The weighting value for captioning loss")
    parser.add_argument("--lm_weight", default=1.0,
                        type=float, help="The weighting value for lm loss")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.0,
                        type=float, help="label smoothing for cross entropy")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Total number of validation steps to perform.",
    )
    parser.add_argument(
        "--max_train_steps_per_epoch",
        type=int,
        default=None,
        help="The number of training steps to perform on a epoch. (for debugging)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_portion", type=float, default=0, help="Portion of total training steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )


    # logging
    parser.add_argument(
        "--logging_steps", type=int, default=0, help="Number of steps for logging (stdout)."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(days=10)),
        DistributedDataParallelKwargs(find_unused_parameters=True)
        ]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        kwargs_handlers=kwargs_handlers , **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    

    # Load model and tokenizer
    model = VELDT5Model.from_encoder_decoder_pretrained(
        args.vision_model, 
        args.language_model
    )
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=not args.use_slow_tokenizer)


    # load lm datasets
    with accelerator.main_process_first():
        raw_datasets = load_dataset(
            args.dataset_name_lm, 
            args.dataset_config_name_lm,
            cache_dir=args.hf_cache_dir, 
            data_dir=args.hf_data_dir_lm,
            )
    
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_lm_dataset = lm_datasets["train"]
    # eval_lm_dataset = lm_datasets["validation"]
    train_lm_dataloader = DataLoader(
        train_lm_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    # eval_lm_dataloader = DataLoader(eval_lm_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)



    # load image text pair datasets
    with accelerator.main_process_first():
        image_text_datasets = load_dataset(
                args.dataset_name, 
                args.dataset_config_name,
                cache_dir=args.hf_cache_dir, 
                data_dir=args.hf_data_dir,
            )
    train_dataset = image_text_datasets["train"]
    eval_dataset = image_text_datasets["validation"]
    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        image_list = [s["image"] for s in samples]
        image_feature = image_tokenizer(images=image_list, return_tensors="pt")
        text_feature = tokenizer([s["description"] for s in samples], return_tensors="pt", padding=True, truncation='longest_first')
        return {
            "pixel_values": image_feature["pixel_values"],
            "input_ids": text_feature["input_ids"],
            "attention_mask": text_feature["attention_mask"],
        }


    # train_dataset = DatasetForVLAlign(
    #         file_path=args.train_path,
    #         image_tokenizer=image_tokenizer,
    #         text_tokenizer=tokenizer,
    #         image_root_dir=args.image_root_dir
    #     )
    # eval_dataset = DatasetForVLAlign(
    #         file_path=args.validation_path,
    #         image_tokenizer=image_tokenizer,
    #         text_tokenizer=tokenizer,
    #         image_root_dir=args.image_root_dir
    #     )
    # collate_fn = eval_dataset.get_collate_fn()



    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps_per_epoch is not None:
        num_update_steps_per_epoch = min(args.max_train_steps_per_epoch, num_update_steps_per_epoch)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.warmup_portion > 0:
        args.num_warmup_steps = int(args.max_train_steps/max(min(args.warmup_portion, 1), 0))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, train_lm_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, train_lm_dataloader, lr_scheduler
    )
    # model, optimizer, train_dataloader, eval_dataloader, train_lm_dataloader, eval_lm_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, train_lm_dataloader, eval_lm_dataloader, lr_scheduler
    # )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps_per_epoch is not None:
        num_update_steps_per_epoch = min(args.max_train_steps_per_epoch, num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("vl_align_no_trainer", experiment_config)


    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            for _ in range(num_update_steps_per_epoch*starting_epoch):
                progress_bar.update(1)
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            for _ in range(num_update_steps_per_epoch*starting_epoch):
                progress_bar.update(1)

    # calc mrr
    def calc_mrr(encoder_logits, decoder_logits):
        with torch.no_grad():
            # outputs: language_repr, vision_repr- [batch_size, model_dim]
            encoder_logits = nn.functional.normalize(encoder_logits)
            decoder_logits = nn.functional.normalize(decoder_logits)

            batch_size = encoder_logits.size(0)
            scores = torch.mm(decoder_logits, encoder_logits.t())
            target = torch.arange(batch_size).to(decoder_logits.device)

            # scores: [batch_size, batch_size]
            ranked = scores.argsort(dim=1, descending=True)
            # [[0.1, 0.3, -0.2, 0.14 ]] -> [[1, 3, 0, 2]] (index of score - descending order)
            idx2ranked_t = ranked.argsort(dim=1)

            # [[1, 3, 0, 2]] -> [[2, 0, 3, 1]] (index to rank)
            rrs = []
            for t, idx2ranked in zip(target, idx2ranked_t):
                rrs.append(1 / (idx2ranked[t].item() + 1))
            
            # reciprocal rank
            return torch.tensor(np.mean(rrs)).to(decoder_logits.device)


    train_lm_dataloader_iterator = iter(train_lm_dataloader)
    # eval_lm_dataloader_iterator = iter(eval_lm_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            total_mrr = 0
            total_lm_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if (resume_step is not None 
                        and step < resume_step
                        and step % args.gradient_accumulation_steps == 0):
                    completed_steps += 1
                    progress_bar.update(1)
                    continue
            
            if args.max_train_steps_per_epoch is not None and (step//args.gradient_accumulation_steps) >= num_update_steps_per_epoch:
                break

            outputs = model(
                pixel_values=batch["pixel_values"], 
                labels=batch["input_ids"], 
                return_contrastive_loss=True,
                decoder_attention_mask=batch["attention_mask"], 
            )
            loss = args.captioning_weight*outputs.loss + args.contrastive_weight*outputs.c_loss
            mrr = calc_mrr(outputs.e_logits_g, outputs.d_logits)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)


            try:
                batch2 = next(train_lm_dataloader_iterator)
            except StopIteration:
                train_lm_dataloader_iterator = iter(train_lm_dataloader)
                batch2 = next(train_lm_dataloader_iterator)
            lm_outputs = model(
                labels=batch2["input_ids"], 
                return_contrastive_loss=True,
                decoder_attention_mask=batch2["attention_mask"], 
            )
            lm_loss = lm_outputs.loss
            lm_loss = lm_loss / args.gradient_accumulation_steps
            accelerator.backward(lm_loss)

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
                total_mrr += mrr.detach().float()
                total_lm_loss += lm_loss.detach().float()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and step % args.gradient_accumulation_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if (args.logging_steps>0 and completed_steps % args.logging_steps == 0 and completed_steps > 0) and step % args.gradient_accumulation_steps == 0:
                logger.info(
                    "train_loss: {:.3f}, train_mrr: {:.3f}, total_lm_loss: {:.3f}".format(
                        total_loss.item()/step, 
                        total_mrr.item()/step, 
                        total_lm_loss.item()/step
                        )
                    )

            if completed_steps >= args.max_train_steps:
                break

        # evaluation proc
        model.eval()
        total_eval_loss = 0
        total_eval_mrr = 0
        total_eval_lm_loss = 0

        for step, batch in enumerate(eval_dataloader):
            if args.max_validation_steps is not None and step >= args.max_validation_steps:
                break
            
            with torch.no_grad():
                outputs = model(
                    pixel_values=batch["pixel_values"], 
                    labels=batch["input_ids"], 
                    return_contrastive_loss=True,
                    decoder_attention_mask=batch["attention_mask"], 
                )
                loss = args.captioning_weight*outputs.loss + args.contrastive_weight*outputs.c_loss
                mrr = calc_mrr(outputs.e_logits_g, outputs.d_logits)

                total_eval_loss += accelerator.reduce(loss).detach().float()
                total_eval_mrr += accelerator.reduce(mrr).detach().float()

                try:
                    batch2 = next(train_lm_dataloader_iterator)
                except StopIteration:
                    train_lm_dataloader_iterator = iter(train_lm_dataloader)
                    batch2 = next(train_lm_dataloader_iterator)
                lm_outputs = model(
                    labels=batch2["input_ids"], 
                    return_contrastive_loss=True,
                    decoder_attention_mask=batch2["attention_mask"], 
                )
                lm_loss = lm_outputs.loss
                total_eval_lm_loss += accelerator.reduce(lm_loss).detach().float()

        logger.info("Evaluation - loss: {}, mrr: {}, lm_loss".format(
                total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader),
                total_eval_mrr.item() / accelerator.num_processes / len(eval_dataloader),
                total_eval_lm_loss.item() / accelerator.num_processes / len(eval_dataloader),
            ))
        
        result = {}
        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["train_mrr"] = total_mrr.item() / len(train_dataloader)
            result["train_lm_loss"] = total_lm_loss.item() / len(train_dataloader)
            result["eval_loss"] = total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader)
            result["eval_mrr"] = total_eval_mrr.item() / accelerator.num_processes / len(eval_dataloader)
            result["total_eval_lm_loss"] = total_eval_lm_loss.item() / accelerator.num_processes / len(eval_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            image_tokenizer.save_pretrained(args.output_dir)
        
        if result is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(
                    {
                        "train_loss": result["train_loss"],
                        "train_mrr": result["train_mrr"],
                        "train_lm_loss": result["train_lm_loss"],
                        "eval_loss": result["eval_loss"],
                        "eval_mrr": result["eval_mrr"],
                        "total_eval_lm_loss": result["total_eval_lm_loss"],
                    },
                    f,
                )


if __name__ == "__main__":
    main()

