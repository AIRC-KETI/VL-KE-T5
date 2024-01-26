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
import json
import logging
import math
import os
import random
from datetime import timedelta

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss

import transformers
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    ViTFeatureExtractor,
    SchedulerType,
    get_scheduler,
)

from data_utils import DatasetForVLAlign
from modeling_encoder import VisionT5MeanBiEncoder


logger = get_logger(__name__)

# epochs=3
# learning_rate=0.001
# scheduler_type=linear
# accelerate launch training_retriever_accelerate.py \
# --train_path ../downloaded_data/train-filtered.json \
# --validation_path ../downloaded_data/validation-filtered.json \
# --image_root_dir ../downloaded_data \
# --vision_model 'google/vit-base-patch16-384' \
# --language_model 'KETI-AIR/ke-t5-base' \
# --gradient_accumulation_steps 32 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --warmup_portion 0.01 \
# --learning_rate $learning_rate \
# --logging_steps 20 \
# --checkpointing_steps 10000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir vl_e${epochs}_${scheduler_type}_lr${learning_rate}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    # data
    # data
    parser.add_argument("--train_path",
                        default="data/vl_parallel/train_384_filtered.json", type=str)
    parser.add_argument("--validation_path",
                        default="data/vl_parallel/validation_384_filtered.json", type=str)
    parser.add_argument("--image_root_dir",
                        default=None, type=str)
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)
    
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
        default=8,
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
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.1,
                        type=float, help="label smoothing for cross entropy")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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

    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(days=10))]
    
    # Initialize the accelerator.
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir, kwargs_handlers=kwargs_handlers) if args.with_tracking else Accelerator(kwargs_handlers=kwargs_handlers)
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    

    # Load model and tokenizer
    model = VisionT5MeanBiEncoder(args)

    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=not args.use_slow_tokenizer)

    train_dataset = DatasetForVLAlign(
            file_path=args.train_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

        
    eval_dataset = DatasetForVLAlign(
            file_path=args.validation_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            image_root_dir=args.image_root_dir
        )

    collate_fn = eval_dataset.get_collate_fn()

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
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.warmup_portion > 0:
        args.num_warmup_steps = int(args.max_train_steps*max(min(args.warmup_portion, 1), 0))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            for _ in range(num_update_steps_per_epoch*starting_epoch):
                progress_bar.update(1)

    loss_fn = CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)
    def compute_loss(outputs):
        # outputs: language_repr, vision_repr- [batch_size, model_dim]

        batch_size = outputs["language_repr"].size(0)

        language_repr = nn.functional.normalize(outputs["language_repr"])
        vision_repr = nn.functional.normalize(outputs["vision_repr"])

        scores = torch.mm(language_repr, vision_repr.t())
        # scores(diagonal): [batch_size, batch_size]

        target = torch.arange(batch_size).to(language_repr.device)

        retrieve_loss = loss_fn(scores/args.logit_temperature, target) + loss_fn(scores.t()/args.logit_temperature, target)

        return retrieve_loss

    def retrieval_eval(outputs):
        with torch.no_grad():
            # outputs: language_repr, vision_repr- [batch_size, model_dim]

            batch_size = outputs["language_repr"].size(0)
            language_repr = nn.functional.normalize(outputs["language_repr"])
            vision_repr = nn.functional.normalize(outputs["vision_repr"])

            scores = torch.mm(language_repr, vision_repr.t())

            target = torch.arange(batch_size).to(outputs["language_repr"].device)

            # scores: [batch_size, batch_size]
            ranked = scores.argsort(dim=1, descending=True)
            # [[0.1, 0.3, -0.2, 0.14 ]] -> [[1, 3, 0, 2]] (index of score - descending order)
            idx2ranked_t = ranked.argsort(dim=1)

            # [[1, 3, 0, 2]] -> [[2, 0, 3, 1]] (index to rank)
            rrs = []
            for t, idx2ranked in zip(target, idx2ranked_t):
                rrs.append(1 / (idx2ranked[t].item() + 1))
            
            # reciprocal rank
            return torch.tensor(np.mean(rrs)).to(outputs["language_repr"].device)


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            total_mrr = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    progress_bar.update(1)
                    continue

            outputs = model(batch)
            loss = compute_loss(outputs)
            mrr = retrieval_eval(outputs)

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
                total_mrr += mrr.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
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
                logger.info("train_loss: {:.3f}, train_mrr: {:.3f}".format(total_loss.item()/step, total_mrr.item()/step))

            if completed_steps >= args.max_train_steps:
                break

        # evaluation proc
        model.eval()
        total_eval_loss = 0
        total_eval_mrr = 0

        for step, batch in enumerate(eval_dataloader):
            if args.max_validation_steps is not None and step >= args.max_validation_steps:
                break
            
            with torch.no_grad():
                outputs = model(batch)
                loss = compute_loss(outputs)
                mrr = retrieval_eval(outputs)

                total_eval_loss += accelerator.reduce(loss).detach().float()
                total_eval_mrr += accelerator.reduce(mrr).detach().float()

        logger.info("Evaluation - loss: {}, mrr: {}".format(
                total_eval_loss.item() / len(eval_dataloader),
                total_eval_mrr.item() / len(eval_dataloader),
            ))
        
        result = {}
        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["train_mrr"] = total_mrr.item() / len(train_dataloader)
            result["eval_loss"] = total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader)
            result["eval_mrr"] = total_eval_mrr.item() / accelerator.num_processes / len(eval_dataloader)
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
            text_tokenizer.save_pretrained(args.output_dir)
            image_tokenizer.save_pretrained(args.output_dir)
        
        if result is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(
                    {
                        "train_loss": result["train_loss"],
                        "train_mrr": result["train_mrr"],
                        "eval_loss": result["eval_loss"],
                        "eval_mrr": result["eval_mrr"],
                    },
                    f,
                )


if __name__ == "__main__":
    main()

