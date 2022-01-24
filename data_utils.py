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

import json
import os
import bz2
import functools
import glob
import tqdm
import random
import logging
import multiprocessing
import copy
import csv


from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, AutoTokenizer
from torch.utils.data._utils.collate import default_convert, default_collate


logger = logging.getLogger(__name__)


def collate_tokens(values, pad_idx):
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new_full((len(values), size), pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:len(v)])
    return res

def get_list(a, n, i):
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]


class DatasetForVLAlign(Dataset):
    def __init__(
                    self,
                    file_path: str,
                    image_tokenizer: ViTFeatureExtractor,
                    text_tokenizer: AutoTokenizer
                 ):
        super().__init__()
        self.file_path = file_path
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer

        logger.info("loading dataset...")
        self.data = json.load(open(file_path, "r"))
        logger.info("{} examples was loaded.".format(len(self.data)))

    def __getitem__(self, index):
        sample = self.data[index]

        path = sample["path"]
        description = sample["description"]

        image = Image.open(path)

        image_feature = self.image_tokenizer(images=image, return_tensors="pt")
        text_feature = self.text_tokenizer(description, return_tensors="pt", truncation=True)

        return {
            "pixel_values": image_feature["pixel_values"],
            "input_ids": text_feature["input_ids"],
            "attention_mask": text_feature["attention_mask"],
        }

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        def collate_fn(samples, pad_id=0):
            if len(samples) == 0:
                return {}
            return {
                "input_ids": collate_tokens([s["input_ids"] for s in samples], pad_id),
                "attention_mask": collate_tokens([s["attention_mask"] for s in samples], 0),
                "pixel_values":  default_collate([s["pixel_values"][0] for s in samples])
            }
        return functools.partial(collate_fn, pad_id=self.text_tokenizer.pad_token_id)


class DatasetForImages(Dataset):
    def __init__(
                    self,
                    file_path: str,
                    image_tokenizer: ViTFeatureExtractor,
                    shard_idx: int=0,
                    num_shards: int=1,
                    image_root_dir=None,
                 ):
        super().__init__()
        self.file_path = file_path
        self.image_tokenizer = image_tokenizer
        self.image_root_dir=image_root_dir

        logger.info("loading dataset...")

        self.data = [
            item for item in csv.DictReader(
                open(file_path, "r"), 
                delimiter="\t", 
                quoting=csv.QUOTE_NONE, 
                fieldnames=['path', 'image_url']
            )
        ]

        self.shard_idx = shard_idx
        if num_shards > 1:
            self.data = get_list(self.data, num_shards, shard_idx)

        logger.info("{} examples was loaded.".format(len(self.data)))

    def __getitem__(self, index):
        sample = self.data[index]

        path = sample["path"]
        if self.image_root_dir is not None:
            path = os.path.join(self.image_root_dir, path)

        image = Image.open(path).convert("RGB")

        image_feature = self.image_tokenizer(images=image, return_tensors="pt")

        return {
            "pixel_values": image_feature["pixel_values"],
        }

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        def collate_fn(samples):
            if len(samples) == 0:
                return {}
            return {
                "pixel_values":  default_collate([s["pixel_values"][0] for s in samples])
            }
        return collate_fn



if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    def denormalize(image, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return image * std[:, None, None] + mean[:, None, None]
        else:
            return image * std + mean

    train_path = "data/vl_parallel/train_384.json"
    vision_model_path = 'google/vit-base-patch16-384'
    text_model_path = 'KETI-AIR/ke-t5-base'

    image_tokenizer = ViTFeatureExtractor.from_pretrained(vision_model_path)
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    dataset = DatasetForVLAlign(
            file_path=train_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer
        )
    collate_fn = dataset.get_collate_fn()

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=16,
        num_workers=0,
        collate_fn=collate_fn
    )

    # loader perf test
    for item in tqdm.tqdm(data_loader):
        pass

    # for idx, item in zip(range(16), data_loader):

    #     # data loader
    #     print(idx, text_tokenizer.decode(item["input_ids"][0]))
    #     pixel_values = item["pixel_values"][0]


    #     image_denorm = denormalize(
    #         pixel_values, 
    #         image_tokenizer.image_mean, 
    #         image_tokenizer.image_std)
    #     image = image_tokenizer.to_pil_image(image_denorm)
    #     image.save(os.path.join("tmp", "{}.jpg".format(idx)))


