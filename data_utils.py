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
import itertools
from typing import Optional

import numpy as np

from PIL import Image
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist

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
                    text_tokenizer: AutoTokenizer,
                    image_root_dir=None,
                    text_max_length=512,
                 ):
        super().__init__()
        self.file_path = file_path
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        self.image_root_dir=image_root_dir
        self.text_max_length = text_max_length

        logger.info("loading dataset...")
        self.data = json.load(open(file_path, "r"))
        logger.info("{} examples was loaded.".format(len(self.data)))

    def __getitem__(self, index):
        sample = self.data[index]

        path = sample["path"]
        if self.image_root_dir is not None:
            path = os.path.join(self.image_root_dir, path)
            
        description = sample["description"]

        image = Image.open(path)

        image_feature = self.image_tokenizer(images=image, return_tensors="pt")
        text_feature = self.text_tokenizer(description, return_tensors="pt", truncation=True, max_length=self.text_max_length)

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


class JsonLineDistributedDataset(IterableDataset):
    """
        Data root for JsonLineDistributedDataset:
            data_root   --- dataset_info.json
                        |-- train
                                |-- *.jsonl
                        |-- validation
                                |-- *.jsonl
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 buffer_size = 10000,
                 num_shard_per_rank: int = 1,
                 drop_last: bool = True,
                 deterministic: bool = True) -> None:
        super(JsonLineDistributedDataset, self).__init__()

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_shard_per_rank = num_shard_per_rank
        self.drop_last = drop_last
        self.deterministic = deterministic

        if self.deterministic:
            random.seed(self.seed + self.epoch)

        # shuffle buffer
        self.buffer_size = buffer_size

        # world size and rank
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_ex = 0

        # get pathes for dataset and number of example for each shard
        self.shard_path_range_pair = self.get_shard_path_range_pair_from_dataset_info(
            data_root, split)

        if len(self.shard_path_range_pair) % self.num_replicas != 0:
            logger.warning("It is recommended to set the num_of_shards%world_size value to 0. # of shards: {}, # of replacas(world_size): {}".format(
                len(self.shard_path_range_pair), self.num_replicas))

        self.shard_path_range_pair = self.preproc_shard_path_range_pair(
            self.shard_path_range_pair)

        self.path_range_pair_rank = self.shuffle_data()
        self.update_len()


    def get_shard_path_range_pair_from_dataset_info(self, data_root, split):
        dataset_info_path = os.path.join(data_root, "dataset_info.json")
        if os.path.isfile(dataset_info_path):
            dataset_info = json.load(open(dataset_info_path, "r"))
            splits = {x["name"]: x for x in dataset_info["splits"]}
            if split not in splits:
                shard_path_range_pair = self.get_shard_path_range_pair(
                    os.path.join(data_root, split))
                dataset_info["splits"].append(
                    {"name": split, "shardLengths": [e for path, (s, e) in shard_path_range_pair]})
                
                if self.rank == 0:
                    with open(dataset_info_path, "w") as f:
                        json.dump(dataset_info, f, indent=4)
                return shard_path_range_pair
            else:
                return list(
                    zip(
                        sorted(glob.glob(os.path.join(data_root, split, "*.jsonl"))), 
                        [(0, int(x)) for x in splits[split]["shardLengths"]]
                    )
                )
        else:
            logger.info("Rank {}: create infomation file for dataset...".format(self.rank))
            shard_path_range_pair = self.get_shard_path_range_pair(
                os.path.join(data_root, split))
            dataset_info = {"splits": []}
            dataset_info["splits"].append(
                {"name": split, "shardLengths": [e for path, (s, e) in shard_path_range_pair]})
            
            if self.rank == 0:
                with open(dataset_info_path, "w") as f:
                    json.dump(dataset_info, f, indent=4)
            return shard_path_range_pair

    def get_shard_path_range_pair(self, data_root):
        logger.info("Rank {}: get shard range pair from {}".format(self.rank, data_root))
        f_list = sorted(glob.glob(os.path.join(data_root, "*.jsonl")))
        return [(x, (0, len(open(x, 'r').readlines()))) for x in tqdm.tqdm(f_list, desc="get shard range pair...")]

    def conv2path_index_pair(self, path_range_pair):
        # path_range_pair: List[(path, (start_index, end_index))]
        path_index_pair = []
        for splp in path_range_pair:
            path_index_pair.extend([(splp[0], x)
                                   for x in range(splp[1][0], splp[1][1])])
        # path_index_pair: List[(path, index)]
        return path_index_pair

    def conv2path_range_pair(self, path_index_pair):
        # path_index_pair: List[(path, index)]
        path_range_pair = []

        prev_path = None
        start_idx = 0
        end_idx = 0

        for path, idx in path_index_pair:
            if prev_path is None:
                prev_path = path
                start_idx = idx
                end_idx = idx
            if prev_path != path:
                path_range_pair.append((prev_path, (start_idx, end_idx + 1)))
                prev_path = path
                start_idx = idx
            elif idx - end_idx > 1:
                path_range_pair.append((prev_path, (start_idx, end_idx + 1)))
                prev_path = path
                start_idx = idx
            end_idx = idx
        if (prev_path, (start_idx, end_idx + 1)) not in path_range_pair:
            path_range_pair.append((prev_path, (start_idx, end_idx + 1)))
        # path_range_pair: List[(path, (start_index, end_index))]
        return path_range_pair

    def split_list_into_n(self, ll, n):
        k, m = divmod(len(ll), n)
        return list((ll[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

    def preproc_shard_path_range_pair(self, path_range_pair):
        if len(path_range_pair) < self.num_replicas or len(path_range_pair) % self.num_replicas != 0:
            path_index_pair = self.conv2path_index_pair(path_range_pair)
            shard_path_index_pair = self.split_list_into_n(
                path_index_pair, self.num_replicas * self.num_shard_per_rank)
            # List[List[Tuple(str, Tuple(int, int))]]
            path_range_pair = [self.conv2path_range_pair(
                x) for x in shard_path_index_pair]
        else:
            path_range_pair = [[x] for x in path_range_pair]
        return path_range_pair

    def __iter__(self):
        
        buf = []
        for path, (start_idx, end_idx) in self.path_range_pair_rank[self.rank]:
            with open(path, "r") as f:
                for line in f.readlines()[start_idx: end_idx]:
                    item = json.loads(line)
                    x = self.proc(item)
                    if self.shuffle:
                        if len(buf) >= self.buffer_size:
                            idx = random.randint(0, self.buffer_size - 1)
                            yield buf[idx]
                            buf[idx] = x
                        else:
                            buf.append(x)
                    else:
                        yield x
        if self.shuffle:
            random.shuffle(buf)
            while buf:
                yield buf.pop()
    
    def proc(self, item):
        return item

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.path_range_pair_rank = self.shuffle_data()
        self.update_len()
    
    def __len__(self):
        return self.num_ex
    
    def update_len(self):
        num_ex = sum([end_idx-start_idx for path, (start_idx, end_idx) in self.path_range_pair_rank[self.rank]])
        self.num_ex = num_ex

    def shuffle_data(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(range(len(self.shard_path_range_pair)))

        else:
            indices = range(len(self.shard_path_range_pair))

        indices4rank = self.split_list_into_n(indices, self.num_replicas)
        path_range_pair_rank = [[self.shard_path_range_pair[x]
                                 for x in y] for y in indices4rank]
        path_range_pair_rank = [list(itertools.chain(*x))
                                for x in path_range_pair_rank]

        if self.drop_last:
            path_range_pair_rank = self.update_shard_to_be_equal(
                path_range_pair_rank)
        return path_range_pair_rank

    def update_shard_to_be_equal(self, path_range_pair_rank):
        num_ex_per_rank = [sum(
            [end_idx-start_idx for path, (start_idx, end_idx) in y]) for y in path_range_pair_rank]
        min_ex = min(num_ex_per_rank)
        drop_num = np.array(num_ex_per_rank) - min_ex

        return [self.drop_n_example(path_range_pair, num_drop) for path_range_pair, num_drop in zip(path_range_pair_rank, drop_num)]

    def drop_n_example(self, path_range_pair, num_drop):
        new_path_range_pair = []

        drop_t = num_drop

        for path, (start_idx, end_idx) in reversed(path_range_pair):
            num_ex = end_idx - start_idx
            n_drop = min(num_ex, drop_t)

            if n_drop > 0:
                drop_t -= n_drop

            if end_idx - start_idx - n_drop > 0:
                new_path_range_pair.append(
                    (path, (start_idx, end_idx - n_drop)))
        return list(reversed(new_path_range_pair))

    def get_collate_fn(self):
        def dynamic_collate(samples):
            if len(samples) == 0:
                return {}

            item = samples[0]

            return {
                k: [s[k] for s in samples] for k in item.keys()
            }
        return dynamic_collate


class DatasetForVLAlignHNDDP(JsonLineDistributedDataset):
    def __init__(
        self,
        image_tokenizer: ViTFeatureExtractor,
        text_tokenizer: AutoTokenizer,
        image_root: str,
        data_root: str,
        split: str = 'train',
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        buffer_size = 10000,
        num_shard_per_rank: int = 2,
        drop_last: bool = True,
        deterministic: bool = False,
        use_top1_hn: bool = False,
        text_max_length=320,
    ) -> None:
        super().__init__(
            data_root,
            split=split,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            buffer_size=buffer_size,
            num_shard_per_rank=num_shard_per_rank,
            drop_last=drop_last,
            deterministic=deterministic)

        self.image_root = image_root
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length

        self.use_top1_hn = use_top1_hn


    def proc(self, sample):
        path_pos = os.path.join(self.image_root, sample["path"])
        hard_negative = sample["hard_negative_img"]
        description = sample["description"]

        if self.use_top1_hn:
            path_negative = os.path.join(self.image_root, hard_negative[0]["path"])
        else:
            random.shuffle(hard_negative)
            path_negative = os.path.join(self.image_root, hard_negative[0]["path"])

        image_pos = Image.open(path_pos)
        image_neg = Image.open(path_negative)

        image_feature_pos = self.image_tokenizer(images=image_pos, return_tensors="pt")
        image_feature_neg = self.image_tokenizer(images=image_neg, return_tensors="pt")
        text_feature = self.text_tokenizer(description, return_tensors="pt", truncation=True, max_length=self.text_max_length)

        return {
            "pos": {
                "pixel_values": image_feature_pos["pixel_values"],
            },
            "neg": {
                "pixel_values": image_feature_neg["pixel_values"],
            },
            "input_ids": text_feature["input_ids"],
            "attention_mask": text_feature["attention_mask"],
        }


    def get_collate_fn(self):
        def collate_fn(samples, pad_id=0):
            if len(samples) == 0:
                return {}
            return {
                "input_ids": collate_tokens([s["input_ids"] for s in samples], pad_id),
                "attention_mask": collate_tokens([s["attention_mask"] for s in samples], 0),
                "pos": {
                    "pixel_values": default_collate([s["pos"]["pixel_values"][0] for s in samples]),
                },
                "neg": {
                    "pixel_values": default_collate([s["neg"]["pixel_values"][0] for s in samples]),
                },
            }
        return functools.partial(collate_fn, pad_id=self.text_tokenizer.pad_token_id)




if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    # def denormalize(image, mean, std):
    #     mean = torch.tensor(mean)
    #     std = torch.tensor(std)
    #     if image.ndim == 3 and image.shape[0] in [1, 3]:
    #         return image * std[:, None, None] + mean[:, None, None]
    #     else:
    #         return image * std + mean

    # train_path = "data/vl_parallel/train_384.json"
    # vision_model_path = 'google/vit-base-patch16-384'
    # text_model_path = 'KETI-AIR/ke-t5-base'

    # image_tokenizer = ViTFeatureExtractor.from_pretrained(vision_model_path)
    # text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    # dataset = DatasetForVLAlign(
    #         file_path=train_path,
    #         image_tokenizer=image_tokenizer,
    #         text_tokenizer=text_tokenizer
    #     )
    # collate_fn = dataset.get_collate_fn()

    # data_loader = DataLoader(
    #     dataset,
    #     shuffle=True,
    #     batch_size=16,
    #     num_workers=0,
    #     collate_fn=collate_fn
    # )

    # # loader perf test
    # for item in tqdm.tqdm(data_loader):
    #     pass

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


    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = dist.get_rank()

    image_root = "../downloaded_data"
    data_root = "../downloaded_data/filtered-wo_cc12m-hn"
    vision_model_path = 'google/vit-base-patch16-384'
    text_model_path = 'KETI-AIR/ke-t5-base'
    split = "train"

    image_tokenizer = ViTFeatureExtractor.from_pretrained(vision_model_path)
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    dataset = DatasetForVLAlignHNDDP(
            data_root=data_root,
            image_root=image_root,
            split=split,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer,
            shuffle=True,
            deterministic=True,
        )
    collate_fn = dataset.get_collate_fn()

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=collate_fn
    )

    import gc

    for item in tqdm.tqdm(data_loader):
        gc.collect()
        pass

    # for idx, item in zip(range(3), data_loader):
    #     if rank == 0:
    #         print(idx, text_tokenizer.decode(item["input_ids"][0]))

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 data_utils.py 