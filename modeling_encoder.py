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


from argparse import ArgumentError
import os
import copy
from typing import Callable, Dict, Type
import importlib

from transformers import AutoModel, T5EncoderModel, T5Config, ViTModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

import torch.nn as nn
import torch
import torch.nn.functional as F


import logging

logger = logging.getLogger(__name__)

def simple_pooling(hidden_states, mask=None):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq]
    first_token_tensor = hidden_states[:, :1]
    
    # pooled_output: [batch_size, 1, model_dim]
    return first_token_tensor.squeeze(1)


def mean_pooling(hidden_states, mask=None, sqrt=True):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq]
    
    if mask is None:
        batch_size, seq_length = hidden_states.shape[:2]
        mask = torch.ones(batch_size, seq_length, device=hidden_states.device, dtype=hidden_states.dtype)

    sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).permute(0, 2, 1)
    # sentence_sums: [batch_size, 1, model_dim]
    divisor = mask.sum(dim=1).view(-1, 1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    # pooled_output: [batch_size, 1, model_dim]
    return sentence_sums.squeeze(1)


class SimplePooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        pooled_output = simple_pooling(hidden_states, mask)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, model_dim]
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask=None, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq]
        pooled_output = mean_pooling(hidden_states, mask, sqrt=sqrt)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, model_dim]
        return pooled_output


class MeanReducer(nn.Module):
    def __init__(self, hidden_size, repr_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, repr_size)
        self.layernorm = nn.LayerNorm(repr_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, repr_size]
        # attention masks: [batch_size, seq]
        pooled_output = mean_pooling(hidden_states, mask, sqrt=sqrt)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, repr_size]
        return pooled_output


class T5EncoderSimple(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderSimple, self).__init__(config)

        self.pooler = SimplePooler(config.d_model)

    def freeze_encoder(self):
        for block_e in self.encoder.block:
            for param_e in block_e.parameters():
                param_e.requires_grad = False
        
        for param_e in self.encoder.final_layer_norm.parameters():
            param_e.requires_grad = False
        
        for param_e in self.shared.parameters():
            param_e.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = self.pooler(hidden_states, attention_mask)

        all_hidden_states = outputs[1] if output_hidden_states else None
        all_attentions = outputs[2] if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class T5EncoderMean(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderMean, self).__init__(config)

        self.pooler = MeanPooler(config.d_model)

    def freeze_encoder(self):
        for block_e in self.encoder.block:
            for param_e in block_e.parameters():
                param_e.requires_grad = False
        
        for param_e in self.encoder.final_layer_norm.parameters():
            param_e.requires_grad = False
        
        for param_e in self.shared.parameters():
            param_e.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = self.pooler(hidden_states, attention_mask)

        all_hidden_states = outputs[1] if output_hidden_states else None
        all_attentions = outputs[2] if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )







# =========================================================================
# Bi encoder
# =========================================================================

class BiEncoderBase(nn.Module):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(BiEncoderBase, self).__init__()

        if args is not None:
            self.load_weight_from_args(args)

        elif vision_encoder is not None and language_encoder is not None:
            self.vision_encoder = vision_encoder
            self.language_encoder = language_encoder

        else:
            raise ArgumentError("You must pass the args or (vision_encoder, language_encoder) as arguments.")
        
    
    def load_weight_from_args(self, args):
        raise NotImplementedError
    
    def encode_text(self, inputs):
        outputs = self.language_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs[1]
    
    def encode_image(self, inputs):
        outputs = self.vision_encoder(
            pixel_values=inputs["pixel_values"],
        )
        return outputs[1]

    def save_pretrained(self, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "vision")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        self.vision_encoder.save_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "language")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        self.language_encoder.save_pretrained(*tuple(args_k), **kwargs)


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError


    def forward(self, batch):
        language_repr = self.encode_text(batch)
        vision_repr = self.encode_image(batch)
        
        return {'language_repr': language_repr, 'vision_repr': vision_repr}


class VisionT5SimpleBiEncoder(BiEncoderBase):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(VisionT5SimpleBiEncoder, self).__init__(
            args=args,
            vision_encoder=vision_encoder,
            language_encoder=language_encoder
        )

    def load_weight_from_args(self, args):
        self.vision_encoder = ViTModel.from_pretrained(args.vision_model)
        self.language_encoder = T5EncoderSimple.from_pretrained(args.language_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "vision")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        vision_encoder = ViTModel.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "language")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        language_encoder = T5EncoderSimple.from_pretrained(*tuple(args_k), **kwargs)

        return cls(vision_encoder=vision_encoder, language_encoder=language_encoder)


class VisionT5MeanBiEncoder(BiEncoderBase):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(VisionT5MeanBiEncoder, self).__init__(
            args=args,
            vision_encoder=vision_encoder,
            language_encoder=language_encoder
        )

    def load_weight_from_args(self, args):
        self.vision_encoder = ViTModel.from_pretrained(args.vision_model)
        self.language_encoder = T5EncoderMean.from_pretrained(args.language_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "vision")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        vision_encoder = ViTModel.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "language")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        language_encoder = T5EncoderMean.from_pretrained(*tuple(args_k), **kwargs)

        return cls(vision_encoder=vision_encoder, language_encoder=language_encoder)


class BiEncoderBaseHN(BiEncoderBase):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(BiEncoderBaseHN, self).__init__(
            args=args,
            vision_encoder=vision_encoder,
            language_encoder=language_encoder
        )

    def forward(self, batch):
        language_repr = self.encode_text(batch)
        vision_repr_pos = self.encode_image(batch["pos"])
        vision_repr_neg = self.encode_image(batch["neg"])
        
        return {
            'language_repr': language_repr, 
            'vision_repr_pos': vision_repr_pos,
            'vision_repr_neg': vision_repr_neg,
        }


class VisionT5SimpleBiEncoderHN(BiEncoderBaseHN):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(VisionT5SimpleBiEncoderHN, self).__init__(
            args=args,
            vision_encoder=vision_encoder,
            language_encoder=language_encoder
        )

    def load_weight_from_args(self, args):
        self.vision_encoder = ViTModel.from_pretrained(args.vision_model)
        self.language_encoder = T5EncoderSimple.from_pretrained(args.language_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "vision")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        vision_encoder = ViTModel.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "language")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        language_encoder = T5EncoderSimple.from_pretrained(*tuple(args_k), **kwargs)

        return cls(vision_encoder=vision_encoder, language_encoder=language_encoder)


class VisionT5MeanBiEncoderHN(BiEncoderBaseHN):
    _ENCODER_TYPE='biencoder'

    def __init__(self,
                args=None,
                vision_encoder=None,
                language_encoder=None):
        super(VisionT5MeanBiEncoderHN, self).__init__(
            args=args,
            vision_encoder=vision_encoder,
            language_encoder=language_encoder
        )

    def load_weight_from_args(self, args):
        self.vision_encoder = ViTModel.from_pretrained(args.vision_model)
        self.language_encoder = T5EncoderMean.from_pretrained(args.language_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "vision")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        vision_encoder = ViTModel.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "language")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        language_encoder = T5EncoderMean.from_pretrained(*tuple(args_k), **kwargs)

        return cls(vision_encoder=vision_encoder, language_encoder=language_encoder)



if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    import argparse
    from transformers import ViTFeatureExtractor, AutoTokenizer
    from data_utils import DatasetForVLAlign
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--train_path",
                        default="data/vl_parallel/train_384.json", type=str)
    parser.add_argument("--validation_path",
                        default="data/vl_parallel/validation_384.json", type=str)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)
    
    # training configurations
    parser.add_argument("--batch_size",
                        default=16, type=int)
    parser.add_argument("--num_workers",
                        default=1, type=int)
    args = parser.parse_args()

    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    dataset = DatasetForVLAlign(
            file_path=args.train_path,
            image_tokenizer=image_tokenizer,
            text_tokenizer=text_tokenizer
        )
    collate_fn = dataset.get_collate_fn()

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    model = VisionT5MeanBiEncoder(args)

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            print(outputs["language_repr"].size())
            print(outputs["vision_repr"].size())
            exit()










