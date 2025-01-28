"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig


from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)

class BLIP2Cir(Blip2Base):
    def __init__(
            self,
            loss: Any,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp32",
            train_vit=False,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=35,
            pooling = "max",
        ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()  
        self.tokenizer = init_tokenizer_(self.tokenizer)

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not self.train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        self.pooling = pooling
        self.loss = loss

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch
        device = ref_img.device
        
        if self.train_vit:
            ref_img = ref_img.to(torch.float32)
            ref_img_embeds = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embeds = self.ln_vision(self.visual_encoder(ref_img))
                
        ref_img_atts = torch.ones(ref_img_embeds.size()[:-1]).to(device)

        query_tokens = self.query_tokens.expand(ref_img_embeds.shape[0], -1, -1)
        

        # Encode the target image
        tar_img_feat = tar_feat.to(device)

        if self.pooling == "max":
            tar_img_feat_pool, _ = torch.max(tar_img_feat, dim=1)

        elif self.pooling == "mean":
            tar_img_feat_pool = torch.mean(tar_img_feat, dim=1)

        # Image-text Matching
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # Try the following if yours does't work. If both work please evaluate the difference between the two and add it to the report.
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #    self.device
        #)
        #My on attention_mask : text_tokens.attention_mask
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
            encoder_hidden_states=ref_img_embeds,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )

        query_feat = output.last_hidden_state[:, : query_tokens.size(1), :]
        
        query_feat = F.normalize(self.text_proj(query_feat), dim=-1) 
        
        if self.pooling == "max":
            query_feat_pool, _ = torch.max(query_feat, dim=1)

        elif self.pooling == "mean":
            query_feat_pool = torch.mean(query_feat, dim=1)


        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat_pool = fabric.all_gather(query_feat_pool, sync_grads=True)
            #print(query_feat_max.shape)
            #query_feat = einops.rearrange(query_feat_max, "d b e -> (d b) e")

            tar_img_feat_pool = fabric.all_gather(tar_img_feat_pool, sync_grads=True)
            #tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")
        return self.loss(query_feat_pool, tar_img_feat_pool, self.temp)


def blip2_cir(model, **kwargs):
    return model


def init_tokenizer_(tokenizer):
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer