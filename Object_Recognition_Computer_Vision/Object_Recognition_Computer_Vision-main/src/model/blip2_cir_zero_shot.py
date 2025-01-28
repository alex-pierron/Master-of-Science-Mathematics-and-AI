from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from lavis.models import load_model_and_preprocess

from src.model.med import BertModel
from src.tools.utils import print_dist


class BLIP2Cir(nn.Module):
    def __init__(
        self,
        loss: Any,
        name="blip2_feature_extractor",
        model_type = "pretrain",
        embed_dim=768,
        is_eval = False,
        train_vit = False
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=name, model_type=model_type,is_eval = is_eval)
        self.visual_encoder = self.model.visual_encoder
        self.embed_dim = embed_dim

        self.temp = torch.nn.Parameter(0.07 * torch.ones([]))


        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        #self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch

        device = ref_img.device
        """
        if self.train_vit:
            sample = {"image": ref_img, "text_input": None}
            ref_img_embs = self.model.extract_features(sample,mode="image")
        else:
            with torch.no_grad():
                sample = {"image": ref_img, "text_input": None}
                ref_img_embs = self.model.extract_features(sample,mode="image")
        """
        # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)
        #text = self.txt_processors['train'](caption).to(device)
        sample = {"image": ref_img, "text_input": caption}
        query_feat = self.model.extract_features(sample)
        query_feat  = query_feat.multimodal_embeds
        query_feat = F.normalize(query_feat[:, 0, :], dim=-1)

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")
        return self.loss(query_feat, tar_img_feat, self.temp)


def blip2_cir(model, **kwargs):
    return model
