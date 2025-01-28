import torch
from torch import nn


from lavis.models import load_model_and_preprocess
class BLIP2_Embs(nn.Module):
    def __init__(
        self,
        model_type ="pretrain",
        name = "blip2_feature_extractor",
        embed_dim=256,
        queue_size=57600,
        negative_all_rank=False,
        is_eval = True,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=name, model_type=model_type,is_eval = is_eval)
        self.visual_encoder = self.model.visual_encoder
        self.embed_dim = embed_dim

        self.queue_size = queue_size
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.negative_all_rank = negative_all_rank


def blip2_embs(pretrained="", **kwargs):
    model = BLIP2_Embs(**kwargs)
    return model
