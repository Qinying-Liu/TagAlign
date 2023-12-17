import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.builder import MODELS
import us


@MODELS.register_module()
class InfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb [B, C]: image embedding
            text_emb [B, C]: text embedding
        """
        assert image_emb.ndim == text_emb.ndim == 2

        B = image_emb.shape[0]
        # get label globally
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()

        # [B, C]
        image_emb = us.normalize(image_emb, dim=-1)
        text_emb = us.normalize(text_emb, dim=-1)

        # cosine similarity
        logits_per_img = image_emb @ us.gather_cat(text_emb, grad=True).t()
        logits_per_text = text_emb @ us.gather_cat(image_emb, grad=True).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss


@MODELS.register_module()
class MultiClassificationLoss(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb, labels, weights=1):
        image_emb = us.normalize(image_emb, dim=-1) # N, D
        text_emb = us.normalize(text_emb, dim=-1)  # 10000, D
        logits_per_img = image_emb @ text_emb.t() # N * 10000

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_img = logits_per_img * logit_scale

        # if weights is not None:
        #     preds = logits_per_img.exp()
        #     weights = torch.tensor(weights, dtype=preds.dtype, device=preds.device)
        #     preds = preds * weights
        #     preds = F.normalize(preds, dim=-1, p=1)
        # else:
        preds = logits_per_img.softmax(dim=-1)

        labels = F.normalize(labels, dim=-1, p=1)
        loss = -(preds.clamp(1e-8).log() * labels).sum(-1).mean()
        return loss
