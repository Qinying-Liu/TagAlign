import copy
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange

from models.builder import MODELS
from models.tagalign.clip_builder import get_clip_imgenc, get_clip_textenc
from models.tagalign.modules import FeatureEncoder, BLCModuleCompatibleBCHW


class LNProjLayer(BLCModuleCompatibleBCHW):
    """Apply layer norm & projection for 1d or 2d inputs.
    """
    def __init__(self, ln: Union[None, nn.LayerNorm], proj: Union[None, torch.Tensor]):
        super().__init__()
        self.ln = ln
        self.proj = proj

    def forward_blc(self, x):
        if self.ln is not None:
            x = self.ln(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


@MODELS.register_module()
class CLIPImageFeatureEncoder(FeatureEncoder):
    def clone_proj(self):
        return copy.deepcopy(self.clip_proj)

    def __init__(self, model_name: str, ignore_last_attn: bool, use_clip_surgery: bool):
        super().__init__()
        # build clip_visual
        clip_visual = get_clip_imgenc(model_name, use_clip_surgery=use_clip_surgery)

        self.clip_proj = LNProjLayer(clip_visual.ln_post, clip_visual.proj)

        clip_visual.ln_post = nn.Identity()
        clip_visual.proj = None

        self.clip_visual = clip_visual
        self.patch_size = self.clip_visual.patch_size
        self.output_dim = self.clip_visual.output_dim
        self.ignore_last_attn = ignore_last_attn

    def _encode(self, x, spatial=True, ignore_last_attn=None):
        if ignore_last_attn is None:
            ignore_last_attn = self.ignore_last_attn

        x = self.clip_visual(
            x,
            spatial=spatial,
            ignore_last_attn=ignore_last_attn
        )  # [B, L, C]
        return x
    
    @torch.no_grad()
    def tagalign_forward(self, x):
        x = self.forward(x, spatial=True, ignore_last_attn=self.ignore_last_attn)
        return x
    

@MODELS.register_module()
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.clip_text = get_clip_textenc(model_name)

    def forward(self, x):
        return self.clip_text(x)

