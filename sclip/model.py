# # ------------------------------------------------------------------------------
# # TCL
# # Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# # ------------------------------------------------------------------------------
# # Modified from CLIP (https://github.com/openai/CLIP)
# # Copyright (c) 2021 OpenAI. All Rights Reserved.
# # ------------------------------------------------------------------------------
# from collections import OrderedDict
# from typing import Tuple, Union

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn
# from einops import rearrange


# def interpolate_pos_emb(pos_emb, H, W):
#     """
#     Args:
#         pos_emb [n_patches, emb_dim]
#         H, W: target shape
#     """
#     N = pos_emb.size(0) - 1
#     if H*W == N and W == H:
#         return pos_emb
#     pe_cls, pe_grid = pos_emb[:1], pos_emb[1:]
#     size = int(N ** 0.5)
#     # interpolate original size -> target size
#     pe_grid = F.interpolate(
#         rearrange(pe_grid, "(h w) d -> () d h w", h=size, w=size),
#         size=(H, W),
#         mode="bicubic",
#         align_corners=False
#     ).squeeze(0)
#     pe_grid = rearrange(pe_grid, "d th tw -> (th tw) d")
#     pe = torch.cat([pe_cls, pe_grid])

#     return pe


# def make_diag(N, device):
#     mask = torch.ones(N, N, dtype=torch.bool, device=device).fill_diagonal_(False)

#     return mask


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1):
#         super().__init__()

#         # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
#         self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = None
#         self.stride = stride

#         if stride > 1 or inplanes != planes * Bottleneck.expansion:
#             # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
#             self.downsample = nn.Sequential(OrderedDict([
#                 ("-1", nn.AvgPool2d(stride)),
#                 ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
#                 ("1", nn.BatchNorm2d(planes * self.expansion))
#             ]))

#     def forward(self, x: torch.Tensor):
#         identity = x

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.avgpool(out)
#         out = self.bn3(self.conv3(out))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x, spatial=False):
#         H, W = x.shape[2:]
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         pos_emb = interpolate_pos_emb(self.positional_embedding, H, W)
#         x = x + pos_emb[:, None, :].to(x.dtype)  # (HW+1)NC

#         if spatial:
#             mask = make_diag(x.shape[0], device=x.device)
#         else:
#             mask = None

#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False,
#             attn_mask=mask
#         )

#         if spatial:
#             # LNC -> NLC  (N=B, L=HW+1)
#             return x.permute(1, 0, 2)
#         else:
#             return x[0]


# class ModifiedResNet(nn.Module):
#     """
#     A ResNet class that is similar to torchvision's but contains the following changes:
#     - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#     - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#     - The final pooling layer is a QKV attention instead of an average pool
#     """

#     def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
#         super().__init__()
#         self.output_dim = output_dim
#         self.input_resolution = input_resolution

#         # the 3-layer stem
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.avgpool = nn.AvgPool2d(2)
#         self.relu = nn.ReLU(inplace=True)

#         # residual layers
#         self._inplanes = width  # this is a *mutable* variable used during construction
#         self.layer1 = self._make_layer(width, layers[0])
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

#         embed_dim = width * 32  # the ResNet feature dimension
#         self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

#     def _make_layer(self, planes, blocks, stride=1):
#         layers = [Bottleneck(self._inplanes, planes, stride)]

#         self._inplanes = planes * Bottleneck.expansion
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, spatial=False):
#         def stem(x):
#             for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
#                 x = self.relu(bn(conv(x)))
#             x = self.avgpool(x)
#             return x

#         x = x.type(self.conv1.weight.dtype)
#         x = stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.attnpool(x, spatial=spatial)

#         return x


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))  # original version
#         return ret.type(orig_type)


# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x


# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.heads = heads
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#         self.orig_attn_mask = attn_mask

#     def forward(self, x: torch.Tensor, ignore_last_attn: bool = False):
#         if ignore_last_attn:
#             mask = torch.empty(x.shape[0], x.shape[0])
#             mask.fill_(float('-inf'))
#             mask.fill_diagonal_(0)
#             self.ignore_last_attn_mask = mask

#             self.resblocks[-1].attn_mask = self.ignore_last_attn_mask
#         else:
#             self.resblocks[-1].attn_mask = self.orig_attn_mask

#         return self.resblocks(x)


# class VisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.width = width
#         self.patch_size = patch_size
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)

#         self.transformer = Transformer(width, layers, heads)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def forward(self, x: torch.Tensor, spatial: bool = False, ignore_last_attn=False):
#         """
#         Args:
#             spatial: return spatial feature map [N, L, D]
#             ignore_last_attn: apply maskclip trick or not
#         """
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         H, W = x.shape[2:]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         pos_emb = interpolate_pos_emb(self.positional_embedding, H, W).to(x.dtype)
#         x = x + pos_emb
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, ignore_last_attn=ignore_last_attn)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         if spatial:
#             x = self.ln_post(x)
#         else:
#             x = self.ln_post(x[:, 0, :])

#         if self.proj is not None:
#             x = x @ self.proj
#         return x


# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()

#         self.context_length = context_length

#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // 64
#             self.visual = VisionTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim
#             )

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     #  @property
#     #  def dtype(self):
#     #      return self.visual.conv1.weight.dtype

#     def encode_image(self, image, spatial=False):
#         dtype = self.visual.conv1.weight.dtype
#         return self.visual(image.type(dtype), spatial=spatial)

#     def encode_text(self, text):
#         dtype = self.token_embedding.weight.dtype
#         x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

#         x = x + self.positional_embedding.type(dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

#         return x

#     def forward(self, image, text):
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         # normalized features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text
# 
# 
# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)


# def build_model(state_dict: dict):
#     vit = "visual.proj" in state_dict

#     if vit:
#         vision_width = state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
#         vision_layers = tuple(counts)
#         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         vision_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32

#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

#     model = CLIP(
#         embed_dim,
#         image_resolution, vision_layers, vision_width, vision_patch_size,
#         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
#     )

#     for key in ["input_resolution", "context_length", "vocab_size"]:
#         if key in state_dict:
#             del state_dict[key]

#     convert_weights(model)
#     model.load_state_dict(state_dict)
#     return model.eval()


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


def interpolate_pos_emb(pos_emb, H, W):
    """
    Args:
        pos_emb [n_patches, emb_dim]
        H, W: target shape
    """
    N = pos_emb.size(0) - 1
    if H*W == N and W == H:
        return pos_emb
    pe_cls, pe_grid = pos_emb[:1], pos_emb[1:]
    size = int(N ** 0.5)
    # interpolate original size -> target size
    pe_grid = F.interpolate(
        rearrange(pe_grid, "(h w) d -> () d h w", h=size, w=size),
        size=(H, W),
        mode="bicubic",
        align_corners=False
    ).squeeze(0)
    pe_grid = rearrange(pe_grid, "d th tw -> (th tw) d")
    pe = torch.cat([pe_cls, pe_grid])

    return pe

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # resnets have only one self-attention, norm and larger scale perform better
        if self.settings == 'resnet':
            k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q = k
            scale = self.scale * 8
        else:
            scale = self.scale
        
        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # clip_surgery
        #x = v.transpose(1, 2).reshape(B, N, C) # mask_clip
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.attn = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim


    def forward(self, x):
        # reform transformer layer after init and load weights, using v only
        if self.attn == None:
            self.attn = Attention(self.output_dim, self.embed_dim, self.num_heads, True)
            self.attn.qkv.weight = torch.nn.Parameter(torch.cat([self.v_proj.weight, self.v_proj.weight, self.v_proj.weight], 0))
            self.attn.qkv.bias = torch.nn.Parameter(torch.cat([self.v_proj.bias, self.v_proj.bias, self.v_proj.bias]))
            self.attn.proj.weight = self.c_proj.weight
            self.attn.proj.bias = self.c_proj.bias

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[0] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, x_ori = self.attn(x.transpose(0, 1))

        # cls token from the original path, and img tokens from the new path
        x[:, 0, :] = x_ori[:, 0, :]
        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        # shape BNC
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):

        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.patch_size = patch_size

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, spatial=False, ignore_last_attn=False, mask_emb=None):

        # reform the architecture during first inference
        if self.attn == None:
            
            # apply architecture surgery on the last 6 blocks
            for i in range(1, 7): # surgery 7, maskclip 2
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        H, W = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        if mask_emb is not None:
            mask = self.random_masking(x, 0.5)
            mask_emb = mask_emb.expand_as(x)
            x_bar = mask * mask_emb + (1 - mask) * x
            
            x_bar = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x_bar.shape[0], 1, x_bar.shape[-1], dtype=x_bar.dtype, device=x_bar.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

            pos = interpolate_pos_emb(self.positional_embedding, H, W).to(x_bar.dtype)
            
            x_bar = x_bar + pos
            x_bar = self.ln_pre(x_bar)
            
            x_bar = x_bar.permute(1, 0, 2)  # NLD -> LND
            x_bar, x_ori_bar = self.transformer(x_bar)
            x_bar[0, :, :] = x_ori_bar[0, :, :] # clip_surgery
            x_bar = x_bar.permute(1, 0, 2)  # LND -> NLD
            
            x_bar = self.ln_post(x_bar)
            if self.proj is not None:
                x_bar = x_bar @ self.proj
            
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + pos
            x = self.ln_pre(x)
            
            x = x.permute(1, 0, 2)  # NLD -> LND
            x, x_ori = self.transformer(x)
            x[0, :, :] = x_ori[0, :, :] # clip_surgery
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            x = self.ln_post(x)
            if self.proj is not None:
                x = x @ self.proj

            return x, x_bar, mask
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos = interpolate_pos_emb(self.positional_embedding, H, W).to(x.dtype)
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, x_ori = self.transformer(x)
        x[0, :, :] = x_ori[0, :, :] # clip_surgery
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        return x
    
    @torch.no_grad()
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.unsqueeze(-1)

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text):
        dtype = self.token_embedding.weight.dtype
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    # def encode_text(self, text):
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    #     x = x + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x).type(self.dtype)

    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    #     return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()