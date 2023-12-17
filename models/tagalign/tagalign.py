import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.builder import MODELS
from models.tagalign.clip_builder import get_clip_textenc
from models.tagalign.encoders import CLIPImageFeatureEncoder
from models.tagalign.loss import InfoNCE, MultiClassificationLoss
from collections import OrderedDict
import us
from timm.models.vision_transformer import Block


@MODELS.register_module()
class TagAlign(nn.Module):
    def __init__(
        self, clip_model, ie_ignore_last_attn, use_clip_surgery, projector, cl_w, ce_w, label_file
    ):
        super().__init__()

        self.clip_text_encoder = get_clip_textenc(clip_model)
        self.clip_image_encoder = CLIPImageFeatureEncoder(
            clip_model,
            ignore_last_attn=ie_ignore_last_attn,
            use_clip_surgery=use_clip_surgery,
        )
        self.patch_size = self.clip_image_encoder.patch_size
        self.embed_dim = self.clip_image_encoder.clip_visual.embed_dim
        self.projector_cfg = projector
        if projector['type'] == 'GDecoder':
            clip_proj = self.clip_image_encoder.clone_proj()
            projector["C"] = self.embed_dim
            self.projector = MODELS.build(projector)
            self.projector = nn.Sequential(OrderedDict([
                ("projector", self.projector),
                ("clip_proj", clip_proj)
                ]))
        else:
            clip_proj = self.clip_image_encoder.clone_proj()
            n_layers = projector['n_layers']
            n_heads = projector['n_heads']
            mlp_ratio = projector['mlp_ratio']
            block_list = [
                Block(self.embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(n_layers)]
            self.projector = nn.Sequential(*block_list)
            self.projector = nn.Sequential(OrderedDict([
                ("projector", self.projector),
                ("clip_proj", clip_proj)
                ]))

        self.label_embedding = nn.Parameter(torch.load(label_file, map_location='cpu').float(), requires_grad=False)

        self.cl_w = cl_w
        self.cl_loss = InfoNCE() if cl_w else None

        self.ce_w = ce_w
        self.ce_loss = MultiClassificationLoss() if cl_w else None

        self.class_weights = None


    def train(self, mode=True):
        """Override the default train() to freeze CLIP backbone
        """
        super().train(mode)
        # CLIP encoders are always frozen
        self.clip_image_encoder.eval()
        self.clip_text_encoder.eval()
        self.projector.train()


    def set_train(self, ):
        # set train mode
        self.train()
        # freeze clip encoders
        self.clip_image_encoder.requires_grad_(False)
        self.clip_text_encoder.requires_grad_(False)
        self.projector.requires_grad_(True)
    

    def forward(self, image, text, tag):
        losses = {} 

        clip_image_feats = self.clip_image_encoder.tagalign_forward(image)
        if self.projector_cfg['type'] == 'GDecoder':
            H, W = image.shape[-2:]
            h = H // self.patch_size
            w = W // self.patch_size
            patch_feats = rearrange(clip_image_feats[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
            patch_feats = self.projector(patch_feats)
            image_feat = patch_feats.mean(dim=(2, 3))
        else:
            patch_feats = clip_image_feats[:, 1:]
            patch_feats = self.projector(patch_feats)
            image_feat = patch_feats.mean(dim=1)

        with torch.no_grad():
            text_emb = self.clip_text_encoder(text)

        if self.cl_loss is not None: 
            cl_loss = self.cl_loss(image_feat, text_emb) 
            losses["cl_loss"] = cl_loss * self.cl_w

        if self.ce_loss is not None:
            ce_loss = self.ce_loss(image_feat, self.label_embedding.data, tag, self.class_weights)
            losses["ce_loss"] = ce_loss * self.ce_w 

        return losses

    @torch.no_grad()
    def build_text_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 1024
        N = text.size(0)
        text_embs = torch.cat([
            self.clip_text_encoder(text[i:i+chunk_size])
            for i in range(0, N, chunk_size)
        ])
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        text_embs = text_embs.mean(dim=1)
        return text_embs

    @torch.no_grad()
    def generate_masks(
        self, image, text_emb, clip_w=0.3, scale=10, bias=-2.5
    ):
        H, W = image.shape[2:]  
        h = H // self.patch_size
        w = W // self.patch_size

        clip_image_feats = self.clip_image_encoder(image)
        if self.projector_cfg['type'] == 'GDecoder':
            patch_feats = rearrange(clip_image_feats[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
            patch_feats = self.projector(patch_feats)
        else:
            patch_feats = clip_image_feats[:, 1:]
            patch_feats = self.projector(patch_feats)
            patch_feats = rearrange(patch_feats, "B (H W) C -> B C H W", H=h, W=w)
        
        text_emb = us.normalize(text_emb, dim=-1)  # NC

        patch_feats = us.normalize(patch_feats, dim=1)  # BCHW
        simmap = torch.einsum("b c h w, n c -> b n h w", patch_feats, text_emb)
        mask = torch.sigmoid(simmap * scale + bias)

        if clip_w > 0:
            patch_feats = rearrange(clip_image_feats[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
            patch_feats = self.clip_image_encoder.clip_proj(patch_feats)
            patch_feats = us.normalize(patch_feats, dim=1)  # BCHW
            clip_simmap = torch.einsum("b c h w, n c -> b n h w", patch_feats, text_emb)
            clip_mask = torch.sigmoid(clip_simmap * scale + bias) 
            mask = (1 - clip_w) * mask + clip_w * clip_mask

        # resize
        mask = F.interpolate(mask, (H, W), mode='bilinear')  # [B, N, H, W]
        return mask
