# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.builder import MODELS
from models.tcl.clip_builder import get_clip_textenc
from models.tcl.encoders import CLIPImageFeatureEncoder
from models.tcl.mi import InfoNCE, ExtendedInfoNCE
from models.tcl.pamr import PAMR
from models.tcl.masker import Masker, Sim2Mask
from collections import OrderedDict
import us
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d):  # 仅对线性层参数进行初始化，可以根据需要修改
        nn.init.normal_(m.weight.data)  # 使用Xavier正态分布初始化权重
        nn.init.constant_(m.bias.data, 0.1)  # 使用常数初始化偏置


def get_similarity_map(sm, shape=None):

    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

    # # reshape
    # side = int(sm.shape[1] ** 0.5) # square output
    # sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

    # # interpolate
    # sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    # sm = sm.permute(0, 2, 3, 1)
    
    return sm


def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):

    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # weights to restrain influence of obvious classes on others
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)

    return similarity


def tv_loss(x):
    """Total variation loss

    Args:
        x: 4-d tensor [*, *, H, W]
    """
    return (
        (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() +
        (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    )


class AreaTCLLoss:
    def __init__(self, prior: float):
        self.prior = prior

    def __call__(self, mask: torch.Tensor):
        return (mask.mean() - self.prior).abs()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg


        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


@MODELS.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
            pred_sigmoid = pred.sigmoid()
            target = target.type_as(pred)
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
            loss = loss.mean()
            return loss


@MODELS.register_module()
class Classification(nn.Module):
    # def __init__(self, T_init=0.07, T_learnable=True):
    #     super().__init__()
    #     self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
    #     if not T_learnable:
    #         self.logit_scale.requires_grad_(False)
    #     self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss()

    def __init__(self, init_w=1.0, init_b=0.0, learnable=True, gumbel_tau=1.0):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.learnable = learnable

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b
        
        # self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss()
        # self.tagging_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)
        # self.focalloss = FocalLoss(alpha=0.75, gamma=0.1)

    def forward(self, image_emb, text_emb, labels):
        # all_labels = us.gather_cat(labels) # N, K
        # labelset = torch.nonzero(all_labels.sum(dim=0))[:, 0] # K
        # text_emb = text_emb[labelset]
        # labels = labels[:, labelset]
        image_emb = us.normalize(image_emb, dim=-1) # N, D
        text_emb = us.normalize(text_emb, dim=-1)  # 10000, D
        logits_per_img = image_emb @ text_emb.t() # N * 10000
        # logits_per_img = torch.einsum('ntd,md->ntm', image_emb, text_emb)
        # logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_img = logits_per_img * self.w + self.b
        # logits_per_img = torch.sigmoid(logits_per_img).mean(dim=1)
        # loss = F.binary_cross_entropy(logits_per_img, labels)
        # loss = self.binary_cross_entropy_with_logits(logits_per_img, labels) 
        # loss = self.tagging_loss_function(logits_per_img, labels) 
        # loss = self.focalloss(logits_per_img, labels) 
        preds = logits_per_img.softmax(dim=-1)
        labels = F.normalize(labels, dim=-1, p=1)
        loss = -(preds.clamp(1e-8).log() * labels).sum(-1).mean()
        # loss = F.cross_entropy(logits_per_img * logit_scale, labels)
        return loss


@MODELS.register_module()
class PatchClassification(nn.Module):
    # def __init__(self, T_init=0.07, T_learnable=True):
    #     super().__init__()
    #     self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
    #     if not T_learnable:
    #         self.logit_scale.requires_grad_(False)
    #     self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss()

    def __init__(self, init_w=1.0, init_b=0.0, learnable=True, gumbel_tau=1.0):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.learnable = learnable

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, image_emb, text_emb, labels):
        image_emb = image_emb.view(image_emb.size(0), image_emb.size(1), -1)
        logits_per_patch = torch.einsum('ndl,cd->nlc', image_emb, text_emb) # N, L, C
        d = image_emb.size(1)
        scale = d ** 0.5
        pred_per_patch = torch.softmax(logits_per_patch / scale, dim=1) # N, L, C
        feat_per_class = torch.einsum('nlc,ndl->ncd', pred_per_patch, image_emb) # N, C, D
        feat_per_class = us.normalize(feat_per_class, dim=-1) # N, C, D
        text_emb = us.normalize(text_emb, dim=-1)  # C, D
        logits_per_img = torch.einsum('ncd,cd->nc', feat_per_class, text_emb)
        logits_per_img = logits_per_img * self.w + self.b
        preds = logits_per_img.softmax(dim=-1)
        labels = F.normalize(labels, dim=-1, p=1)
        loss = -(preds.clamp(1e-8).log() * labels).sum(-1).mean()
        return loss


class Embedding(nn.Module):
    """Apply layer norm & projection for 1d or 2d inputs.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.main_branch = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.in_dim, self.out_dim))
        # self.residual_branch = nn.Conv2d(self.in_dim, self.out_dim, 1)
        self._initialize_weights() 

    def _initialize_weights(self):
        for m in self.main_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # for m in self.residual_branch.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.eye_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.main_branch(x) # + 0.5 * self.residual_branch(x)


@MODELS.register_module()
class TCL(nn.Module):
    def __init__(
        self, clip_model, ie_freeze, ie_ignore_last_attn, masker,
        tcl_w, area_w, tv_w, label_file
    ):
        super().__init__()
        self.pamr = None  # lazy init

        self.clip_text_encoder = get_clip_textenc(clip_model)
        assert ie_freeze >= 1, f"for now, ie_freeze >= 1 is required, but {ie_freeze} is given."
        self.clip_image_encoder = CLIPImageFeatureEncoder(
            clip_model,
            feature_extract_index=ie_freeze-1,
            ignore_last_attn=ie_ignore_last_attn,
        )
        self.patch_size = self.clip_image_encoder.patch_size

        # self.mask_emb = nn.Parameter(torch.randn((1, 1, self.clip_image_encoder.clip_visual.embed_dim)), requires_grad=True)

        image_proj = self.clip_image_encoder.clone_proj()
        output_dim = self.clip_image_encoder.clip_visual.output_dim if self.clip_image_encoder.clip_visual.proj is not None else self.clip_image_encoder.clip_visual.embed_dim
        decoder_cfg = masker['decoder']
        decoder_cfg["C"] = output_dim
        decoder = MODELS.build(decoder_cfg)
        decoder = nn.Sequential(OrderedDict([
            ("decoder", decoder),
            ("image_proj", image_proj)
        ]))
        self.decoder = decoder
        # self.visual_mlp = Embedding(self.clip_image_encoder.clip_visual.output_dim, self.clip_image_encoder.clip_visual.output_dim)

        image_proj_bar = self.clip_image_encoder.clone_proj()
        decoder_bar = MODELS.build(decoder_cfg)
        decoder_bar = nn.Sequential(OrderedDict([
            ("decoder", decoder_bar),
            ("image_proj", image_proj_bar)
        ]))
        self.decoder_bar = decoder_bar
        # self.visual_mlp_bar = Embedding(self.clip_image_encoder.clip_visual.output_dim, self.clip_image_encoder.clip_visual.output_dim)

        self.vit = self.clip_image_encoder.clip_visual
        
        # self.text_mlp = Embedding(self.clip_image_encoder.clip_visual.output_dim, self.clip_image_encoder.clip_visual.output_dim)

        # masker_backbone = self.clip_image_encoder.clone_masker_backbone(ie_freeze)
        # masker_backbone.patch_size = self.patch_size
        # image_proj = self.clip_image_encoder.clone_proj()
        # self.masker = Masker(
        #     backbone=masker_backbone,
        #     image_proj=image_proj,
        #     ignore_last_attn=ie_ignore_last_attn,
        #     **masker
        # )
        # self.sim2mask = Sim2Mask(**masker['sim2mask'])

        self.tv_w = tv_w
        self.tv_loss = ExtendedInfoNCE() if tv_w else None

        self.tcl_w = tcl_w
        self.tcli_loss = InfoNCE() if tcl_w else None

        self.area_w = area_w
        self.area_loss = Classification(**masker['sim2mask']) if area_w else None

        self.label_embedding = nn.Parameter(torch.load(label_file, map_location='cpu').float(), requires_grad=False)

        # self.area_w = area_w
        # self.area_loss = AreaTCLLoss(0.4)
        # self.neg_area_loss = AreaTCLLoss(0.0)

        # self.tv_w = tv_w
        # self.ust = False

    def train(self, mode=True):
        """Override the default train() to freeze CLIP backbone
        """
        super().train(mode)
        # CLIP encoders are always frozen
        self.clip_image_encoder.eval()
        self.clip_text_encoder.eval()
        self.decoder.train()
        self.decoder_bar.train()
        # self.mask_emb.requires_grad_(True)
        # self.visual_mlp.train()
        # self.visual_mlp_bar.train()
        # self.text_mlp.train()


    def set_train(self, decoder_only: bool, config):
        """Update requires_grad_ and train/eval mode by `decoder_only` flag.
        """

        # set train mode
        self.train()

        # freeze clip encoders
        self.clip_image_encoder.requires_grad_(False)
        self.clip_text_encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)
        self.decoder_bar.requires_grad_(True)
        # self.mask_emb.requires_grad_(True)
        # self.visual_mlp.requires_grad_(True)
        # self.visual_mlp_bar.requires_grad_(True)
        # self.text_mlp.requires_grad_(True)


    def masked_pool(self, spatial_image_emb, mask, eps=1e-6):
        """Average pool spatial_image_emb with mask

        Args:
            spatial_image_emb [BCHW]: spatial embedding
            mask [BNHW]: hard or soft mask

        Return:
            image_emb [BNC] : mask-pooled tensor
        """
        mask_sum = mask.sum((2,3), keepdim=True)  # [BN11]
        weight = mask / (mask_sum + eps)
        masked_image_emb = torch.einsum("bchw,bnhw->bnc", spatial_image_emb, weight)  # [BNC]

        return masked_image_emb
    

    def forward(self, image, text, tag):
        # key of loss should have `loss` string (key w/o `loss` is not accumulated for final loss).
        ret = {}  # losses + logs

        H, W = image.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size

        # forward CLIP & extract features
        # clip_image_feats, clip_image_feats_bar, mask = self.clip_image_encoder.maskclip_forward(image, ret_feats=False, mask_emb=self.mask_emb)
        clip_image_feats = self.clip_image_encoder.maskclip_forward(image, ret_feats=False)
        clip_image_feats_bar = clip_image_feats

        clip_image_feats = rearrange(clip_image_feats[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
        clip_image_feats = self.decoder(clip_image_feats)
        # image_feat = clip_image_feats.mean(dim=-1).mean(dim=-1)
        # clip_image_feats = self.visual_mlp(clip_image_feats)
        # clip_image_feats = F.normalize(clip_image_feats, p=2, dim=1)
        image_feat = clip_image_feats.mean(dim=-1).mean(dim=-1)
        # image_feat1 = self.visual_mlp(image_feat)

        clip_image_feats_bar = rearrange(clip_image_feats_bar[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
        clip_image_feats_bar = self.decoder_bar(clip_image_feats_bar)
        # image_feat_bar = clip_image_feats_bar.mean(dim=-1).mean(dim=-1)
        # clip_image_feats_bar = self.visual_mlp_bar(clip_image_feats_bar)
        # clip_image_feats_bar = F.normalize(clip_image_feats_bar, p=2, dim=1)
        image_feat_bar = clip_image_feats_bar.mean(dim=-1).mean(dim=-1)
        # image_feat_bar1 = self.visual_mlp_bar(image_feat_bar)

        with torch.no_grad():
            text_emb = self.clip_text_encoder(text)

        # text_emb = self.text_mlp(text_emb)
        # if self.tv_loss is not None:
        #     tv_loss = self.tv_loss(clip_image_feats, text_emb)  # ExtendedInfoNCE
        #     ret["tv_loss"] = tv_loss * self.tv_w

        if self.tcli_loss is not None: # infornce 
            tcli_loss = self.tcli_loss(image_feat, text_emb) # + self.tcli_loss(image_feat_bar, text_emb)
            ret["tcli_loss"] = tcli_loss * self.tcl_w

        if self.area_loss is not None:
            area_loss = self.area_loss(image_feat, self.label_embedding.data, tag)
            area_loss_bar = self.area_loss(image_feat_bar, self.label_embedding.data, tag)
            ret["area_loss"] = area_loss * self.area_w + area_loss_bar * self.area_w

        return ret

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
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1)
        text_embs = us.normalize(text_embs, dim=-1)

        return text_embs

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    @torch.no_grad()
    def forward_seg(self, image_emb, text_emb, deterministic=True, hard=False):
        """Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, C]
            deterministic (bool): deterministic inference flag for gumbel noise
            hard (bool): decide hard or soft returning segmentation mask.
                Note that soft mask is required for proper evaluation

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        image_emb = us.normalize(image_emb, dim=1)  # BCHW
        text_emb = us.normalize(text_emb, dim=-1)  # NC

        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)

        # hard_mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)
        # mask = hard_mask if hard else soft_mask
        # # mask = hard_mask
        mask = torch.sigmoid(10 * simmap - 2.5)

        return mask, simmap

    # @torch.no_grad()
    # def generate_masks(
    #     self, image, text_emb, text_is_token=False, apply_pamr=False,
    #     kp_w=0.3,
    # ):
    #     """Generate masks for each text embeddings

    #     Args:
    #         image [B, 3, H, W]
    #         text_emb [N, C]

    #     Returns:
    #         softmask [B, N, H, W]: softmasks for each text embeddings
    #     """
    #     if text_is_token:
    #         text_emb = self.clip_text_encoder(text_emb)

    #     H, W = image.shape[2:]  # original image shape

    #     # # pad image when (image_size % patch_size != 0)
    #     # pad = self.compute_padsize(H, W, self.patch_size)
    #     # if any(pad):
    #     #     image = F.pad(image, pad)  # zero padding

    #     # # padded image size
    #     # pH, pW = image.shape[2:]

    #     ############### Generate mask ################
    #     # soft mask
    #     clip_image_feats = self.clip_image_encoder.maskclip_forward(image, ret_feats=False)

    #     h = (H - self.patch_size) // self.patch_size + 1
    #     w = (W - self.patch_size) // self.patch_size + 1

    #     clip_image_feats = us.normalize(clip_image_feats, dim=-1)  # BCHW
    #     text_emb = us.normalize(text_emb, dim=-1)  # NC

    #     simmap = clip_feature_surgery(clip_image_feats, text_emb)
    #     # simmap = get_similarity_map(simmap[:, 1:])
    #     simmap = simmap[:, 1:]

    #     simmap = simmap.reshape(simmap.shape[0], h, w, -1).permute(0, 3, 1, 2)

    #     # _, mask = self.sim2mask(simmap)
    #     mask = torch.sigmoid(simmap)

    #     # refinement
    #     if apply_pamr:
    #         mask = self.apply_pamr(image, mask)

    #     # resize
    #     mask = F.interpolate(mask, (H, W), mode='bilinear')  # [B, N, H, W]

    #     assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

    #     return mask, simmap
    
    @torch.no_grad()
    def generate_masks(
        self, image, text_emb, text_is_token=False, apply_pamr=False,
        kp_w=0.3,
    ):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]
            text_emb [N, C]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """
        if text_is_token:
            text_emb = self.clip_text_encoder(text_emb)

        H, W = image.shape[2:]  # original image shape

        # # pad image when (image_size % patch_size != 0)
        # pad = self.compute_padsize(H, W, self.patch_size)
        # if any(pad):
        #     image = F.pad(image, pad)  # zero padding

        # # padded image size
        # pH, pW = image.shape[2:]

        ############### Generate mask ################
        # soft mask
        # clip_image_feats = self.clip_image_encoder.maskclip_forward(image, ret_feats=False)
        # # image_feat = clip_image_feats[:, 0]
        # clip_image_feats = clip_image_feats[:, 1:]

        # h = (H - self.patch_size) // self.patch_size + 1
        # w = (W - self.patch_size) // self.patch_size + 1
        # stride = self.clip_image_encoder.clip_visual.conv1.stride
        # h = (H - self.patch_size) // stride[0] + 1
        # w = (W - self.patch_size) // stride[1] + 1

        # clip_image_feats = rearrange(clip_image_feats, "B (H W) C-> B C H W", H=h, W=w)

        h = H // self.patch_size
        w = W // self.patch_size

        clip_image_feats = self.clip_image_encoder.maskclip_forward(image, ret_feats=False)

        clip_image_feats = rearrange(clip_image_feats[:, 1:], "B (H W) C -> B C H W", H=h, W=w)

        clip_image_feats_bar = self.decoder_bar(clip_image_feats)
        clip_image_feats = self.decoder(clip_image_feats)
        # print('0 is', self.decoder[0].net[0].conv.conv.bias.data[:10])
        # print('1 is', self.decoder_bar[0].net[0].conv.conv.bias.data[:10])
        # print(clip_image_feats_bar[0, 10, 10, :10])
        # print(clip_image_feats[0, 10, 10, :10])
        # exit() 
        # B * 20 * H  * W

        mask, simmap = self.forward_seg(clip_image_feats, text_emb, hard=False)  # [B, N, H', W']
        mask_bar, simmap_bar = self.forward_seg(clip_image_feats_bar, text_emb, hard=False)  # [B, N, H', W']

        # mask = mask_bar ##single
        # simmap = (simmap + simmap_bar) / 2
        # mask = torch.sigmoid(10 * simmap - 2.5)

        # refinement
        if apply_pamr:
            mask = self.apply_pamr(image, mask)

        # if kp_w:
        #     mask = self.kp_branch(img_emb, text_emb, mask, kp_w=kp_w)
        # ##############################################

        # resize
        mask = F.interpolate(mask, (H, W), mode='bilinear')  # [B, N, H, W]

        # # mask cutting for padded image
        # if any(pad):
        #     l, t = pad[0], pad[2]
        #     mask = mask[:, :, t:t+H, l:l+W]

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

        return mask, simmap
    
    # @torch.no_grad()
    # def generate_masks(
    #     self, image, text_emb, text_is_token=False, apply_pamr=False,
    #     kp_w=0.3,
    # ):
    #     """Generate masks for each text embeddings

    #     Args:
    #         image [B, 3, H, W]
    #         text_emb [N, C]

    #     Returns:
    #         softmask [B, N, H, W]: softmasks for each text embeddings
    #     """
    #     if text_is_token:
    #         text_emb = self.clip_text_encoder(text_emb)

    #     H, W = image.shape[2:]  # original image shape

    #     # pad image when (image_size % patch_size != 0)
    #     pad = self.compute_padsize(H, W, self.patch_size)
    #     if any(pad):
    #         image = F.pad(image, pad)  # zero padding

    #     # padded image size
    #     pH, pW = image.shape[2:]

    #     ############### Generate mask ################
    #     # soft mask
    #     img_emb, clip_image_feats = self.clip_image_encoder.tcl_forward(image, ret_feats=True)
    #     image_feat = clip_image_feats[0]
    #     clip_image_feats = clip_image_feats[1:]
    #     mask, simmap = self.masker.forward_seg(image, image_feat, text_emb, hard=False)  # [B, N, H', W']

    #     # refinement
    #     if apply_pamr:
    #         mask = self.apply_pamr(image, mask)

    #     if kp_w:
    #         mask = self.kp_branch(img_emb, text_emb, mask, kp_w=kp_w)
    #     ##############################################

    #     # resize
    #     mask = F.interpolate(mask, (pH, pW), mode='bilinear')  # [B, N, H, W]

    #     # mask cutting for padded image
    #     if any(pad):
    #         l, t = pad[0], pad[2]
    #         mask = mask[:, :, t:t+H, l:l+W]

    #     assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

    #     return mask, simmap

    def kp_branch(self, clip_feat, text_emb, org_mask, kp_w):
        assert self.masker.ignore_last_attn, "KP branch is only implemented for ignore_last_attn=True"
        image_emb = self.clip_image_encoder.clip_proj(clip_feat)

        image_emb = us.normalize(image_emb, dim=1)  # BCHW
        text_emb = us.normalize(text_emb, dim=-1)  # NC

        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)

        # kp mask
        mask = torch.sigmoid((simmap - 0.25) * 10.0)
        mask = F.interpolate(mask , org_mask.shape[2:], mode='bilinear')

        # mix
        mask = kp_w * mask + (1. - kp_w) * org_mask

        return mask
